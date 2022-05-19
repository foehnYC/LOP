import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import Mixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
from utils.th_utils import get_parameters_num
import torch.nn.functional as F


class NDANLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.tutor_params = list(mac.parameters('tutor'))
        self.trainee_params = list(mac.parameters('trainee'))

        if args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.tutor_params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.tutor_optimiser = Adam(params=self.tutor_params, lr=args.tutor_lr)
            self.trainee_optimiser = Adam(params=self.trainee_params, lr=args.trainee_lr)
        else:
            self.tutor_optimiser = RMSprop(params=self.tutor_params, lr=args.tutor_lr, alpha=args.optim_alpha,
                                           eps=args.optim_eps)
            self.trainee_optimiser = RMSprop(params=self.trainee_params, lr=args.trainee_lr, alpha=args.optim_alpha,
                                             eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.train_t = 0

        # th.autograd.set_detect_anomaly(True)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        trainee_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, agent='tutor')
            mac_out.append(agent_outs)
            trainee_outs = self.mac.forward(batch, t=t, agent='trainee')
            trainee_out.append(trainee_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        trainee_out = th.stack(trainee_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t, agent='tutor')
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -99999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])
            # targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals[:, 1:]

            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                                self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = L_td = masked_td_error.sum() / mask.sum()

        # Optimise Tutor
        self.tutor_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.tutor_params, self.args.grad_norm_clip)
        self.tutor_optimiser.step()

        # Compute and optimise trainee loss
        trainee_out[avail_actions == 0] = -9999
        target_distribution = F.softmax(mac_out_detach, dim=-1)[:, :-1]
        actions_probs = F.softmax(trainee_out, dim=-1)[:, :-1]
        cur_max_actions = cur_max_actions[:, :-1].squeeze(3)
        mask_actor = mask.expand_as(cur_max_actions)

        _, _, _, v = actions_probs.size()  # bs, num_env, num_agent, v
        actions_probs = actions_probs.reshape(-1, v)  # bs, num_env, num_agent, v
        target_distribution = target_distribution.reshape(-1, v)

        actor_loss = F.kl_div(target_distribution.log(), actions_probs, reduction='sum')
        self.trainee_optimiser.zero_grad()
        actor_loss.backward()
        th.nn.utils.clip_grad_norm_(self.trainee_params, self.args.grad_norm_clip)
        self.trainee_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("loss_ce", actor_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            # self.logger.log_stat("accuracy", accuracy, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.tutor_optimiser.state_dict(), "{}/tutor_opt.th".format(path))
        th.save(self.trainee_optimiser.state_dict(), "{}/trainee_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.tutor_optimiser.load_state_dict(
            th.load("{}/tutor_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.trainee_optimiser.load_state_dict(
            th.load("{}/trainee_opt.th".format(path), map_location=lambda storage, loc: storage))
