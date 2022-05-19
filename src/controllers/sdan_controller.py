from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class SDANMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent_dict = {
            'tutor': None,
            'trainee': None
        }

        tutor_input_shape, trainee_input_shape = self._get_input_shape(scheme)
        self._build_agents(tutor_input_shape, trainee_input_shape)
        self.agent_output_type = args.agent_output_type

        self.trainee_action_selector = action_REGISTRY[args.trainee_action_selector](args)
        self.tutor_action_selector = action_REGISTRY[args.tutor_action_selector](args)
        self.action_selector = self.trainee_action_selector
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.agent_hidden_state_dict = {
            'tutor': None,
            'trainee': None
        }

    def select_actions(self, ep_batch, t_ep, t_env, agent='trainee', bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if agent == 'trainee':
            agent_outputs = self.forward(ep_batch, t_ep, 'trainee', test_mode=test_mode)
            return self.trainee_action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                                    test_mode=test_mode)
        elif agent == 'tutor':
            agent_outputs = self.forward(ep_batch, t_ep, 'tutor', test_mode=test_mode)
            return self.tutor_action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                                test_mode=test_mode)
        else:
            print("Wrong agent name for mac. Exiting.")
            exit()

    def forward(self, ep_batch, t, agent, test_mode=False):
        tutor_inputs, trainee_inputs = self._build_inputs(ep_batch, t)
        input_dict = {'tutor': tutor_inputs,
                      'trainee': trainee_inputs}
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.agent_hidden_state_dict[agent] = self.agent_dict[agent](input_dict[agent], self.agent_hidden_state_dict[agent])
        if agent == "trainee" and self.agent_output_type == 'pi_logits':
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        for agent in self.agent_dict.keys():
            self.agent_hidden_state_dict[agent] = self.agent_dict[agent].init_hidden()
            if self.agent_hidden_state_dict[agent] is not None:
                self.agent_hidden_state_dict[agent] = self.agent_hidden_state_dict[agent].unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self, agent):
        return self.agent_dict[agent].parameters()

    def load_state(self, other_mac):
        for agent in self.agent_dict.keys():
            self.agent_dict[agent].load_state_dict(other_mac.agent_dict[agent].state_dict())

    def cuda(self):
        for agent in self.agent_dict.keys():
            self.agent_dict[agent].cuda()

    def save_models(self, path):
        for agent in self.agent_dict.keys():
            th.save(self.agent_dict[agent].state_dict(), "{}/{}_agent.th".format(path, agent))

    def load_models(self, path):
        for agent in self.agent_dict.keys():
            self.agent_dict[agent].load_state_dict(th.load("{}/{}_agent.th".format(path, agent), map_location=lambda storage, loc: storage))

    def _build_agents(self, tutor_input_shape, trainee_input_shape):
        self.agent_dict['tutor'] = agent_REGISTRY[self.args.tutor_agent](tutor_input_shape, self.args)
        self.agent_dict['trainee'] = agent_REGISTRY[self.args.trainee_agent](trainee_input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        tutor_inputs = {'state': [], 'obs': []}
        tutor_inputs['state'].append(batch['state'][:, t].unsqueeze(1).repeat(1, self.args.n_agents, 1))
        tutor_inputs['obs'].append(batch["obs"][:, t])  # b1av
        trainee_inputs = []
        trainee_inputs.append(batch["obs"][:, t])  # b1av

        if self.args.obs_last_action:
            if t == 0:
                tutor_inputs['state'].append(th.zeros_like(batch["actions_onehot"][:, t]))
                tutor_inputs['obs'].append(th.zeros_like(batch["actions_onehot"][:, t]))
                trainee_inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                tutor_inputs['state'].append(batch["actions_onehot"][:, t - 1])
                tutor_inputs['obs'].append(th.zeros_like(batch["actions_onehot"][:, t - 1]))
                trainee_inputs.append(th.zeros_like(batch["actions_onehot"][:, t - 1]))
        if self.args.obs_agent_id:
            tutor_inputs['state'].append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            tutor_inputs['obs'].append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            trainee_inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        tutor_inputs['state'] = th.cat([x.reshape(bs, self.n_agents, -1) for x in tutor_inputs['state']], dim=-1)
        tutor_inputs['obs'] = th.cat([x.reshape(bs, self.n_agents, -1) for x in tutor_inputs['obs']], dim=-1)
        trainee_inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in trainee_inputs], dim=-1)
        return tutor_inputs, trainee_inputs

    def _get_input_shape(self, scheme):
        tutor_input_shape = dict()
        tutor_input_shape['state'] = scheme["state"]["vshape"]
        tutor_input_shape['obs'] = scheme["obs"]["vshape"]
        trainee_input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            tutor_input_shape['state'] += scheme["actions_onehot"]["vshape"][0]
            tutor_input_shape['obs'] += scheme["actions_onehot"]["vshape"][0]
            trainee_input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            tutor_input_shape['state'] += self.n_agents
            tutor_input_shape['obs'] += self.n_agents
            trainee_input_shape += self.n_agents

        return tutor_input_shape, trainee_input_shape
