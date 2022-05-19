REGISTRY = {}

from .basic_controller import BasicMAC
from .sdan_controller import SDANMAC
from .odan_controller import ODANMAC


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["sdan_mac"] = SDANMAC
REGISTRY["odan_mac"] = ODANMAC
