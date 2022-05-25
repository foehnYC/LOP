REGISTRY = {}

from .basic_controller import BasicMAC
from .slop_controller import SLOPMAC
from .olop_controller import OLOPMAC


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["slop_mac"] = SLOPMAC
REGISTRY["olop_mac"] = OLOPMAC
