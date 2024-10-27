from __future__ import annotations
from typing import NamedTuple, Any, Callable, Literal, Generator, List

import math
import random
import numpy as np

from shape_utils import *
from math_utils import *
from type_utils import *
from engine_utils import *
from dsl_utils import *
from flow_utils import *
from calc_utils import *
from assert_utils import *

# # Special import, should never overlap or be used
# try:
#     from _engine_utils_mi import I0BpHzM2Xn_primitive_call_from_minecraft
# except:
#     print("Warning: was not able to import I0BpHzM2Xn_primitive_call_from_minecraft. This may cause issues if you are attempting to run a Mitsuba-translated version of a Minecraft DSL output.")