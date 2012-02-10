import pycuda.autoinit
import pycuda.driver as driver

import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""

