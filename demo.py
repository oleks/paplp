import pycuda.autoinit
import pycuda.driver as driver

import numpy

from pycuda.compiler import SourceModule

kernel = SourceModule("""
__global__ void multiply(float *destination, float *x, float *y)
{
  const int i = threadIdx.x;
  destination[i] = x[i] * y[i];
}
""")

multiply = kernel.get_function("multiply")
a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

destination = numpy.zeros_like(a)
multiply(driver.Out(destination), driver.In(a), driver.In(b), block=(400, 1, 1), grid=(1,1))

print destination-a*b
