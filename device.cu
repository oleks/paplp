#include <iostream>
#include <sstream>
#include <stdarg.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

using std::cout;
using std::endl;

using std::string;
using std::stringstream;

template<typename T>
void print(string const& prefix, T const& value, string const& postfix = "")
{
  cout << prefix << value << postfix << endl;
}

template<typename T>
void print1D(string const& prefix, int count, T const& value, string const& infix, string const& postfix = "")
{
  cout << prefix;
  int i;
  for (i = 0; i < count - 1; ++i)
  {
    cout << value[i] << infix;
  }
  cout << value[i] << postfix << endl;
}

class DevicePrinter
{
    typedef void (DevicePrinter::*devicePrinter)();
    const cudaDeviceProp & deviceProp;
    const cudaDeviceProp & getDeviceProperties();
    void printWithLines(int count, devicePrinter * printers);
    void printGeneralInformation();
    void printProcessorInformation();
    void printMemoryInformation();
    void printComputabilityInformation();
  public:
    DevicePrinter();
    ~DevicePrinter();
    void printAll();
};


int main()
{
  DevicePrinter printer;
  printer.printAll();
}

const cudaDeviceProp & DevicePrinter::getDeviceProperties()
{
  cudaDeviceProp *deviceProp = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp));
  cudaGetDeviceProperties(deviceProp, 0);
  return *deviceProp;
}

DevicePrinter::DevicePrinter() : deviceProp(getDeviceProperties()) { }

DevicePrinter::~DevicePrinter()
{
  free((void*)&(this->deviceProp));
}

void DevicePrinter::printAll()
{
  devicePrinter printers [] = 
  {
    &DevicePrinter::printGeneralInformation,
    &DevicePrinter::printProcessorInformation,
    &DevicePrinter::printMemoryInformation,
    &DevicePrinter::printComputabilityInformation
  };
  
  this->printWithLines(4, printers);
}


void DevicePrinter::printWithLines(int count, devicePrinter * printers)
{
  int i;
  for(i = 0; i < count; ++i)
  {
    cout << endl;
    (this->*(printers[i]))();
  } 
}

void DevicePrinter::printGeneralInformation()
{
  print("Name: ", this->deviceProp.name);
  print("Kernel execution time-out enabled: ", this->deviceProp.kernelExecTimeoutEnabled);
}

void DevicePrinter::printProcessorInformation()
{
  print("Clock freqyency: ", this->deviceProp.clockRate, " (KHz)");
  print("Multi-processor count: ", this->deviceProp.multiProcessorCount);
  print("Warp-size: ", this->deviceProp.warpSize, " threads");
  print1D("Max grid size: ", 3, this->deviceProp.maxGridSize, " x ");
  print1D("Max block size: ", 3, this->deviceProp.maxThreadsDim, " x ");
  print("Max threads per block: ", this->deviceProp.maxThreadsPerBlock);
}

void DevicePrinter::printMemoryInformation()
{
  print("Total global memory: ", this->deviceProp.totalGlobalMem, " bytes");
  print("Total const memory: ", this->deviceProp.totalConstMem, " bytes");
  print("Shared memory (available) pr. block: ", this->deviceProp.sharedMemPerBlock, " bytes");
  print("# of 32-bit registers (available) pr. block: ", this->deviceProp.regsPerBlock);
}

void DevicePrinter::printComputabilityInformation()
{
  int computability [] = { this->deviceProp.major, this->deviceProp.minor };

  print1D("Computability: ", 2, computability, ".");
}
