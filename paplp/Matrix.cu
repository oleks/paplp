#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Matrix.hpp"

using namespace paplp;

typedef Matrix :: HostIterator HostIterator;

Matrix :: Matrix(size_t rows, size_t columns)
  : rows(rows), columns(columns),
    first(new Data[rows * columns]) { }

Matrix :: ~Matrix()
{
  delete [] first;
}

HostIterator Matrix :: begin()
{
  return HostIterator(first, columns, first + rows * columns);
}

HostIterator Matrix :: end()
{
  return HostIterator(first + rows * columns);
}

HostIterator :: HostIterator(Data *begin, size_t step, Data const * const end)
  : current(begin), step(step), end(end) { }

HostIterator :: HostIterator()
  : current(NULL), step(size_t(0)), end(NULL) { }

const Data* HostIterator :: operator*() const
{
  return current;
}

const Data* HostIterator :: operator->() const
{
  return current;
}

HostIterator& HostIterator :: operator ++()
{
  current += step;
  return *this;
}

bool HostIterator :: operator==(const HostIterator& other) const
{
  return this -> current == other.current;
}
