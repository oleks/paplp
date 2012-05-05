#pragma once

#include "types.hpp"

/*

-> get a device vector for a row
-> get a blocked device vector

-> fill the matrix from host.

*/

namespace paplp
{
  class Matrix {

/** A row-major matrix, i.e. the elements of a single row are stored in a
 * consecutive memory locations, and iteration is defined over rows. */

    private:
      Data *first;
      size_t rows, columns;
    public:
      Matrix(size_t rows, size_t columns);
      /** \assume{rows * columns < sizeof(size_t)} */
      ~Matrix();

      class HostIterator
      {
        friend class Matrix;
        private:
          Data *current;
          Data const * const end;
          size_t step;
          explicit HostIterator(
            Data *begin,
            size_t step = size_t(0),
            Data const * const end = NULL);
          HostIterator();
          /** Synthesized copy constructor and assignment. */
        public:
          const Data* operator*() const;
          const Data* operator->() const;
          HostIterator& operator++();
          bool operator==(const HostIterator& other) const;
      };

      HostIterator begin();
      HostIterator end();
  };
}
