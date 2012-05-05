#pragma once

#include <cstddef>
using std::size_t;

#include <float.h>
/* Included for definitions such as \code{FLT\_MIN}, \code{DBL\_MIN}, etc.
 * These are practical in conjunction with e.g. \code{DATA\_MIN} and
 * \code{DATA\_MAX} definitions. */

#include "lib.h"

typedef float Data;
/* The data type for all values computed on, such as, coefficients, epsilon,
 * etc. It is assumed that values of this type (whatever it may be) can
 * overflow (without warning) and therefore have a designated maximum and
 * minimum value. It is therefore, that when changing this type it is very
 * important to change the definitions of \code{DATA\_MAX} and
 * \code{DATA\_MIN}.  Otherwise, erroneous behaviour is very likely.
 * \label{data-type-definition} */

#define DATA_MAX (FLT_MAX)
#define DATA_MIN (FLT_MIN)
/* See discussion for \ref{comment:data-type-definition}. */

#include <thrust/host_vector.h>

typedef thrust::host_vector<Data> HostVector;

#include <thrust/device_vector.h>

typedef thrust::device_vector<Data> DeviceVector;
