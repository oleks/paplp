#include <stdlib.h>
/* malloc and free. */

#include <iostream>
using std::cout;
using std::endl;

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include "types.hpp"

#include "problem.hpp"

/* \begin{ApproximationProblem} */

ApproximationProblem::ApproximationProblem(
  const Data epsilon)
  : epsilon(epsilon)
{
  // Intentionally left blank. See initialization list.
};

ApproximationProblem::ApproximationProblem(
  ApproximationProblem const &otherProblem)
  : epsilon(otherProblem.epsilon)
{
  // Intetionally left blank. See initialization list.
}

/* \end{ApproximationProblem} */

/* \begin{Problem} */

Problem::Problem(
  const Data epsilon,
  HostVector &objective,
  HostVector &constraintMatrix,
  HostVector &constraintBounds)
  : ApproximationProblem(epsilon),
    objective(objective),
    constraintMatrix(constraintMatrix),
    constraintBounds(constraintBounds)
{
  // Intentionally left blank. See initialization list.
};

/* \end{Problem} */

/*
inline void specialize(Problem const &problem, SpecialProblem * const);
*/

/*
struct normalize
{
  __host__ __device__
    Data operator()(const Data& c)
}
*/

SpecialProblem::SpecialProblem(Problem const &initialProblem)
  : ApproximationProblem(initialProblem),
    objective(initialProblem.objective.size()),
    constraintMatrix(initialProblem.constraintMatrix.size())
{
  size_t noOfVariables = this->NoOfVariables();
  size_t noOfConstraints = this->NoOfConstraints();

/*

  DeviceVector deviceVector;

  HostVector::const_iterator i = this->constraintMatrix.begin();


  typedef thrust::tuple<DeviceVector

  for ( ; i < this->constraintMatrix.end() ; i += noOfVariables )
  {
    DeviceVector deviceVector(i, i + noOfVariables);
//    thrust::transform_reduce(deviceVector.begin(), deviceVector.end(), 
  }
*/
//  specialize(initialProblem, this);
}

size_t SpecialProblem::NoOfConstraints(void)
{
  return this->constraintMatrix.size() / this->objective.size();
}

size_t SpecialProblem::NoOfVariables(void)
{
  return this->objective.size();
}
