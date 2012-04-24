#include <stdlib.h>
/* malloc and free. */

#include <iostream>
using std::cout;
using std::endl;

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "types.hpp"

#include "problem.hpp"

ApproximationProblem::ApproximationProblem(
  const Data epsilon,
  const size_t noOfVariables,
  const size_t noOfConstraints)
  : epsilon(epsilon),
    noOfVariables(noOfVariables),
    noOfConstraints(noOfConstraints)
{
  // Intentionally left blank. See initialization list.
};

ApproximationProblem::ApproximationProblem(
  ApproximationProblem const &otherProblem)
  : epsilon(otherProblem.epsilon),
    noOfVariables(otherProblem.noOfVariables),
    noOfConstraints(otherProblem.noOfConstraints)
{
  // Intetionally left blank. See initialization list.
}

Problem::Problem(
  const Data epsilon,
  const size_t noOfVariables,
  const size_t noOfConstraints,
  Data * const objective,
  Data * const constraintMatrix,
  Data * const constraintBounds)
  : ApproximationProblem(epsilon, noOfVariables, noOfConstraints),
    objective(objective),
    constraintMatrix(constraintMatrix),
    constraintBounds(constraintBounds)
{
  // Intentionally left blank. See initialization list.
};

inline void specialize(Problem const &problem, SpecialProblem * const);

SpecialProblem::SpecialProblem(Problem const &initialProblem)
  : ApproximationProblem(initialProblem),
    objective(
      (Data*)malloc(sizeof(Data) * initialProblem.noOfVariables)),
    constraintMatrix(
      (Data*)malloc(sizeof(Data) * initialProblem.noOfVariables * initialProblem.noOfConstraints))
{
  if (this->objective == NULL || this->constraintMatrix == NULL)
  {
    throw "out of memory";
  }

/*
  thrust::device_vector<Data> objective(problem.objective, problem.objective + this->noOfVariables);
  thrust::device_vector<Data> constraintBounds(this->constraintBounds)
*/
  typedef thrust::device_vector<Data>::iterator DataIterator;
  typedef thrust::tuple<DataIterator, DataIterator, DataIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

//  specialize(initialProblem, this);
}

SpecialProblem::~SpecialProblem()
{
  free(this->objective);
  free(this->constraintMatrix);
}

inline Data firstNormalization(Problem const &problem, SpecialProblem * const specialProblem);
inline void secondNormalization(SpecialProblem * const specialProblem, Data beta);

inline void specialize(Problem const &problem, SpecialProblem * const specialProblem)
{
/* Performs normalization as described in \cite[4/(455)]{luby-nisan-93}.
 *
 * \timeComplexity{O(firstNormalization) + O(secondNormalization)}
 * \spaceComplexity{O(firstNormalization) + O(secondNormalization)}
 */

/* The normalization is 2-step.  First computing the $c'_{i,j}$ and $\beta$,
 * and then $c''_{i,j}$.  This is expressed by the following function
 * composition. */

  Data beta = firstNormalization(problem, specialProblem);
  secondNormalization(specialProblem, beta);

/* We will not be needing $b$'s as all of them are simply set to 1 by the
 * normalization procedure.  This will be superimposed by a change of the
 * problem type to NORMALIZED.  A similar statement applies to $d$'s and $z$'s,
 * which effectively do not change, but theoretically become different (yet
 * related) variables. */
}

inline Data firstNormalization(Problem const &problem, SpecialProblem * const specialProblem)
{

/* Let $c'_{i,j}={c_{i,j}\over b_j\cdot d_i}$ and $\beta_j=max_i{c'_{i,j}}$ and
 * $\beta=min_j{\beta_j}$.  This function replaces $c_{i,j}$ by $c'_{i,j}$ in
 * the passed in structure and returns $\beta$. 
 *
 * \timeComplexity{O(m*n)}
 * \spaceComplexity{O(1)}
 * \where[m]{problem->noOfConstraints}
 * \where[n]{problem->noOfVariables}
 * \assume{\length{problem->noOfConstraints} > 0}
 * \assume{\length{problem->noOfVariables} > 0}
 */

  size_t N = specialProblem->noOfConstraints;
  size_t M = specialProblem->noOfVariables;
  Data* d = problem.objective;
  Data* b = problem.constraintBounds;
  Data* c = problem.constraintMatrix;
  Data* c_prime = specialProblem->constraintMatrix;

  Data beta = DATA_MAX;
  for (size_t j = 0; j < M; ++j)
  {
    Data beta_j = DATA_MIN;
    for (size_t i = 0; i < N; ++i)
    {
      *c_prime = *c / ( b[j] * d[i] );
      beta_j = MAX(beta_j, *c_prime);
      c++;
      c_prime++;
    }
    beta = MIN(beta, beta_j);
  }

  return beta;
}

inline void secondNormalization(SpecialProblem * const specialProblem, Data beta)
{

/* This part of the normalization floors large $c'_{i,j}$ to $\beta\cdot m\over
 * \epsilon$ and floors small $c'_{i,j}$ to $0$, in particular, those that are
 * $<\epsilon\cdot\beta\over m$.
 *
 * \timeComplexity{O(m*n)}
 *
 * \spaceComplexity{O(1)}
 *
 * \where[m]{problem->noOfConstraints}
 *
 * \where[n]{problem->noOfVariables}
 */

  Data epsilon = specialProblem->epsilon;
  Data *c = specialProblem->constraintMatrix;
  size_t N = specialProblem->noOfVariables;
  size_t M = specialProblem->noOfConstraints;
  size_t noOfCoefficients = N * M;

  Data upperBound = ( beta * M ) / epsilon;
  Data lowerBound = ( epsilon * beta ) / M;
  Data t = DATA_MIN;

  for (size_t i = 0; i < noOfCoefficients; ++i, c++)
  {
    if(*c > upperBound)
    {
      *c = upperBound;
    }
    else if (*c < lowerBound)
    {
      *c = 0;
    }
    t = MAX(t, *c);
  }

  c = specialProblem->constraintMatrix;

  for (size_t i = 0; i < noOfCoefficients; ++i, c++)
  {
    *c /= t;
  }
}

