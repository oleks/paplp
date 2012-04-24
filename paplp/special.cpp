/* Normalization as described in \cite[4/(455)]{luby-nisan-93}.  It is
 * recommended to have the article at hand as you review this code.
 */

#include <math.h>
/* Included for pow, needed in \function{checkNormalization}. */

#include <assert.h>
/* \function{checkNormalization} asserts the property that $t/b\leq\gamma$ as
 * described in \cite[4/(455)]{luby-nisan-93}. */

#include <stdio.h>

#include "lib.h"
/* Included bceause of MIN/MAX definitions.*/

#include "types.hpp"
#include "problem.hpp"

#include "special.hpp"

inline Data firstNormalization(Problem const &problem, SpecialProblem const &specialProblem);
inline void secondNormalization(SpecialProblem const &problem, Data beta);

void convertToSpecialForm(Problem const &problem, SpecialProblem const &specialProblem)
{
/* Performs normalization as described in \cite[4/(455)]{luby-nisan-93}.
 *
 * \timeComplexity{O(firstNormalization) + O(secondNormalization)}
 * \spaceComplexity{O(firstNormalization) + O(secondNormalization)}
 */

/* The normalization is 2-step.  First computing the $c'_{i,j}$ and $\beta$,
 * and then $c''_{i,j}$.  This is expressed by the following function
 * composition. */

  secondNormalization(problem, firstNormalization(problem, specialProblem));

/* We will not be needing $b$'s as all of them are simply set to 1 by the
 * normalization procedure.  This will be superimposed by a change of the
 * problem type to NORMALIZED.  A similar statement applies to $d$'s and $z$'s,
 * which effectively do not change, but theoretically become different (yet
 * related) variables. */
}

inline Data firstNormalization(Problem const &problem, SpecialProblem const &specialProblem)
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

  Data* d = problem.objective;
  Data* b = problem.constraintBounds;
  Data* c = problem.constraintMatrix;
  Data* c_prime = specialProblem.constraintMatrix;

  Data beta = DATA_MAX;
  for (size_t j = 0; j < problem.noOfConstraints; ++j)
  {
    Data beta_j = DATA_MIN;
    for (size_t i = 0; i < problem.noOfVariables; ++i)
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

#define GET_COEFFICIENTS() \
  Data *c = problem.constraintMatrix;\
  size_t noOfCoefficients = problem.noOfConstraints * problem.noOfVariables;

inline void secondNormalization(SpecialProblem const &problem, Data beta)
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

  GET_COEFFICIENTS();

  Data upperBound = ( beta * problem.noOfConstraints ) / problem.epsilon;
  Data lowerBound = ( problem.epsilon * beta ) / problem.noOfConstraints;
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

  c = problem.constraintMatrix;

  for (size_t i = 0; i < noOfCoefficients; ++i, c++)
  {
    *c /= t;
  }
}

void ensureSpecialForm(SpecialProblem const &problem)
{

/* Let $t = max_{i,j}\set{ c_{i,j} }$, $b = min_{i,j}\set{ c_{i,j} : c_{i,j} >
 * 0 }$, and $\gamma = { m^2 \over \epsilon^2 }$. Then, by
 * \cite[4/(455)]{luby-nisan-93}, ${ t \over b} \leq \gamma$ for any normalized
 * problem. This function computes $t$, $b$, and %\gamma% given a problem, and
 * asserts for this property to hold. Also, this function checks the type field
 * in the problem structure.
 *
 * \timeComplexity{O(m*n)} \where[m]{problem->noOfConstraints}
 * \where[n]{problem->noOfVariables} \spaceComplexity{O(1)}
 */

  GET_COEFFICIENTS();

  Data t = DATA_MIN;
  Data b = DATA_MAX;

  for (size_t i = 0; i < noOfCoefficients; ++i, c++)
  {
    t = MAX(t, *c);
    if (*c > 0)
    {
      b = MIN(b, *c);
    }
  }

  Data ratio = t / b;
  Data gamma = pow(problem.noOfConstraints, 2) / pow(problem.epsilon, 2);

  printf("ratio: %e, gamma: %e\n", ratio, gamma);

  assert(ratio <= gamma);
}

#undef GET_COEFFICIENTS


