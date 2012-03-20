#include <float.h>
/* Included for DBL_MAX and DBL_MIN. Used in order to compute minimal and
 * maximal values. */

#include <math.h>
/* Included for pow, needed when \function{checkNormalization}. */

#include <assert.h>
/* \function{checkNormalization} asserts the property that $t/b\leq\gamma$ as
 * described in \cite[4/(455)]{luby-nisan-93}. */

#include "lib.h"
/* Included bceause of MIN/MAX definitions.*/

#include "normalize.h"

inline double firstNormalization(Problem* problem);
inline void secondNormalization(Problem* problem, double beta);

void normalize(Problem* problem)
{

/* Performs normalization as described in \cite[4/(455)]{luby-nisan-93}.  It is
 * recommended to have the article at hand as you review this code.
 *
 * \timeComplexity{O(firstNormalization) + O(secondNormalization)}
 * \spaceComplexity{O(firstNormalization) + O(secondNormalization)}
 */

/* The normalization is 2-step.  First computing the $c'_{i,j}$ and $\beta$,
 * and then $c''_{i,j}$.  This is expressed by the following function
 * composition. */

  secondNormalization(problem, firstNormalization(problem));

/* We will not be needing $b$'s as all of them are simply set to 1 by the
 * normalization procedure.  This will be superimposed by a change of the
 * problem type to NORMALIZED.  A similar statement applies to $d$'s and $z$'s,
 * which effectively do not change, but theoretically become different (yet
 * related) variables. */

  problem->type = NORMALIZED;
}

inline double firstNormalization(Problem* problem)
{

/* Let $c'_{i,j}={c_{i,j}\over b_j\cdot d_i}$ and $\beta_j=max_i{c'_{i,j}}$ and
 * $\beta=min_j{\beta_j}$.  This function replaces $c_{i,j}$ by $c'_{i,j}$ in
 * the passed in structure and returns $\beta$. 
 *
 * \timeComplexity{O(m*n)} \where[m]{problem->noOfConstraints}
 * \where[n]{problem->noOfVariables} \spaceComplexity{O(1)}
 */

  double*  d = problem->objectiveCoefficients;
  double*  b = problem->constraintValues;
  double*  c = problem->constraintCoefficients;

  double beta = DBL_MAX;
  for (uint32_t j = 0; j < problem->noOfConstraints; ++j)
  {
    double beta_j = DBL_MIN;
    for (uint32_t i = 0; i < problem->noOfVariables; ++i)
    {
      *c = *c / ( b[j] * d[i] );
      beta_j = MAX(beta_j, *c);
      c++;
    }
    beta = MIN(beta, beta_j);
  }

  return beta;
}

#define GET_COEFFICIENTS() \
  double* c = problem->constraintCoefficients;\
  uint32_t noOfCoefficients = problem->noOfConstraints * problem->noOfVariables;

inline void secondNormalization(Problem *problem, double beta)
{

/* This part of the normalization limits $c'_{i,j}$ to be at most $\beta\cdot
 * m\over \epsilon$ and let's small $c'_{i,j}$ be $0$, in particular, those
 * that are $<\epsilon\cdot\beta\over m$.
 *
 * \timeComplexity{O(m*n)} \where[m]{problem->noOfConstraints}
 * \where[n]{problem->noOfVariables} \spaceComplexity{O(1)}
 */

  GET_COEFFICIENTS();

  double upperBound = ( beta * problem->noOfConstraints ) / problem->epsilon;
  double lowerBound = ( problem->epsilon * beta ) / problem->noOfConstraints;

  for (uint32_t i = 0; i < noOfCoefficients; ++i, c++)
  {
    if(*c > upperBound)
    {
      *c = upperBound;
    }
    else if (*c < lowerBound)
    {
      *c = 0;
    }
  }
}

void checkNormalization(Problem* problem)
{

/* Let $t = max_{i,j}\set{ c_{i,j} }$, $b = min_{i,j}\set{ c_{i,j} : c_{i,j} >
 * 0 }$, and $\gamma = { m^2 \over \epsilon^2 }$. Then, by
 * \cite[4/(455)]{luby-nisan-93}, ${ t \over b} \leq \gamma$. This property is
 * asserted at the end of the function.
 *
 * \timeComplexity{O(m*n)} \where[m]{problem->noOfConstraints}
 * \where[n]{problem->noOfVariables} \spaceComplexity{O(1)}
 */

  GET_COEFFICIENTS();

  double t = DBL_MIN;
  double b = DBL_MAX;

  for (uint32_t i = 0; i < noOfCoefficients; ++i, c++)
  {
    t = MAX(t, *c);
    if (*c > 0)
    {
      b = MIN(b, *c);
    }
  }

  double ratio = t / b;
  double gamma = pow(problem->noOfConstraints, 2) / pow(problem->epsilon, 2);

  assert(ratio <= gamma);
}

#undef GET_COEFFICIENTS
