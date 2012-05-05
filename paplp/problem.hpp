#pragma once

#include "types.hpp"

/* The problem hierarhy..
 *
 * Abstract classing is done via the protected constructor since no common
 * method seemed to unite the problems. A suitable candidate could probably be
 * a \mono{Solve} method, but this is (at the moment) orthogonal to the current
 * solver architecture.
 *
 * All problems are Positive Linear (it's in the name, PAPLP), single objective.
 *
 * The relation between Problem and SpecialProblem is \emph{not} a hierarchical
 * one, but rather, a Problem can be asymmetrically converted into a
 * SpecialProblem. Hence the presence of the type-cast override in the Problem
 * class definition.
 */

class ApproximationProblem
{
protected:
  ApproximationProblem(
    const Data epsilon);
  ApproximationProblem(
    ApproximationProblem const &otherProblem);
public:
  const Data epsilon;
};

class Problem : public ApproximationProblem
{
public:
  HostVector &objective;
  HostVector &constraintMatrix;
  HostVector &constraintBounds;
  Problem(
    const Data epsilon,
    HostVector &objective,
    HostVector &constraintMatrix,
    HostVector &constraintBounds);
};

class SpecialProblem : public ApproximationProblem
{
public:
  HostVector objective;
  HostVector constraintMatrix;

  SpecialProblem(Problem const &problem);
  size_t NoOfVariables(void);
  size_t NoOfConstraints(void);
};
