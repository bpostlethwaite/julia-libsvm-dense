#ifndef SVMTEST_H
#define SVMTEST_H

#include "svm.h"

#ifdef __cplusplus
extern "C" {
#endif

  struct svm_problem *constructProblem(double *y, int ndata, double **x, int nvals);

  void freeProblem(struct svm_problem *prob);

  void printProblem(struct svm_problem *prob);


#ifdef __cplusplus
}
#endif


#endif
