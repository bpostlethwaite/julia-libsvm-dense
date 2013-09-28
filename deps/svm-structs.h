#ifndef SVMSTRUCTS_H
#define SVMSTRUCTS_H

#include "svm.h"

#ifdef __cplusplus
extern "C" {
#endif

  struct svm_problem *initNode(double *x, int dim);

  struct svm_problem *initProblem(double *y, int ndata, double **x, int nvals);

  struct svm_parameter *initParameter(int *ints, double *floats);

  void printParameter(struct svm_parameter *param);

  void freeNode(struct svm_node *node);

  void freeProblem(struct svm_problem *prob);

  void printProblem(struct svm_problem *prob);


#ifdef __cplusplus
}
#endif


#endif
