#include <stdio.h>
#include <stdlib.h>

#include "svm.h"
#include "svm-test.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


int main() {


  int ndata = 1;
  int nvals = 5;
  int i;
  double *y;
  double *x;
  struct svm_problem *prob;

  y = Malloc(double, ndata);
  x = Malloc(double, nvals);

  /*
   * Construct Synthetic Classes
   */
  for (i = 0; i < ndata; i++) {
    y[i] = i % 2;
  }

  for (i = 0; i < nvals; i++) {
    x[i] = (double) i * 1.42;
  }

  prob = constructProblem(y, ndata, x, nvals );

  printProblem(prob);

  free(x);
  free(y);

  freeProblem(prob);

  return 0;
}
