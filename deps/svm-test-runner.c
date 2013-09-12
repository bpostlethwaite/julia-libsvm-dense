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
    x[i] = (double) i * 2.4;
  }

  prob = constructProblem(y, ndata, x, nvals );

  printProblem(prob);

  /*
   * Free Memory
   */
  free(x);
  free(y);
  free(prob->y);
  for (int i = 0; i < prob->l; ++i)
    free((prob->x+i)->values);

  free(prob->x);

  return 0;
}
