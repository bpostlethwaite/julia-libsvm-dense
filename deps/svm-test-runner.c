#include <stdio.h>
#include <stdlib.h>

#include "svm.h"
#include "svm-structs.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


int main() {


  int ndata = 13;
  int nvals = 3;
  int i, j;
  double *y;
  double **x;
  struct svm_problem *prob;

  y = Malloc(double, ndata);
  x = Malloc(double*, ndata);

  /*
   * Construct Synthetic Classes
   */
  for (i = 0; i < ndata; i++) {
    y[i] = i % 2;
  }

  for (i = 0; i < ndata; i++) {
    x[i] = Malloc(double, nvals);
    for (j = 0; j < nvals; j++) {
      x[i][j] = (double) i*nvals + j;
    }
  }

  prob = constructProblem(y, ndata, x, nvals );

  printProblem(prob);

  free(x);
  free(y);

  freeProblem(prob);

  return 0;
}
