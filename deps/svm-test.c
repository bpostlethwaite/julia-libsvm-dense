#include <stdlib.h>
#include <stdio.h>
#include "svm.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

  /*
   * Allocates resources to the svm_problem struct.
   * Once Julia support passing of structs, move this
   * Into Julia. Remember, we will need to hold references
   * of structs in an array while they are in C, on say
   * a function return in Julia so they don't get eaten by GC
   */
extern "C"
struct svm_problem *constructProblem(double *y, int ndata, double **x, int nvals) {
  int i, j;
  struct svm_problem *prob;

  prob = Malloc(struct svm_problem, 1);

  prob->l = ndata;
  prob->y = Malloc(double, prob->l);
  prob->x = Malloc(struct svm_node, prob->l);

  for(i=0; i < ndata; i++) {

    (prob->x+i)->values = Malloc(double, nvals);
    (prob->x+i)->dim = nvals;

    prob->y[i] = y[i];

    for ( j=0; j < nvals; j++ ) {
      (prob->x+i)->values[j] = x[i][j];
    }
  }


  return prob;

}

extern "C"
void freeProblem(struct svm_problem *prob) {
  int i;
  free(prob->y);
  for (i = 0; i < prob->l; i++)
    free((prob->x+i)->values);

  free(prob->x);

}

extern "C"
void printProblem(struct svm_problem *prob) {

  int i, j;

  for (i = 0; i < prob->l; i++) {
    printf("---- node %i ----\n", i);

    printf("%1.0f :: ", prob->y[i]);

    for (j = 0; j < (prob->x[i]).dim; j++) {
      printf("%2.2f ", prob->x[i].values[j]);
    }

    printf("\n");

  }

}

extern "C"
void get2darray(double **x, int n, int m) {

  int i, j;

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      printf("%2.2f ", x[i][j]);
    }
    printf("\n");
  }
}
