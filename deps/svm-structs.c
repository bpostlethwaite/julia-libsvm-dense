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

struct svm_problem *constructParameters(double *y, int ndata, double **x, int nvals) {
  int i, j;

  struct svm_parameter *param;

  param.svm_type = C_SVC;
  param.kernel_type = RBF;
  param.degree = 3;
  param.gamma = 0;
  param.coef0 = 0;
  param.nu = 0.5;
  param.cache_size = 100;
  param.C = 1;
  param.eps = 1e-3;
  param.p = 0.1;
  param.shrinking = 1;
  param.probability = 0;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
}
