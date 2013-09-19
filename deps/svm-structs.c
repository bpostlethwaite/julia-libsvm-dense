#include <stdlib.h>
#include <stdio.h>
#include "svm.h"

#ifdef __cplusplus
extern "C" {
#endif


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

  /*
   * Allocates resources to the svm_problem struct.
   * Once Julia support passing of structs, move this
   * Into Julia. Remember, we will need to hold references
   * of structs in an array while they are in C, on say
   * a function return in Julia so they don't get eaten by GC
   */
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

void freeProblem(struct svm_problem *prob) {
  int i;
  free(prob->y);
  for (i = 0; i < prob->l; i++)
    free((prob->x+i)->values);

  free(prob->x);

}

void printProblem(struct svm_problem *prob) {

  int i, j;
  int dim = prob->x[1].dim;

  printf("\n\n%i\n", prob->l);

  for (i = 0; i < prob->l; i++) {
    printf("---- node %i ----\n", i);

    printf("%1.0f :: ", prob->y[i]);
    /* printf("%2.2f ", prob->x[i].values[1]); */
    /* printf("%i", prob->x[i].dim); */
    for (j = 0; j < dim; j++) {
      printf("%2.2f ", prob->x[i].values[j]);
    }

    printf("\n");
  }

}

struct svm_parameter *constructParameter(int *ints, double *floats) {

  struct svm_parameter *param;

  param = Malloc(struct svm_parameter, 1);

  param->svm_type = ints[0];
  param->kernel_type = ints[1];
  param->degree = ints[2];
  param->nr_weight = ints[3];
  param->shrinking = ints[4];
  param->probability = ints[5];

  param->gamma = floats[0];
  param->coef0 = floats[1];
  param->cache_size = floats[2];
  param->eps = floats[3];
  param->C = floats[4];
  param->nu = floats[5];
  param->p = floats[6];

  param->weight_label = NULL;
  param->weight = NULL;

  return param;

}


void printParameter(struct svm_parameter *param) {


  printf("svm_type     %i\n", param->svm_type);
  printf("kernel type  %i\n", param->kernel_type);
  printf("degree       %i\n", param->degree);
  printf("nr_weight    %i\n", param->nr_weight);
  printf("shrinking    %i\n", param->shrinking);
  printf("probability  %i\n", param->probability);

  printf("gamma        %2.2f\n",  param->gamma);
  printf("coef0        %2.2f\n",  param->coef0);
  printf("cache_size   %2.2f\n",  param->cache_size);
  printf("eps          %2.2e\n",  param->eps);
  printf("C            %2.2f\n",  param->C);
  printf("nu           %2.2f\n",  param->nu);
  printf("p            %2.2f\n",  param->p);

  printf("weight label %p\n", param->weight_label);
  printf("weight       %p\n", param->weight);


}

#ifdef __cplusplus
}
#endif
