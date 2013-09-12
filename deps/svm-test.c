#include <stdlib.h>
#include <stdio.h>
#include "svm.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

//struct svm_parameter param;		// set by parse_command_line
//struct svm_model *model;

// struct svm_problem
// {
// 	int l;
// 	double *y;
// 	struct svm_node *x;
// };

// struct svm_node
// {
// 	int dim;
// 	double *values;
// };

//ccall(:foo, Void, (Int32, Ptr{Ptr{Uint8}}), length(argv), argv)
extern "C"
struct svm_problem *constructProblem(double *y, int ndata, double *x, int nvals) {

  int i, j;
  struct svm_problem *prob;

  prob = Malloc(struct svm_problem, 1);

  prob->l = ndata;
  prob->y = Malloc(double, prob->l);
  prob->x = Malloc(struct svm_node, prob->l);

  for(i=0; i < prob->l; i++) {

    (prob->x+i)->values = Malloc(double, nvals);
    (prob->x+i)->dim = nvals;

    prob->y[i] = y[i];

    for ( j=0; j < nvals; j++ ) {
      (prob->x+i)->values[j] = x[j];
    }
  }


  return prob;

}

extern "C"
void printProblem(struct svm_problem *prob) {

  int i, j;

  for (i = 0; i < prob->l; i++) {
    printf("---- node %i ----\n", i);
    for (j = 0; j < (prob->x[i]).dim; j++) {
      printf("%2.2f ", prob->x[i].values[j]);
    }
    printf("\n");
  }

}


