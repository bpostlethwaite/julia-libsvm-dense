# julia-libsvm

Bindings for Julia lib-svm with the intent to move as much functionality into the Julia side as possible.

## Todo
1. construct svm_node struct arrays in C. Pass pointers back and forth from Julia
2. construct svm_problem structs in C. Pass comfortably between interfaces
3. construct svm_parameter struct in C. Pass comfotably between interfaces
4. Read into Julia test SVM datasets
5. Implement training function
6. Implement predict function
   - Try to match results using command line LIBSVM
7. Implement Julia scaling function
   - Try to match results using command line LIBSVM
8. Implement cross validation
  - Try to match results using command line LIBSVM
9. Implement grid-search style cross-validation for optimal parameter estimation


## Notes
- I think that the svm_node represents the multiple-value feature (1,0,0) which is an array of doubles, with the dimension set to n doubles.
- svm_predict is an array of
