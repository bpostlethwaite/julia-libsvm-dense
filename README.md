# julia-libsvm

Bindings for Julia lib-svm with the intent to move as much functionality into the Julia side as possible.

## Todo
1. construct svm_node structs in C. Pass comfortably between interfaces
2. construct svm_problem structs in C. Pass comfortably between interfaces
3. construct svm_parameter struct in C. Pass comfortably between interfaces
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
