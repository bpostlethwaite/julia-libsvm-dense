# julia-libsvm

Bindings for Julia lib-svm with the intent to move as much functionality into the Julia side as possible.

## Todo
1. construct svm_node structs in C. Pass comfortably between interfaces [COMPLETE]
2. construct svm_problem structs in C. Pass comfortably between interfaces [COMPLETE]
3. construct svm_parameter struct in C. Pass comfortably between interfaces
4. implement wrappers around ccall struct constructors
5. implement Julia reader for SVM test datasets
6. implement wrappers around LIBSVM training and prediction functions
7. implement wrappers around the get_model_param functions
8. test functionality, compare to command line LIBSVM
9. implement Julia scaling function and test against scaled command line usage
11. implement cross validation and test against command line
12. implement grid-search style cross-validation for optimal parameter estimation
13. implement visualization of grid-search


## Notes
