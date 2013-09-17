# julia-libsvm-dense

Bindings for Julia lib-svm with the intent to move as much functionality into the Julia side as possible.

## Todo
1. construct svm_node structs in C. Pass comfortably between interfaces      [COMPLETE]
2. construct svm_problem structs in C. Pass comfortably between interfaces   [COMPLETE]
3. construct svm_parameter struct in C. Pass comfortably between interfaces  [COMPLETE]
4. implement wrappers around ccall struct constructors                       [COMPLETE]
5. implement Julia reader/writer for LIBSVM datasets for testing purposes
6. implement wrappers around LIBSVM training and prediction functions
7. implement wrappers around the get_model_param functions
8. test functionality on test data, compare to command line LIBSVM
9. implement Julia scaling function and test against scaled command line util
10. implement cross validation and test against command line
11. implement grid-search style cross-validation for parameter estimation
12. implement visualization of grid-search


## Notes
