#struct svm_problem *constructProblem(double *y, int ndata, double *x, int nvals);

#void printProblem(struct svm_problem *prob);


ndata = 13
nvals = 3

Y = float64([i % 2 for i = 0 : ndata-1])
X = float64([i for i = 0 : nvals * ndata - 1])

X = reshape(X, ndata, nvals)

# prob = ccall( (:constructProblem, "../deps/libsvm-test.so"), Ptr{Void},
#              (Ptr{Float64}, Int32, Ptr{Float64}, Int32),
#              Y, int32(ndata), X, int32(nvals))

# ccall( (:printProblem, "../deps/libsvm-test.so"), Void,
#       (Ptr{Void},), prob)

# ccall( (:freeProblem, "../deps/libsvm-test.so"), Void,
#       (Ptr{Void},), prob)


a = Array(Ptr{Float64}, size(X,2))
for i = 1 : size(X,2)
  a[i] = pointer(X[:,i], 1)
end

ccall( (:get2darray, "../deps/libsvm-test.so"), Void,
      (Ptr{Ptr{Float64}}, Int32, Int32), pointer(a,1), int32(ndata), int32(nvals))
