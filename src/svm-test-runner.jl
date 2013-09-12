#struct svm_problem *constructProblem(double *y, int ndata, double *x, int nvals);

#void printProblem(struct svm_problem *prob);


ndata = 5
nvals = 5

Y = float64([i % 2 for i = 0 : ndata-1])
X = float64([i for i = 0 : nvals * ndata - 1])

#
# Keep X around so we don't garbage collect pointers to data!
#
X = reshape(X, ndata, nvals)

a = Array(Ptr{Float64}, size(X,2))

for i = 1 : size(X,2)
  a[i] = pointer(X[:,i], 1)
end


prob = ccall( (:constructProblem, "../deps/libsvm-test.so"), Ptr{Void},
             (Ptr{Float64}, Int32, Ptr{Ptr{Float64}}, Int32),
             Y, int32(ndata), a, int32(nvals))

ccall( (:printProblem, "../deps/libsvm-test.so"), Void,
      (Ptr{Void},), prob)

ccall( (:freeProblem, "../deps/libsvm-test.so"), Void,
      (Ptr{Void},), prob)


# ccall( (:get2darray, "../deps/libsvm-test.so"), Void,
#       (Ptr{Ptr{Float64}}, Int32, Int32), pointer(a,1), int32(ndata), int32(nvals))
