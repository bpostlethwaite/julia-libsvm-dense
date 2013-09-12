#struct svm_problem *constructProblem(double *y, int ndata, double *x, int nvals);

#void printProblem(struct svm_problem *prob);


ndata = int32(1)
nvals = int32(5)

Y = float64([i % 2 for i = 1 : ndata])
X = float64([i * 1.42 for i = 1 : nvals])

prob = ccall( (:constructProblem, "../deps/libsvm-test.so"), Ptr{Void},
             (Ptr{Float64}, Int32, Ptr{Float64}, Int32),
             Y, ndata, X, nvals)

ccall( (:printProblem, "../deps/libsvm-test.so"), Void,
      (Ptr{Void},), prob)



#@assert total == sum(X)

# function vecmult(x::Vector, y::Vector)
#     # Check everything before sending to C functions
#     # or enter a world of pain
#     assert(length(x) == length(y))
#     assert (typeof(x) == Array{Float32,1})
#     assert (typeof(y) == Array{Float32,1})

#     n = int32(length(x))
#     z = Array(Float32, n)

#     # Pass in z as the buffer to write results
#     ccall( (:vecmult, "./libvecmult.so"), Void,
#           (Ptr{Float32}, Ptr{Float32}, Ptr{Float32},Int32),
#           x, y, z, n)
#     # Return this array.
#     return z
# end

# Z = vecmult(X,Y)

# print(Z)

# for i = 1:length(X)
#     @printf("%4.2f    ", Z[i])
#     if Z[i] == (X[i] * Y[i])
#         println("PASS")
#     else
#         println("FAIL")
#     end
# end

