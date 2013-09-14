type SVM_node
	dim::Int32
	values::Vector{Float64}
end

SVM_node(x::Vector) = SVM_node(int32(length(x)), x)

type SVM_problem
	l::Int32
	y::Vector{Float64}
	x::Array{SVM_node, 1}
  SVM_problem(n, y, x) = (n == length(y) == length(x)) ? error("dimensional mismatch") : new(n, x, y)
end

SVM_problem(y::Vector, x::Array{SVM_node}) = SVM_problem(int32(length(y)), y::Vector, x::Array{SVM_node})

type SVM_parameter
	svm_type::Int32     # a
	kernel_type::Int32  # b
	degree::Int32	      # c for poly
	gamma::Float64	    # d for poly/rbf/sigmoid
	coef0::Float64	    # e for poly/sigmoid

  # These are for Training Only
	cache_size::Float64 # f in MB
	eps::Float64	      # g stopping criteria
	C::Float64	        # h for C_SVC, EPSILON_SVR and NU_SVR
	nr_weight::Int32		# i for C_SVC
	weight_label::Array{Int32, 1}	# j  for C_SVC
	weight::Vector{Float64} 		  # k for C_SVC
	nu::Float64	        # l for NU_SVC, ONE_CLASS, and NU_SVR
	p::Float64	        # m for EPSILON_SVR
	shrinking::Int32	  # n use the shrinking heuristics
	probability::Int32  # o do probability estimates

end



ndata = 10
nvals = 5

Y = float64([i % 2 for i = 0 : ndata-1])
X = float64([i for i = 0 : nvals * ndata - 1])

#
# Keep X around so we don't garbage collect pointers to data!
#
X = reshape(X, nvals, ndata)

println(size(X))

a = Array(Ptr{Float64}, size(X,2))
svm_nodeArray = Array(SVM_node, size(X, 2))

for i = 1 : size(X,2)
  a[i] = pointer(X[:,i], 1)
  svm_nodeArray[i] = SVM_node(X[:,i])
end

println(length(Y))
println(length(svm_nodeArray))

S = SVM_problem(int32(length(Y)), Y, svm_nodeArray)

println(S)

prob = ccall( (:constructProblem, "../deps/libsvm-structs.so"), Ptr{Void},
             (Ptr{Float64}, Int32, Ptr{Ptr{Float64}}, Int32),
             Y, int32(ndata), a, int32(nvals))

ccall( (:printProblem, "../deps/libsvm-structs.so"), Void,
      (Ptr{Void},), prob)

ccall( (:freeProblem, "../deps/libsvm-structs.so"), Void,
      (Ptr{Void},), prob)


