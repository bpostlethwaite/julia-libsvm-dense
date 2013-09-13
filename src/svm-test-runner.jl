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


prob = ccall( (:constructProblem, "../deps/libsvm-structs.so"), Ptr{Void},
             (Ptr{Float64}, Int32, Ptr{Ptr{Float64}}, Int32),
             Y, int32(ndata), a, int32(nvals))

ccall( (:printProblem, "../deps/libsvm-structs.so"), Void,
      (Ptr{Void},), prob)

ccall( (:freeProblem, "../deps/libsvm-structs.so"), Void,
      (Ptr{Void},), prob)


type svm_node
	dim::Int32
	values::Array{Float64, 1}
end

type svm_problem
	l::Int32
	y::Array{Float64, 1}
	x::Array{svm_node, 1}
end


type svm_parameter
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
	weight::Array{Float64, 1}		  # k for C_SVC
	nu::Float64	        # l for NU_SVC, ONE_CLASS, and NU_SVR
	p::Float64	        # m for EPSILON_SVR
	shrinking::Int32	  # n use the shrinking heuristics
	probability::Int32  # o do probability estimates

end

