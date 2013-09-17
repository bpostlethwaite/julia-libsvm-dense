module LibSVM_dense


export
# types
    SVMnode,
    SVMproblem,
    SVMparameter,

# constants
    C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR,
    LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED,

# functions
    init_struct!,
    free_struct!





##### BUILD CONSTANTS
macro enumC(syms...)
    blk = quote
    end
    for (i,sym) in enumerate(syms)
        push!(blk.args, :(const $(esc(sym)) = $(int32(i-1))))
    end
    push!(blk.args, :nothing)
    blk.head = :toplevel
    return blk
end

@enumC C_SVC NU_SVC ONE_CLASS EPSILON_SVR NU_SVR	# svm_type
@enumC LINEAR POLY RBF SIGMOID PRECOMPUTED # kernel_type





##### SVMNODE
type SVMnode
	dim::Int32
	values::Vector{Float64}
end

SVMnode(x::Vector) = SVMnode(int32(length(x)), x)
SVMnode(dim::Int64, x::Vector) = SVMnode(int32(dim), x)







##### SVMPROBLEM
type SVMproblem
	l::Int32
	y::Vector{Float64}
	x::Array{SVMnode, 1}
  cpointer::Union(Bool, Ptr{Void}) # This is a stop-gap until we get struct passing support in Julia
  SVMproblem(l, y, x, p) = (l == length(y) == length(x)) ? new(l, y, x, p) : error("dimensional mismatch")
end

SVMproblem(y::Vector, x::Array{SVMnode}) = SVMproblem(int32(length(y)), y::Vector, x::Array{SVMnode}, false )
SVMproblem(l::Int64, y::Vector, x::Array{SVMnode}) = SVMproblem(int32(l), y::Vector, x::Array{SVMnode}, false )






##### SVMPARAMETER
type SVMparameter
	svm_type::Int32     #
	kernel_type::Int32  #
	degree::Int32	      # for poly
	gamma::Float64	    # for poly/rbf/sigmoid
	coef0::Float64	    # for poly/sigmoid

  # These are for Training Only
	cache_size::Float64 # in MB
	eps::Float64	      # stopping criteria
	C::Float64	        # for C_SVC, EPSILON_SVR and NU_SVR
	nr_weight::Int32		# for C_SVC
	weight_label::Union(Array{Int32, 1}, Ptr{Void}) # for C_SVC
	weight::Union(Vector{Float64}, Ptr{Void}) # for C_SVC
	nu::Float64	        # for NU_SVC, ONE_CLASS, and NU_SVR
	p::Float64	        # for EPSILON_SVR
	shrinking::Int32	  # use the shrinking heuristics
	probability::Int32  # do probability estimates
  cpointer::Union(Bool, Ptr{Void}) # struct-less Julia stop-gap
end

# Set Defaults
SVMparameter() = SVMparameter(C_SVC,    # svm type
                              RBF,      # kernel type
                              int32(3), # degree
                              0.0,      # gamma
                              0.0,      # Coeffs
                              ##### For training purposes Only
                              100.0,    # cache_size
                              1e-3,     # eps
                              1.0,      # C
                              int32(0), # nr_weight
                              C_NULL,   # weight_label
                              C_NULL,   # weight
                              0.5,      # nu
                              0.1,      # p
                              int32(0), # shrinking
                              int32(0), # probability
                              false     # Until structs are passable
                              )





##### FUNCTIONS

function init_struct!(prob::SVMproblem)

  if prob.cpointer == false
    a = Array(Ptr{Float64}, prob.l)
    for i = 1 : prob.l
      a[i] = pointer(prob.x[i].values)
    end

    point = ccall( (:constructProblem, "../deps/libsvm-structs.so"), Ptr{Void},
                  (Ptr{Float64}, Int32, Ptr{Ptr{Float64}}, Int32),
                  prob.y, prob.l, a, prob.x[1].dim)

    prob.cpointer = point
  end

end


function free_struct!(prob::SVMproblem)
  if typeof(prob.cpointer) != false
    ccall( (:freeProblem, "../deps/libsvm-structs.so"), Void,
          (Ptr{Void},), prob.cpointer)
    prob.cpointer = false
  end
end



end #module