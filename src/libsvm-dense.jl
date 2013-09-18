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
    free_struct!,
    readdlm




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
	dim::Cint
	values::Vector{Float64}
end

SVMnode(x::Vector) = SVMnode(int32(length(x)), x)
SVMnode(dim::Int64, x::Vector) = SVMnode(int32(dim), x)







##### SVMPROBLEM
type SVMproblem
	l::Cint
	y::Vector{Float64}
	x::Array{SVMnode, 1}
  cpointer::Union(Bool, Ptr{Void}) # This is a stop-gap until we get struct passing support in Julia
  SVMproblem(l, y, x, p) = (l == length(y) == length(x)) ? new(l, y, x, p) : error("dimensional mismatch")
end

SVMproblem(y::Vector, x::Array{SVMnode}) = SVMproblem(int32(length(y)), y::Vector, x::Array{SVMnode}, false )
SVMproblem(l::Int64, y::Vector, x::Array{SVMnode}) = SVMproblem(int32(l), y::Vector, x::Array{SVMnode}, false )






##### SVMPARAMETER
type SVMparameter
	svm_type::Cint     #
	kernel_type::Cint  #
	degree::Cint	      # for poly
	gamma::Float64	    # for poly/rbf/sigmoid
	coef0::Float64	    # for poly/sigmoid

  # These are for Training Only
	cache_size::Float64 # in MB
	eps::Float64	      # stopping criteria
	C::Float64	        # for C_SVC, EPSILON_SVR and NU_SVR
	nr_weight::Cint		# for C_SVC
	weight_label::Union(Array{Cint, 1}, Ptr{Void}) # for C_SVC
	weight::Union(Vector{Float64}, Ptr{Void}) # for C_SVC
	nu::Float64	        # for NU_SVC, ONE_CLASS, and NU_SVR
	p::Float64	        # for EPSILON_SVR
	shrinking::Cint	  # use the shrinking heuristics
	probability::Cint  # do probability estimates
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





##### STRUCT FUNCTIONS

function init_struct!(prob::SVMproblem)

  if prob.cpointer == false
    a = Array(Ptr{Float64}, prob.l)
    for i = 1 : prob.l
      a[i] = pointer(prob.x[i].values)
    end

    point = ccall( (:constructProblem, "../deps/libsvm-structs.so"), Ptr{Void},
                  (Ptr{Float64}, Cint, Ptr{Ptr{Float64}}, Cint),
                  prob.y, prob.l, a, prob.x[1].dim)

    prob.cpointer = point
  end

end

function init_struct!(param::SVMparameter)

  if param.cpointer == false
    ints = Array(Cint, 6)
    ints[1] = param.svm_type
    ints[2] = param.kernel_type
    ints[3] = param.degree
    ints[4] = param.nr_weight
    ints[5] = param.shrinking
    ints[6] = param.probability

    floats = Array(Float64, 7)
    floats[1] = param.gamma
    floats[2] = param.coef0
    floats[3] = param.cache_size
    floats[4] = param.eps
    floats[5] = param.C
    floats[6] = param.nu
    floats[7] = param.p

    point = ccall( (:constructParameter, "../deps/libsvm-structs.so"), Ptr{Void},
                  (Ptr{Cint}, Ptr{Float64}), ints, floats)

    param.cpointer = point
  end

end

function free_struct!(prob::SVMproblem)
  if typeof(prob.cpointer) != false
    ccall( (:freeProblem, "../deps/libsvm-structs.so"), Void,
          (Ptr{Void},), prob.cpointer)
    prob.cpointer = false
  end
end


function free_struct!(param::SVMparameter)
  if typeof(param.cpointer) != false
    ccall( (:svm_destroy_param, "../deps/libsvm.so"), Void,
          (Ptr{Void},), param.cpointer)
    param.cpointer = false
  end
end

#### I/O FUNCTIONS

function readdlm(source, SVMproblem)

  fstream = open(source, "r")
  chunk = readall(fstream)
  close(fstream)

  lines = split(strip(chunk), "\n")

  # Strip out short lines
  ll = map((x) -> length(x) > 4, lines)
  lines = lines[ll]

  ndata = length(lines)

  temp = cell(ndata)

  y = Array(Float64, ndata)
  x = Array(SVMnode, ndata)

  # First we need to get the max index from all the rows
  # May as well save some of the parsed text
  maxIndex = 0
  for i = 1:ndata
    fields = split(strip(lines[i]), " ")
    y[i] = float64(fields[1])
    fields = map( (x) -> split(x, ":"), fields[2:end])
    temp[i] = map( (x) -> (int64(x[1]), float64(x[2])), fields)
    maxIndex = max(temp[i][end][1], maxIndex)
  end

  # Now lets loop through the saved text and fill
  # in the right values for the given indexes
  # and assign to SVMnode
  for i = 1:ndata
    data = zeros(Float64, maxIndex)
    inds = map( (x) -> x[1], temp[i] )
    vals = map( (x) -> x[2], temp[i] )
    data[inds] = vals
    x[i] = SVMnode(maxIndex, data)
  end

  prob = SVMproblem(ndata, y, x)


end


function writedlm(source, SVMproblem)

  fstream = open(source, "w")

  close(fstream)

end


end #module