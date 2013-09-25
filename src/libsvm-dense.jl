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
    readdlm,
    svm_train,
    svm_save_model,
    svm_free_and_destroy_model!,
    svm_check_parameter,
    svm_cross_validation,
    cross_validation_gridsearch


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

##################### LIBRARIES ###########################
macro LIBWRAP()
  return "../deps/libsvm-structs.so"
end

macro LIBSVM()
  return "../deps/libsvm.so"
end

####################### TYPES #############################

type SVMnode
	dim::Cint
	values::Vector{Float64}
end

SVMnode(x::Vector) = SVMnode(int32(length(x)), x)
SVMnode(dim::Int64, x::Vector) = SVMnode(int32(dim), x)






type SVMproblem
	l::Cint
	y::Vector{Float64}
	x::Array{SVMnode, 1}
  cpointer::Union(Bool, Ptr{Void}) # This is a stop-gap until we get struct passing support in Julia
  SVMproblem(l, y, x, p) = (l == length(y) == length(x)) ? new(l, y, x, p) : error("dimensional mismatch")
end

SVMproblem(y::Vector, x::Array{SVMnode}) = SVMproblem(int32(length(y)), y::Vector, x::Array{SVMnode}, false )
SVMproblem(l::Int64, y::Vector, x::Array{SVMnode}) = SVMproblem(int32(l), y::Vector, x::Array{SVMnode}, false )





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
                              false)     # Until structs are passable




# Right now just the minimum.
# Eventually pull functionality into Julia
type SVMmodel
  cpointer::Union(Bool, Ptr{Void})
end

SVMmodel() = SVMmodel(false)

####################### C STRUCT HANDLERS #############################

function init_struct!(prob::SVMproblem)

  if prob.cpointer == false
    a = Array(Ptr{Float64}, prob.l)
    for i = 1 : prob.l
      a[i] = pointer(prob.x[i].values)
    end

    point = ccall( (:constructProblem, @LIBWRAP), Ptr{Void},
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

    point = ccall( (:constructParameter,  @LIBWRAP), Ptr{Void},
                  (Ptr{Cint}, Ptr{Float64}), ints, floats)

    param.cpointer = point
  end

end




function free_struct!(prob::SVMproblem)
  if prob.cpointer != false
    ccall( (:freeProblem,  @LIBWRAP), Void,
          (Ptr{Void},), prob.cpointer)
    prob.cpointer = false
  end
end



function free_struct!(param::SVMparameter)
  if param.cpointer != false
    ccall( (:svm_destroy_param, @LIBSVM), Void,
          (Ptr{Void},), param.cpointer)
    param.cpointer = false
  end
end


####################### LIBSVM WRAPPERS  #############################

# [DONE] struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
# void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

# int svm_save_model(const char *model_file_name, const struct svm_model *model);
# struct svm_model *svm_load_model(const char *model_file_name);

# int svm_get_svm_type(const struct svm_model *model);
# int svm_get_nr_class(const struct svm_model *model);
# void svm_get_labels(const struct svm_model *model, int *label);
# void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
# int svm_get_nr_sv(const struct svm_model *model);
# double svm_get_svr_probability(const struct svm_model *model);

# double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
# double svm_predict(const struct svm_model *model, const struct svm_node *x);
# double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

# void svm_free_model_content(struct svm_model *model_ptr);
# void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
# void svm_destroy_param(struct svm_parameter *param);

# const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
# int svm_check_probability_model(const struct svm_model *model);

# void svm_set_print_string_function(void (*print_func)(const char *));

function svm_train(prob::SVMproblem, param::SVMparameter)

    model = SVMmodel()

    point = ccall( (:svm_train, @LIBSVM), Ptr{Void},
                  (Ptr{Void}, Ptr{Void}), prob.cpointer, param.cpointer)

    model.cpointer = point

    return model

end

function svm_cross_validation(prob::SVMproblem, param::SVMparameter, nr_fold::Cint)

  target = Array(Float64, prob.l)

	ccall( (:svm_cross_validation, @LIBSVM), Void,
          (Ptr{Void}, Ptr{Void}, Cint, Ptr{Float64}),
          prob.cpointer, param.cpointer, nr_fold, target)

  return target
end

svm_cross_validation(x::SVMproblem, y::SVMparameter, nr_fold::Int) = svm_cross_validation(x, y, int32(nr_fold))


function svm_save_model(fname::ASCIIString, model::SVMmodel)

  # Return true on success and false on failure
  fp = convert(Ptr{Uint8}, fname)
  err = ccall( (:svm_save_model, @LIBSVM), Cint,
              (Ptr{Uint8}, Ptr{Void}),
              fp, model.cpointer)

  return (err == 0) ? false : true

end

function svm_free_and_destroy_model!(model::SVMmodel)

  if model.cpointer != false
    A = Array(Ptr{Void}, 1)
    A[1] = model.cpointer
    ccall( (:svm_free_and_destroy_model, @LIBSVM), Void,
          (Ptr{Ptr{Void}},), A)
    model.cpointer = false
  end
end

function svm_check_parameter(prob::SVMproblem, param::SVMparameter)
  # This function is "type-unstable" but who cares, its not hot
  val = ccall( (:svm_check_parameter, @LIBSVM), Ptr{Uint8},
              (Ptr{Void}, Ptr{Void}), prob.cpointer, param.cpointer)

  # function returns NULL if correct, otherwise returns an error
  if val != C_NULL
    error = bytestring(val)
  else
    error = false
  end

  return error

end

####################### OPTIMIZATION #############################

function cross_validation_gridsearch(probt::SVMproblem,
                                     paramt::SVMparameter,
                                     nr_fold::Int,
                                     cRange::Vector{Float64},
                                     gRange::Vector{Float64})

# Perform Multi-scale Gridsearch over CRange and gammaRange using cross-validation

  nr_fold = convert(Cint, nr_fold)

  # Free if already assigned memory in C
  free_struct!(probt)
  free_struct!(paramt)

  agrid = Array(Float64, length(cRange), length(gRange))

  np = nprocs()

  refs = Array(Any, np)

  # Break ranges into np chunks, slice up columns (gamma)
  ng = int(floor(length(gRange) / np))
  remg = rem(length(gRange), np)

  gi = Array(Int64, np, 2)

  lstg = 1

  # Add the remainder evenly across workers
  for i = 1:np
    nxtg = lstg + ng - 1 + (remg > 0 ? 1 : 0)
    remg -= 1
    gi[i,:] = [lstg, nxtg]
    lstg = nxtg + 1
  end

  println(gi)

  # Perform parallized gridsearch
  for i = 1:np
    refs[i] = @spawnat i gridsearch(cRange, gRange[gi[i,1] : gi[i,2]], nr_fold)
  end


  for i = 1:np
    pgrid = fetch(refs[i])
    println(typeof(pgrid))
    agrid[ :, gi[i,1] : gi[i,2] ] = pgrid
  end

  return agrid
end

function gridsearch(cRange, gRange, probt, paramt, nr_fold)

  pgrid = Array(Float64, length(cRange), length(gRange))

  for (ic, c) in enumerate(cRange)
    for (ig, g) in enumerate(gRange)

      prob = deepcopy(probt)
      init_struct!(prob)

      param = deepcopy(paramt)
      param.C = c
      param.gamma = g
      init_struct!(param)

      target = svm_cross_validation(prob, param, nr_fold)

      total_correct = 0
      for i=1:prob.l
	      if target[i] == prob.y[i]
		      total_correct += 1
        end
      end

      pgrid[ic, ig] = 100 * total_correct / prob.l

      free_struct!(prob)
      free_struct!(param)
    end
  end

  return pgrid

end

####################### IO FUNCTIONS #############################


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

  #maxIndex += 1 # Accounting for zero based C

  # Now lets loop through the saved text and fill
  # in the right values for the given indexes
  # and assign to SVMnode
  for i = 1:ndata
    data = zeros(Float64, maxIndex)
    inds = map( (x) -> x[1], temp[i] )
    #inds += 1
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