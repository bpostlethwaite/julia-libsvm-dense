module LibsvmDense


export
# types
    SVMnode,
    SVMproblem,
    SVMparameter,
    SVMmodel,

# constants
    C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR,
    LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED,


# functions
    init_struct!,
    free_struct!,
    svm_train,
    svm_save_model,
    svm_predict,
    svm_free_and_destroy_model!,
    svm_check_parameter,
    svm_cross_validation


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
  cpointer::Union(Bool, Ptr{Void})
end

SVMnode(x::Vector) = SVMnode(int32(length(x)), x, false)
SVMnode(dim::Int64, x::Vector) = SVMnode(int32(dim), x, false)




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

function init_struct!(node::SVMnode)

  if node.cpointer == false

    point = ccall( (:initNode, @LIBWRAP), Ptr{Void},
                  (Ptr{Float64}, Cint), node.values, node.dim)

    node.cpointer = point
  end

end


function init_struct!(prob::SVMproblem)

  if prob.cpointer == false
    a = Array(Ptr{Float64}, prob.l)
    for i = 1 : prob.l
      a[i] = pointer(prob.x[i].values)
    end

    point = ccall( (:initProblem, @LIBWRAP), Ptr{Void},
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

    point = ccall( (:initParameter,  @LIBWRAP), Ptr{Void},
                  (Ptr{Cint}, Ptr{Float64}), ints, floats)

    param.cpointer = point
  end

end

function free_struct!(node::SVMnode)
  if node.cpointer != false
    ccall( (:freeNode,  @LIBWRAP), Void,
          (Ptr{Void},), node.cpointer)
    node.cpointer = false
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
# [DONE] void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

# [DONE] int svm_save_model(const char *model_file_name, const struct svm_model *model);
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
# [DONE] void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
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

function svm_predict(model::SVMmodel, node::SVMnode)

  class::Float64

  class = ccall( (:svm_predict, @LIBSVM), Float64,
                (Ptr{Void}, Ptr{Void}),
                model.cpointer, node.cpointer)

  return class

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

include("svm-tools.jl")

end #module