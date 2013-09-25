include("libsvm-dense.jl")

using LibSVM_dense
using Winston
######################### TESTING #####################################



### TEST PROBLEM ####
function testProblem()
  ndata = 10
  nvals = 15

  Y = float64([i % 2 for i = 0 : ndata-1])
  X = float64([i for i = 0 : nvals * ndata - 1])

  X = reshape(X, nvals, ndata)


  nodeArray = Array(SVMnode, size(X, 2))
  for i = 1 : size(X,2)
    nodeArray[i] = SVMnode(X[:,i])
  end


  prob = SVMproblem(Y, nodeArray)

  init_struct!(prob)


  ccall( (:printProblem, "../deps/libsvm-structs.so"), Void,
        (Ptr{Void},), prob.cpointer)

  free_struct!(prob)
end

prob = readdlm("heart_scale", SVMproblem)
init_struct!(prob)

param = SVMparameter()
param.gamma = 1.0/(prob.x[1].dim)
init_struct!(param)

err = svm_check_parameter(prob, param)
if err
  println(err)
end

# ccall( (:printProblem, "../deps/libsvm-structs.so"), Void,
#       (Ptr{Void},), prob.cpointer)

cRange = [2.0^i for i = -9:3]
gammaRange = [2.0^i for i = -7:5]

tic()
agrid = cross_validation_gridsearch(prob, param, 5, cRange, gammaRange)
toc()
imagesc( (cRange[1], cRange[end]),
         (gammaRange[1], gammaRange[end]),
         agrid)

#@printf("Cross Validation Accuracy = %2.3f%%\n",100.0* total_correct/prob.l);


#model = svm_train(prob, param)

#err = svm_save_model("savedModel", model)

#if err
#  println(err)
#end

free_struct!(prob)
free_struct!(param)


readavailable(STDIN)
#svm_free_and_destroy_model!(model)