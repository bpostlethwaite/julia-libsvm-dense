include("libsvm-dense.jl")

using LibSVM_dense

######################### TESTING #####################################


prob = readdlm("heart_scale", SVMproblem)
init_struct!(prob)

param = SVMparameter()
#param.gamma = 1.0/(prob.x[1].dim)
param.C = 853.0
param.gamma = 3e-5

init_struct!(param)

err = svm_check_parameter(prob, param)
if err
  println(err)
end

#include("cross-validation-gridsearch.jl")
#nr_fold = 5
#(c, gamma) = cross_validation_gridsearch(nr_fold)


target = svm_cross_validation(prob, param, 5)

total_correct = 0
for i=1:prob.l
	if target[i] == prob.y[i]
		total_correct += 1
  end
end

@printf("Cross Validation Accuracy = %2.3f%%\n",100.0* total_correct/prob.l);



#model = svm_train(prob, param)

#err = svm_save_model("savedModel", model)

#if err
#  println(err)
#end

free_struct!(prob)
free_struct!(param)


#readavailable(STDIN)
#svm_free_and_destroy_model!(model)















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
