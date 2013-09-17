include("libsvm-dense.jl")

using LibSVM_dense
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



### TEST PARAMETER ###

function testParameter()
  param = SVMparameter()

  init_struct!(param)

  ccall( (:printParameter, "../deps/libsvm-structs.so"), Void,
        (Ptr{Void},), param.cpointer)

  free_struct!(param)
end



readdlm("heart_scale", SVMparameter)
