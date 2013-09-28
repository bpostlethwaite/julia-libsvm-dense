include("../src/LibsvmDense.jl")
using LibsvmDense

######################### TESTING #####################################

datafile = joinpath(pwd(),"data/svmguide1")

cross_validation_gridsearch(datafile)

prob = readsvm(data/svmguide1, SVMproblem)
test = readsvm("data/svmguide1.t", SVMproblem)

param = SVMparameter()
param.gamma = 1.0/(prob.x[1].dim)

scales = svm_scale!(prob)
svm_scale!(test, scales)

init_struct!(prob)
init_struct!(test)
init_struct!(param)

err = svm_check_parameter(prob, param)
if err
  println(err)
end


model = svm_train(prob, param)

preds = predict_test_data(model, test)

free_struct!(test)
#err = svm_save_model("savedModel", model)

#if err
#  println(err)
#end

free_struct!(prob)
free_struct!(param)
svm_free_and_destroy_model!(model)

#readavailable(STDIN)




