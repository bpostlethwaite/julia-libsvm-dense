if myid() != 1

  include("libsvm-dense.jl")

  using LibSVM_dense

end

function gridsearch(cRange, gRange, nr_fold)

  pgrid = Array(Float64, length(cRange), length(gRange))

  for (ic, c) in enumerate(cRange)
    for (ig, g) in enumerate(gRange)


      prob = readdlm("heart_scale", SVMproblem)
      init_struct!(prob)

      param = SVMparameter()
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
