
export predict_test_data,
       cross_validation_gridsearch,
       svm_scale!,
       readsvm




function predict_test_data(model::SVMmodel, test::SVMproblem)

  preds = Array(Float64, test.l)

  for (i, node) in enumerate(test.x)
    init_struct!(node)
    preds[i] = svm_predict(model, node)
    free_struct!(node)
  end

  correct = sum( test.y .== preds )
  @printf("Accuracy = %2.2f%% (%d/%d) (classification)\n",
			    correct/test.l*100.0,correct, test.l)


  return preds

end

if myid() == 1
  println("requiring")

end

println(myid())
require("/home/bpostlet/programming/julia/libsvm-dense/src/gridsearch.jl")

function cross_validation_gridsearch(datafile::ASCIIString)

  println(myid())

  foundMax = false
  maxIter = 4
  nr_fold = 5
  gexp = -15:1:3
  cexp = -5:1:15

  gammaRange = [(2.0^i)::Float64 for i in gexp]
  cRange = [(2.0^i)::Float64 for i in cexp]

  gind = 0
  cind = 0

  while !foundMax

    tic()
    agrid = parallel_gridsearch(datafile, nr_fold, cRange, gammaRange)
    toc()

    imagesc( (gammaRange[1], gammaRange[end]),
            (cRange[1], cRange[end]),
            agrid)

    (maxv, maxi) = findmax(agrid)
    (cind, gind) = ind2sub(size(agrid), maxi)

    @printf("C is %2.4e with index %i\n", cRange[cind], cind)
    @printf("gamma is %2.4e with index %i\n", gammaRange[gind], gind)

    if cind == 1
      cexp -= 2
    elseif cexp == length(cRange)
      cexp += 2
    end

    if gind == 1
      gexp -= 2
    elseif gind == length(gammaRange)
      gexp += 2
    end

    if cind != 1 && cexp != length(cRange) && gind != 1 && gind != length(gammaRange)
      foundMax = true
    end

    if maxIter < 1
      error("max iteration reached in gridsearch optimization")
    end

    maxIter -= 1

    gammaRange = [(2.0^i)::Float64 for i in gexp]
    cRange = [(2.0^i)::Float64 for i in cexp]


  end

  println("reticulating scales")

  @printf("C is %2.4e with index %i\n", cRange[cind], cind)
  @printf("gamma is %2.4e with index %i\n", gammaRange[gind], gind)

  gammaRange = linspace(gammaRange[gind - 1], gammaRange[gind + 1], 10)
  cRange = linspace(cRange[cind - 1], cRange[cind + 1], 10)

  agrid = parallel_gridsearch(datafile, nr_fold, cRange, gammaRange)

  imagesc( (gammaRange[1], gammaRange[end]),
          (cRange[1], cRange[end]),
          agrid)


  (maxv, maxi) = findmax(agrid)
  (cind, gind) = ind2sub(size(agrid), maxi)

  c_opt = cRange[cind]
  g_opt = gammaRange[gind]

  @printf("optimal C is %2.4e\n", c_opt)
  @printf("optimal gamma is %2.4e\n", g_opt)

  return (c_opt, g_opt)

end


function parallel_gridsearch(datafile::ASCIIString,
                             nr_fold::Int,
                             cRange::Vector{Float64},
                             gRange::Vector{Float64})

# Perform Roving Multi-scale Gridsearch over CRange and gammaRange using cross-validation

  nr_fold = convert(Cint, nr_fold)

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

  # Perform parallized gridsearch



  for i = 1:np
    #refs[i] = remotecall(i, gridsearch, datafile, cRange, gRange[gi[i,1] : gi[i,2]], nr_fold)
    refs[i] = @spawnat i gridsearch(datafile, cRange, gRange[gi[i,1] : gi[i,2]], nr_fold)
  end

  for i = 1:np
    pgrid = fetch(refs[i])
    agrid[ :, gi[i,1] : gi[i,2] ] = pgrid
  end

  return agrid
end




immutable Scale
  a::Float64
  b::Float64
  min::Float64
  max::Float64
  fx::Function
end

Scale(a::Float64, b::Float64, mn::Float64,
      mx::Float64) = Scale(a, b, mn, mx,
                             x -> (b - a) * (x - mn) / (mx - mn) + a)
Scale(mn::Float64,
      mx::Float64) = Scale(-1.0, 1.0, mn, mx,
                             x -> 2 * (x - mn) / (mx - mn) - 1)


function svm_scale!(prob::SVMproblem)

  # Get max and min for each set of features
  mns = deepcopy(prob.x[1].values)
  mxs = deepcopy(prob.x[1].values)

  for node in prob.x
    imin = node.values .< mns
    mns[imin] = node.values[imin]
    imax = node.values .> mxs
    mxs[imax] = node.values[imax]
  end

  # Array of scale types, one for each attribute set
  scales = [Scale(mns[i], mxs[i]) for i = 1:length(mxs)]
  # scaler(minvals[i], maxvals[i])
  # Apply scaling functions to each attribute value
  for node in prob.x
    svm_scale!(node, scales)
  end

  return scales

end


function svm_scale!(prob::SVMproblem, scales::Array{Scale})
  for node in prob.x
    svm_scale!(node, scales)
  end
end

function svm_scale!(node::SVMnode, scales::Array{Scale})

  if length(scales) != length(node.values)
    error("Number of scaling factors does not match SVMnode values!")
  end

  for i = 1:node.dim
    node.values[i] = scales[i].fx(node.values[i])
  end

end

####################### IO FUNCTIONS #############################


function readsvm(source, SVMproblem)

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








function writesvm(source, SVMproblem)

  fstream = open(source, "w")

  close(fstream)

end




