using Winston

function cross_validation_gridsearch(nr_fold::Int)
  foundMax = false
  maxIter = 4
  gexp = -15:1:3
  cexp = -5:1:15

  gammaRange = [(2.0^i)::Float64 for i in gexp]
  cRange = [(2.0^i)::Float64 for i in cexp]

  gind = 0
  cind = 0

  while !foundMax

    tic()
    agrid = parallel_gridsearch(nr_fold, cRange, gammaRange)
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

  agrid = parallel_gridsearch(nr_fold, cRange, gammaRange)

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

function parallel_gridsearch(nr_fold::Int,
                             cRange::Vector{Float64},
                             gRange::Vector{Float64})

# Perform Roving Multi-scale Gridsearch over CRange and gammaRange using cross-validation

  require("gridsearch")

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
    refs[i] = @spawn gridsearch(cRange, gRange[gi[i,1] : gi[i,2]], nr_fold)
  end

  for i = 1:np
    pgrid = fetch(refs[i])
    agrid[ :, gi[i,1] : gi[i,2] ] = pgrid
  end

  return agrid
end
