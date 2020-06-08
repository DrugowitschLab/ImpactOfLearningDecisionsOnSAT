# script that outputs some statistics of the model fits

using Printf, Random, DataFrames, CSV

include("common.jl")
include("models.jl")
include("behaviorstats.jl")
include("simmodel.jl")
include("modelfitting.jl")

# settings (when manually evaluating the objective function)
const blockreps = 10
const simtrials = 10^5
const burnin = 1000
const nrats = 1 # do not divide by nrats = 4
const δt = 0.0001
const datapath = "../fitdata/fits"

# models
const modeldict = Dict("opt" => OptimalModel,
                       "optlapse" => OptimalLapseModel,
                       "optcollapse" => OptimalColLapseModel,
                       "optcolbiaslapse" => OptimalColBiasLapseModel,
                       "adflapse" => ADFLapseModel,
                       "adfcollapse" => ADFColLapseModel,
                       "adfbiaslapse" => ADFBiasLapseModel,
                       "adfcolbiaslapse" => ADFColBiasLapseModel,
                       "randomcolbiaslapse" => RandomColBiasLapseModel,
                       "randomcollapse" => RandomColLapseModel,
                       "gammalapse" => GammaLapseModel,
                       "gammacollapse" => GammaColLapseModel,
                       "gammabiaslapse" => GammaBiasLapseModel,
                       "gammacolbiaslapse" => GammaColBiasLapseModel,                     
                       "deltalapse" => DeltaLapseModel,
                       "deltacollapse" => DeltaColLapseModel,
                       "deltabiaslapse" => DeltaBiasLapseModel,
                       "deltacolbiaslapse" => DeltaColBiasLapseModel,
                       "deltaexpcolbiaslapse" => DeltaExpColBiasLapseModel,
                       "lcalapse" => LCALapseModel,
                       "lcacollapse" => LCAColLapseModel)
# separate list to preserve order that keys(modeldict) wouldn't have
const modelnames = ["opt", "optlapse", "optcollapse", "optcolbiaslapse",
                    "adflapse", "adfcollapse", "adfbiaslapse", "adfcolbiaslapse",
                    "randomcolbiaslapse", "randomcollapse",
                    "gammalapse", "gammacollapse", "gammabiaslapse", "gammacolbiaslapse",
                    "deltalapse", "deltacollapse", "deltabiaslapse", "deltacolbiaslapse",
                    "deltaexpcolbiaslapse", "lcalapse", "lcacollapse"]
@assert issubset(Set(modelnames), Set(keys(modeldict))) # unordered comparison

# objective functions
# second tuple component is trials, 1 = ident, 2 = categ, 3 = itlvd
const objfndict = Dict("psychchron1" => (PsychChron1ObjFn, [1, 2]),
                       "psychchron1ident" => (PsychChron1IdentObjFn, [1]),
                       "psychchron1categ" => (PsychChron1CategObjFn, [2]),
                       "psychchron2" => (PsychChron2ObjFn, [1, 2]),
                       "psychchron3" => (PsychChron3ObjFn, [1, 2]),
                       "psychchronseq1" => (PsychChronSeq1ObjFn, [1, 2]),
                       "psychchronseq2" => (PsychChronSeq2ObjFn, [1, 2]),
                       "psychchronseq3" => (PsychChronSeq3ObjFn, [1, 2]),
                       "psychchron1itlvd" => (PsychChron1ItlvdObjFn, [3]))
#const objfns = ["psychchron1", "psychchron1ident", "psychchron2", "psychchron3",
#                "psychchronseq1", "psychchronseq2", "psychchronseq3"]
const objfns = ["psychchron1", "psychchron1ident", "psychchron1categ", "psychchronseq1"]
const objfnsitlvd = ["psychchron1itlvd"]
const allobjfns = [objfns; objfnsitlvd]
@assert issubset(Set(objfns), Set(keys(objfndict)))

# model fit criteria, converted to log-evidence in log(10)
BIC(avgllh, ϕn, trials) = (trials * avgllh - 0.5ϕn * log(trials)) / log(10)
AIC(avgllh, ϕn, trials) = (trials * avgllh - ϕn) / log(10)
AICc(avgllh, ϕn, trials) = AIC(avgllh, ϕn, trials) - (ϕn * (ϕn+1) / (trials-ϕn-1)) / log(10)

# returns number of trials in categ/ident dataset as vector
datatrials() = [size(d, 1) / nrats for d in (load_data(:categ), load_data(:ident), load_data(:itlvd))]

# returns dataframe with best-fitting parameters and avg. log-likelihood
function loadmodelfits(modelname, objfnname)
    fitfilename = "$(datapath)/fit_$(modelname)_$(objfnname).csv"
    if !isfile(fitfilename)
        error("Could not find fit file $fitfilename")
    end
    d = CSV.read(fitfilename)
    # identify max-llh parameters
    maxllh, maxi = findmax(d[:llh])
    if maxllh != d[end,:llh]
        warning("Last element in $fitfilename is not best-fitting")
    end
    return d[maxi,:]
end

# prints the fit parameters for all objective functions for a given model
function printfitparams(modelname)
    # use first objective function to get parameter names
    d = loadmodelfits(modelname, objfns[1])
    ϕnames = names(d)
    # print parameters
    objfnlen = maximum([length(s) for s in allobjfns])
    println(repeat("-", objfnlen), "  ",
            join([@sprintf("%7s", string(ϕi)) for ϕi in ϕnames], "  "))
    # print fits
    for objfn in allobjfns
        d = loadmodelfits(modelname, objfn)
        println(repeat(" ", objfnlen - length(objfn)), objfn, "  ",
                join([@sprintf("%7.2f", x) for x in Vector(d)], "  "))
    end
end

# prints the fit quality for all models for the given objective function
function printfitquality(objfn, trials)
    modellen = maximum([length(s) for s in modelnames])
    println(repeat("-", modellen), "  ",
            join([@sprintf("%10s", s) for s in ["avg. llh", "BIC", "AIC", "AICc"]], "  "))
    for modelname in modelnames
        d = loadmodelfits(modelname, objfn)
        avgllh = d[:llh]
        ϕn = length(d) - 1
        fitq = [avgllh; [f(avgllh, ϕn, trials) for f in [BIC, AIC, AICc]]]
        println(repeat(" ", modellen - length(modelname)), modelname, "  ",
                join([@sprintf("%10.2f", x) for x in fitq], "  ")) 
    end
end

# plots the fit quality for all models using the parameters corresponding to
# paramobjfn and uses the given objective function type objfn to compute the
# performance for all models
function printsepfitquality(paramobjfn, evalobjfn)
    # trial sequence to fit data on
    d_ident = load_data(:ident)
    d_categ = load_data(:categ)
    Random.seed!(0)
    evidence, ids = catevidence(ident_cs, categ_cs, 10, 10^5)
    # iterate over all models
    modellen = maximum([length(s) for s in modelnames])
    println(repeat("-", modellen), "  ",
            join([@sprintf("%10s", s) for s in ["avg. llh", "BIC", "AIC", "AICc"]], "  "))
    evalobjfntype, evalobjfntrials = objfndict[evalobjfn]
    evalobjfntrials = sum([size(d_ident, 1), size(d_categ, 1)][evalobjfntrials]) / nrats
    for modelname in modelnames
        # instantiate model and extract parameters
        m = modeldict[modelname]()
        ϕ = Vector(loadmodelfits(modelname, paramobjfn)[ϕnames(m)])
        ϕn = length(ϕ)
        # evaluate objective function
        o = evalobjfntype(m, evidence, ids, burnin, d_ident, d_categ, nrats)
        avgllh = llh(o, ϕ)
        fitq = [avgllh; [f(avgllh, ϕn, evalobjfntrials) for f in [BIC, AIC, AICc]]]
        println(repeat(" ", modellen - length(modelname)), modelname, "  ",
                join([@sprintf("%10.2f", x) for x in fitq], "  "))         
    end
end

# print model parameters
for modelname in modelnames
    println("Model $modelname")
    printfitparams(modelname)
    println()
end
println()

# print model fit statistics
const trials = datatrials()
for objfn in allobjfns
    println("Objective function $objfn")
    printfitquality(objfn, sum(trials[objfndict[objfn][2]]))
    println()
end

# print model fit statistics when evaluated with psychchronseq1 objective
for objfn in objfns
    println("Fitted on $objfn, evaluated on psychchronseq1")
    printsepfitquality(objfn, "psychchronseq1")
    println()
end