# script to write data to mat file
#
# The script is called as follows
#
# genmodelsims.jl [model] [objective] [ratid] [extendedstats]
# 
# to generate the simulated behavior in the mat files. [ratid] and
# [extendedstats] are optional. [ratid] defaults to 0 if not given.
# [extendedstats] defaults to "false" if not given.
#
# [extendedstats] is either "true" or "false" if given. If true, it adds
# additional statistics to the data file. This only works for a (small) subset
# of models, and not for the interleaved condition.
#
# If [ratid] is given, the simulated behavior is written to
# ../fitdata/modelsims/[model]_[objective]_[categ/ident]_rat[ratid].mat
# Otherwise, it is written to
# ../fitdata/modelsims/[model]_[objective]_[categ/ident].mat

using Random, Printf, MAT, CSV

include("common.jl")
include("simmodel.jl")
include("behaviorstats.jl")
include("models.jl")

const nrats = 1
const fitpath = "modelsims"
const datapath = "../fitdata/$(fitpath)"

# returns model for given name
function createmodel(modelname)
    modeldict = Dict("opt" => OptimalModel,
                     "optlapse" => OptimalLapseModel,
                     "optcollapse" => OptimalColLapseModel,
                     "optbiaslapse" => OptimalBiasLapseModel,
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
    if modelname in keys(modeldict)
        return modeldict[modelname]()
    else
        error("Uknown model identifier $modelname")
    end
end

# process arguments
if !(length(ARGS) ∈ 2:4)
    error("Script needs to be called with two to four argument")
end
const ratid = length(ARGS) ≥ 3 ? parse(Int64, ARGS[3]) : 0
const useratid = ratid > 0
const extendedstats = length(ARGS) == 4 && lowercase(ARGS[4]) == "true"
const modelname = lowercase(ARGS[1])
const m = createmodel(modelname)
const ϕns = ϕnames(m)
const ϕn = length(ϕns)
const objfnname = lowercase(ARGS[2])
const isitlvd = occursin("itlvd", objfnname)
const basefilename = (useratid ? "$(modelname)_$(objfnname)_rat$(ratid)" 
                               : "$(modelname)_$(objfnname)")
const fitfilename = "$(datapath)/fit_$(basefilename).csv"
if !isfile(fitfilename)
    error("Fitting data $fitfilename not found")
end
println("Loading fitting data from $fitfilename")
const d = CSV.read(fitfilename)
(maxllh, maxi) = findmax(d[:llh])
if maxi != size(d,1) && maxllh > d[end,end]
    warn("Maximum llh $maxllh found in line $maxi instead of last line")
end
const ϕ = Vector(d[maxi, ϕns])
#ϕ[1] *= 3.0  # raise bound 3-fold
println("Simulating model $modelname with parameters")
println(foldl((a, b) -> a * ", " * b, 
        [@sprintf("%s=%.2f", ϕns[i], ϕ[i]) for i in 1:ϕn]))

# helper functions to convert vectors to ones useable for matwrite
dfarr2mat(X::Array{Float64,1}) = X
dfarr2mat(X::Array{Int64,1}) = X
dfarr2mat(X::Array{Bool,1}) = X    
dfarr2mat(X::BitArray{1}) = convert(Vector{Bool}, X)    

if isitlvd
    # interleaved trials only
    if extendedstats
        error("Extended statistics not supported for the inverleaved condition")
    end

    Random.seed!(0)
    const evidence, ids = catitlvdevidence(itlvd_cs, 10, 10^5)
    const m_itlvd = simitlvdmodel(m, ϕ, evidence, ids)

    # write simulations to file (the below failes with non-ascii column names)
    d_itlvd = Dict{String,Any}([string(s) => dfarr2mat(m_itlvd[s]) for s in names(m_itlvd)])
    matwrite("$(datapath)/$(basefilename)_itlvd.mat", d_itlvd)
    println("Data written to $(datapath)/$(basefilename)_itlvd.mat")
else
    # separate categ/ident trials
    Random.seed!(0)
    const evidence, ids = catevidence(ident_cs, categ_cs, 10, 10^5);
    const m_ident, m_categ = (extendedstats ? 
        simmodelwithstats(m, ϕ, evidence, ids) : simmodel(m, ϕ, evidence, ids))

    # write simulations to file (the below failes with non-ascii column names)
    d_ident = Dict{String,Any}([string(s) => dfarr2mat(m_ident[s]) for s in names(m_ident)])
    d_categ = Dict{String,Any}([string(s) => dfarr2mat(m_categ[s]) for s in names(m_categ)])
    matwrite("$(datapath)/$(basefilename)_ident.mat", d_ident)
    println("Data written to $(datapath)/$(basefilename)_ident.mat")
    matwrite("$(datapath)/$(basefilename)_categ.mat", d_categ)
    println("Data written to $(datapath)/$(basefilename)_categ.mat")
end