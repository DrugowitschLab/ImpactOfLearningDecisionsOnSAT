# script to fit model to data
#
# The script is called as
#
# fitmodel.jl [model] [objective] [[rat]]
#
# and writes data to ../fitdata/fits/fit_[model]_[objective]_[rat]. Here, [rat]
# is optional and indicates the rat identifier (1-4). If not provided, the data
# for all rats is collapsed into a single rat.

using Random, CSV

# include required scripts
include("common.jl")
include("models.jl")
include("simmodel.jl")
include("behaviorstats.jl")
include("modelfitting.jl")

# settings
const blockreps = 10
const simtrials = 10^5
const burnin = 1000
const nrats = 1 # (do not divide dataset by 4 rats, nrats = 4)
const datapath = "../fitdata/fits"
const paraminiobjfn = "psychchronseq1"

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
                     "randomcollapse" => RandomColLapseModel,
                     "randomcolbiaslapse" => RandomColBiasLapseModel,
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

# returns fitting objective for given name and model
function objectivefn(objfnname, m, ratid::Int=0)
    # load data for individual / all rats
    d_ident = load_data(:ident, ratid)
    d_categ = load_data(:categ, ratid)
    d_itlvd = load_data(:itlvd, ratid)
    # simulated trials
    Random.seed!(0)
    evidence, ids = catevidence(ident_cs, categ_cs, 10, 10^5)
    itlvdevidence, itlvdids = catitlvdevidence(itlvd_cs, 10, 10^5)
    # create objective function
    if objfnname == "psychchron1"
        o = PsychChron1ObjFn(m, evidence, ids, burnin, d_ident, d_categ, nrats)
    elseif objfnname == "psychchron1ident"
        o = PsychChron1IdentObjFn(m, evidence, ids, burnin, d_ident, d_categ, nrats)
    elseif objfnname == "psychchron1categ"
        o = PsychChron1CategObjFn(m, evidence, ids, burnin, d_ident, d_categ, nrats)
    elseif objfnname == "psychchron2"
        o = PsychChron2ObjFn(m, evidence, ids, burnin, d_ident, d_categ, nrats)
    elseif objfnname == "psychchron3"
        o = PsychChron3ObjFn(m, evidence, ids, burnin, d_ident, d_categ, nrats)
    elseif objfnname == "psychchronseq1"
        o = PsychChronSeq1ObjFn(m, evidence, ids, burnin, d_ident, d_categ, nrats)
    elseif objfnname == "psychchronseq2"
        o = PsychChronSeq2ObjFn(m, evidence, ids, burnin, d_ident, d_categ, nrats)
    elseif objfnname == "psychchronseq3"
        o = PsychChronSeq3ObjFn(m, evidence, ids, burnin, d_ident, d_categ, nrats)
    elseif objfnname == "psychchron1itlvd"
        o = PsychChron1ItlvdObjFn(m, itlvdevidence, itlvdids, burnin, d_itlvd, nrats)
    elseif objfnname == "psychchron1itlvdedge"
        # only fit stimuli that also appear in ident/categ condition
        o = PsychChron1ItlvdObjFn(m, itlvdevidence, itlvdids, burnin, d_itlvd, nrats,
                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 24, 25, 32])
    else
        error("Unknown object function identifier $objfnname")
    end
end

# process arguments
if length(ARGS) ∉ [2,3]
    error("Expected two or three arguments")
end
const modelname = lowercase(ARGS[1])
const objfnname = lowercase(ARGS[2])
ratid = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0  # ratid = 0 for all rats
m = createmodel(modelname)
o = objectivefn(objfnname, m, ratid)

# check if model has already been fit with parameter initialization objective fn
datafile(modelname, objfnname, ratid) = ratid == 0 ?
    "$(datapath)/fit_$(modelname)_$(objfnname).csv" :
    "$(datapath)/fit_$(modelname)_$(objfnname)_rat$(ratid).csv"
paraminifilename = datafile(modelname, paraminiobjfn, ratid)
if ratid != 0 && !isfile(paraminifilename)
    # default to collective rat fits if they don't exist for individual rats
    paraminifilename = datafile(modelname, paraminiobjfn, 0)
end
if isfile(paraminifilename)
    # loading initial parameters from previous model fit with $paraminiobjfn
    println("Loading initial parameters from $paraminifilename")
    d = CSV.read(paraminifilename)
    (maxllh, maxi) = findmax(d[:llh])
    if maxi != size(d,1) && maxllh > d[end,end]
        warn("Maximum llh $maxllh found in line $maxi instead of last line")
    end
    ϕini = Vector(d[maxi,1:end-1])
else
    # use initial parameters are determined by model
    ϕini = initialϕ(m)
end

# fit model
if ratid == 0
    println("Fitting $modelname model with objective $objfnname across all rats")
else
    println("Fitting $modelname model with objective $objfnname for rat $ratid")
end
d, ret = fitmodel(m, o, ϕini)
println("Optimizer stopped with return value $ret")
println("Minimum found at")
println(d[end,:])

if !(ret ∈ (:SUCCESS, :XTOL_REACHED, :STOPVAL_REACHED,
            :FTOL_REACHED, :MAXEVAL_REACHED, :MAXTIME_REACHED))
    println("Not writing to file, as optimization didn't complete successfully")
else
    # write to file
    outfile = datafile(modelname, objfnname, ratid)
    println("Writing data to $(outfile)")
    CSV.write(outfile, d)
end
