# script that outputs some statistics of the model fits

using DataFrames, Gadfly, Colors, MAT, CSV
import Cairo, Fontconfig

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
const t = Theme(minor_label_font_size=6pt,
                major_label_font_size=6pt,
                grid_line_width=0pt, grid_color=RGB(1.0, 1.0, 1.0))
const fitpath = "fits"
const datapath = "../fitdata/$(fitpath)"
const figpath = "../fitdata/$(fitpath)" 

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
# const modelnames = ["opt", "optlapse", "optcollapse",
#                     "adflapse", "adfcollapse", "adfbiaslapse", "adfcolbiaslapse",
#                     "gammalapse", "gammacollapse", "gammabiaslapse", "gammacolbiaslapse",
#                     "deltalapse", "deltacollapse", "deltabiaslapse", "deltacolbiaslapse",
#                     "deltaexpcolbiaslapse"]
const modelnames = ["opt", "optlapse", "optcollapse", "optcolbiaslapse",
                    "adflapse", "adfcollapse", "adfbiaslapse", "adfcolbiaslapse",
                    "randomcolbiaslapse", "randomcollapse",
                    "gammalapse", "gammacollapse", "gammabiaslapse", "gammacolbiaslapse",
                    "deltaexpcolbiaslapse", "lcalapse", "lcacollapse"]
# const modelnames = ["opt", "optlapse", "optcollapse",
#                     "adflapse", "adfcollapse", "adfbiaslapse", "adfcolbiaslapse",
#                     "gammalapse", "gammacollapse", "gammabiaslapse", "gammacolbiaslapse",
#                     "deltaexpcolbiaslapse"]
const basemodel = "adfcolbiaslapse"  # base model for model comparison
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
const objfns = ["psychchron1", "psychchronseq1"]
@assert issubset(Set(objfns), Set(keys(objfndict)))

# model fit criteria, converted to log-evidence in log(10)
BIC(avgllh, ϕn, trials) = (trials * avgllh - 0.5ϕn * log(trials)) / log(10)
AIC(avgllh, ϕn, trials) = (trials * avgllh - ϕn) / log(10)
AICc(avgllh, ϕn, trials) = AIC(avgllh, ϕn, trials) - (ϕn * (ϕn+1) / (trials-ϕn-1)) / log(10)

# returns number of trials in categ/ident dataset as vector
datatrials(ratid::Int=0) = [size(d, 1) / nrats 
    for d in (load_data(:categ, ratid), load_data(:ident, ratid))]
datatrials(objfn::AbstractString, ratid::Int=0) = sum(datatrials(ratid)[objfndict[objfn][2]])

# returns dataframe with best-fitting parameters and avg. log-likelihood
function loadmodelfits(modelname, objfnname, ratid::Int=0)

    fitfilename = ratid == 0 ? "$(datapath)/fit_$(modelname)_$(objfnname).csv" :
                               "$(datapath)/fit_$(modelname)_$(objfnname)_rat$(ratid).csv"
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

# prints the fit parameters for all objective functions for a given model and given rats
function printfitparams(modelname, ratids::Vector{Int}=[0])
    # use first objective function to get parameter names
    d = loadmodelfits(modelname, objfns[1], ratids[1])
    ϕnames = names(d)
    # print parameters
    objfnlen = maximum([length(s) for s in objfns])
    ratidlen = maximum([floor(Int, ratid > 0 ? log10(ratid) : 0.0)+1 
                        for ratid in ratids]) + 3
    ratidstrs = [ratid > 0 ? "rat$(ratid)" : "all" for ratid in ratids]
    println(repeat("-", objfnlen + ratidlen + 1), "  ",
            join([@sprintf("%7s", string(ϕi)) for ϕi in ϕnames], "  "))
    # print fits
    for objfn in objfns
        for ratidx in 1:length(ratids)
            d = loadmodelfits(modelname, objfn, ratids[ratidx])
            if ratidx == 1
                print(repeat(" ", objfnlen - length(objfn)), objfn)
            else
                print(repeat(" ", objfnlen))
            end
            println(repeat(" ", ratidlen - length(ratidstrs[ratidx]) + 1),
                    ratidstrs[ratidx], "  ",
                    join([@sprintf("%7.2f", x) for x in Vector(d)], "  "))
        end
    end
end

# prints the fit quality for all models for the given objective function
function printfitquality(objfn, ratids::Vector{Int}=Int[])
    # remove 0's from ratids's vector, as they are default
    fitmeasures = ["BIC", "AIC", "AICc"]
    ratids = ratids[ratids .!= 0]
    ratnum = length(ratids)
    modelnum = length(modelnames)
    ratidlen = maximum([floor(Int, log10(ratid))+1 for ratid in ratids]) + 3
    ratidstrs = ["rat$(ratid)" for ratid in ratids]
    modellen = max(maximum([length(s) for s in modelnames]), 3) # include "all" here
    println(repeat("-", modellen + ratidlen + 1), "  ",
            join([@sprintf("%10s      ", s) for s in ["avg. llh", "BIC", "AIC", "AICc"]], "  "))
    fitstats = Array{Float64}(undef, ratnum, modelnum, length(fitmeasures)+1)
    fitstatsall = Array{Float64}(undef, modelnum, length(fitmeasures))
    for modelidx = 1:modelnum
        # fits across all rats
        modelname = modelnames[modelidx]
        d = loadmodelfits(modelname, objfn)
        avgllh = d[:llh]
        ϕn = length(d) - 1
        trials = datatrials(objfn)
        fitq = [avgllh; [f(avgllh, ϕn, trials) for f in [BIC, AIC, AICc]]]
        println(repeat(" ", modellen - length(modelname)), modelname, "  ",
                repeat(" ", ratidlen - 3), "all ",
                join([@sprintf("%10.2f      ", x) for x in fitq], "  "))
        fitstatsall[modelidx, :] = fitq[2:end]
        # fits for individual rats
        if length(ratids) == 0
            continue
        end
        for ratidx in 1:ratnum
            ratid = ratids[ratidx]
            d = loadmodelfits(modelname, objfn, ratid)
            avgllh = d[:llh]
            ϕn = length(d) - 1
            trials = datatrials(objfn, ratid)
            fitstats[ratidx,modelidx,:] = [avgllh; [f(avgllh, ϕn, trials) for f in [BIC, AIC, AICc]]]
            println(repeat(" ", modellen + 1), " ", 
                    repeat(" ", ratidlen - length(ratidstrs[ratidx])), ratidstrs[ratidx], " ",
                    join([@sprintf("%10.2f      ", x) for x in fitstats[ratidx,modelidx,:]], "  "))
        end
        # averages across rats
        println(repeat(" ", modellen + 1), " ", repeat(" ", ratidlen - 3), "avg ",
                join([@sprintf("%10.2f±%6.2f", mean(fitstats[:,modelidx,i]), √(var(fitstats[:,modelidx,i]) / ratnum)) 
                      for i in 1:ratnum], " "))
    end
    return fitstats[:,:,2:end], fitstatsall
end

# returns the absolute and relative fit quality plots for all models for the 
# given statistics
function genfitqualityplots(fitstats, fitstatsall, ratids::Vector{Int})
    fitmeasurenames = ["BIC", "AIC", "AICc"]
    ratnum = length(ratids)
    modelnum = length(modelnames)
    @assert(ratnum > 0)
    @assert(size(fitstats, 1) == ratnum)
    @assert(size(fitstats, 2) == modelnum)
    @assert(size(fitstats, 3) == length(fitmeasurenames))
    @assert(size(fitstatsall, 1) == modelnum)
    @assert(size(fitstatsall, 2) == length(fitmeasurenames))
    # separate plot per fit measure
    ps = Array{Any}(undef, length(fitmeasurenames))
    for fitmeasureidx = 1:length(fitmeasurenames)
        substats = fitstats[:,:,fitmeasureidx]
        fitmeasurename = fitmeasurenames[fitmeasureidx]
        # average statistics
        fitmean = mean(substats, dims=1)
        fitse = sqrt.(var(substats, dims=1) / ratnum)
        ps[fitmeasureidx] = plot(
            layer(x=1:modelnum, y=fitstatsall[:,fitmeasureidx] / ratnum,
                  Geom.point, Theme(t, default_color=RGB(0.8, 0.0, 0.0))),
            layer(x=1:modelnum, y=fitmean, ymin=fitmean.-fitse, ymax=fitmean.+fitse,
                  Geom.point, Geom.errorbar),
            layer(x=vec(repeat(collect(1:modelnum)', ratnum)), y=vec(substats),
                  Geom.point, Stat.x_jitter(range=0.4), Theme(t, default_color=RGB(0.5, 0.5, 0.5))),
            layer(yintercept=[0.0], Geom.hline(color="black")),
            Guide.xticks(ticks=collect(1:modelnum), orientation=:vertical),
            Scale.x_discrete(labels=(i->modelnames[i])),
            Guide.xlabel(""),
            Guide.ylabel("log model evidence (using $(fitmeasurename); log10)", orientation=:vertical),
            Coord.cartesian(ymax=0.0))
    end
    p_abs = vstack(ps...)
    # plot comparison to base model
    basemodelidx = findall(modelnames .== basemodel)
    fitdiff = fitstats[:, basemodelidx, :] .- fitstats[:, modelnames .!= basemodel, :]
    fitdiffall = fitstatsall[basemodelidx, :] .- fitstatsall[modelnames .!= basemodel, :]
    diffmodels = modelnames[modelnames .!= basemodel]
    ps = Array{Any}(undef, length(fitmeasurenames))
    for fitmeasureidx = 1:length(fitmeasurenames)
        substats = fitdiff[:,:,fitmeasureidx]
        fitmeasurename = fitmeasurenames[fitmeasureidx]
        # average statistics
        fitmean = mean(substats, dims=1)
        fitse = sqrt.(var(substats, dims=1) / ratnum)
        ps[fitmeasureidx] = plot(
            layer(x=1:(modelnum-1), y=fitdiffall[:,fitmeasureidx] / ratnum,
                  Geom.point, Theme(t, default_color=RGB(0.8, 0.0, 0.0))),
            layer(x=1:(modelnum-1), y=fitmean, ymin=fitmean.-fitse, ymax=fitmean.+fitse,
                  Geom.point, Geom.errorbar),
            layer(x=vec(repeat(collect(1:(modelnum-1))', ratnum)), y=vec(substats),
                  Geom.point, Stat.x_jitter(range=0.4), Theme(t, default_color=RGB(0.5, 0.5, 0.5))),
            layer(yintercept=[0.0], Geom.hline(color="black")),
            Guide.xticks(ticks=collect(1:(modelnum-1)), orientation=:vertical),
            Scale.x_discrete(labels=(i->diffmodels[i])),
            Guide.xlabel(""),
            Guide.ylabel("log model evidence (using $(fitmeasurename); log10)\n$(basemodel) vs. others",
                         orientation=:vertical))
    end
    p_rel = vstack(ps...)
    return p_abs, p_rel
end


# plots the fit quality for all models for the given objective function
function plotfitquality(objfn, ratids::Vector{Int})
    fitmeasures = [BIC, AIC, AICc]
    ratnum = length(ratids)
    @assert(ratnum > 0)
    # collect model fit statistics
    modelnum = length(modelnames)
    fitstats = Array{Float64}(undef, ratnum, modelnum, length(fitmeasures))
    fitstatsall = Array{Float64}(undef, modelnum, length(fitmeasures))
    for modelidx = 1:modelnum
        # first global, then for individual rats
        d = loadmodelfits(modelnames[modelidx], objfn)
        fitstatsall[modelidx, :] = [
            fitmeasure(d[:llh], length(d)-1, datatrials(objfn))
            for fitmeasure in fitmeasures]
        for ratidx = 1:ratnum
            d = loadmodelfits(modelnames[modelidx], objfn, ratids[ratidx])
            fitstats[ratidx, modelidx, :] = [
                fitmeasure(d[:llh], length(d)-1, datatrials(objfn, ratids[ratidx]))
                for fitmeasure in fitmeasures]
        end
    end
    # generate and save plots
    p_abs, p_rel = genfitqualityplots(fitstats, fitstatsall, ratids)
    draw(PDF("$figpath/fitqual_$(objfn).pdf", 8.5inch, 11inch), p_abs)
    println("Plot writting to $figpath/fitqual_$(objfn).pdf")
    draw(PDF("$figpath/fitqual_$(objfn)_relativ.pdf", 8.5inch, 11inch), p_rel)
    println("Plot written to $figpath/fitqual_$(objfn)_relativ.pdf")
end


# returns average log-likelihood for one objective function, using another
# objective function for the parameters. It returns this average
# log-likelihood and the number of trials, and the number of parameters
function avgcrossllh(paramobjfn, evalobjfn, modelname, ratid::Int=0)
    # trial sequence to fit data on
    d_ident = load_data(:ident, ratid)
    d_categ = load_data(:categ, ratid)
    # instantiate model and get model parameters
    m = modeldict[modelname]()
    ϕ = Vector(loadmodelfits(modelname, paramobjfn)[ϕnames(m)])
    ϕn = length(ϕ)
    # evaluate objective function
    evalobjfntype = objfndict[evalobjfn][1]
    evalobjfntrials = datatrials(evalobjfn, ratid)
    Random.seed!(0)
    evidence, ids = catevidence(ident_cs, categ_cs, 10, 10^5)
    o = evalobjfntype(m, evidence, ids, burnin, d_ident, d_categ, nrats)
    avgllh = llh(o, ϕ)
    return avgllh, evalobjfntrials, ϕn
end


# prints the cross fit quality, using parameters for the paramobjfn and
# evaluating it on the evalobjfn. It returns an array of fitmeasures per rat
# and model, and one per model for the meta rat (fits across all rats)
function printcrossfitquality(paramobjfn, evalobjfn, ratids::Vector{Int})
    fitmeasures = [(BIC, "BIC"), (AIC, "AIC"), (AICc, "AICc")]
    ratids = ratids[ratids .!= 0]
    ratnum = length(ratids)
    ratidlen = maximum([floor(Int, log10(ratid))+1 for ratid in ratids]) + 3
    ratidstrs = ["rat$(ratid)" for ratid in ratids]
    @assert(ratnum > 0)
    # collect model fit statistics
    modelnum = length(modelnames)
    modellen = max(maximum([length(s) for s in modelnames]), 3) # include "all" here
    fitstats = Array{Float64}(undef, ratnum, modelnum, length(fitmeasures)+1)
    fitstatsall = Array{Float64}(undef, modelnum, length(fitmeasures))
    for modelidx = 1:modelnum
        modelname = modelnames[modelidx]
        # fits across all rats
        avgllh, evalobjfntrials, ϕn = avgcrossllh(paramobjfn, evalobjfn, modelname)
        fitq = [avgllh; [f(avgllh, ϕn, evalobjfntrials) for f in [BIC, AIC, AICc]]]
        println(repeat(" ", modellen - length(modelname)), modelname, "  ",
                repeat(" ", ratidlen - 3), "all ",
                join([@sprintf("%10.2f      ", x) for x in fitq], "  "))
        fitstatsall[modelidx, :] = fitq[2:end]
        # fits for individual rats
        if length(ratids) == 0
            continue
        end
        avgllhs = Array{Float64}(undef, ratnum)
        for ratidx in 1:ratnum
            ratid = ratids[ratidx]
            avgllh, evalobjfntrials, ϕn = avgcrossllh(paramobjfn, evalobjfn, modelname, ratid)
            fitq = [avgllh; [f(avgllh, ϕn, evalobjfntrials) for f in [BIC, AIC, AICc]]]
            println(repeat(" ", modellen + 1), " ", 
                    repeat(" ", ratidlen - length(ratidstrs[ratidx])), ratidstrs[ratidx], " ",
                    join([@sprintf("%10.2f      ", x) for x in fitq], "  "))
            fitstats[ratidx,modelidx,:] = fitq
        end
        # averages across rats
        println(repeat(" ", modellen + 1), " ", repeat(" ", ratidlen - 3), "avg ",
                join([@sprintf("%10.2f±%6.2f", mean(fitstats[:,modelidx,i]), 
                               √(var(fitstats[:,modelidx,i]) / ratnum)) 
                      for i in 1:ratnum], " "))
    end
    # don't return the avgllh fit for individual rats
    return fitstats[:,:,2:end], fitstatsall
end


# writes the fit quality measures to a MAT file
function writefitquality(fitstats, fitstatsall, ratids::Vector{Int}, fname)
    fitmeasurenames = ["BIC", "AIC", "AICc"]
    ratnum = length(ratids)
    modelnum = length(modelnames)
    @assert(ratnum > 0)
    @assert(size(fitstats, 1) == ratnum)
    @assert(size(fitstats, 2) == modelnum)
    @assert(size(fitstats, 3) == length(fitmeasurenames))
    @assert(size(fitstatsall, 1) == modelnum)
    @assert(size(fitstatsall, 2) == length(fitmeasurenames))
    # write matrices to MAT file
    d = Dict(
        "fitstats" => fitstats,
        "fitstatsall" => fitstatsall,
        "fitmeasures" => fitmeasurenames,
        "modelnames" => modelnames,
        "ratids" => ratids
        )
    matwrite(fname, d)
end


# print model parameters
# for modelname in modelnames
#     println("Model $modelname")
#     printfitparams(modelname, [0, 1, 2, 3, 4])
#     println()
# end
# println()

# print model fit statistics
for objfn in objfns
    println("Objective function $objfn")
    fitstats, fitstatsall = printfitquality(objfn, [0, 1, 2, 3, 4])
    fname = "$datapath/fitqual_$(objfn).mat"
    writefitquality(fitstats, fitstatsall, [1, 2, 3, 4], fname)
    println("Data written to $fname")
    println()
end

# plot model fit stats
for objfn in objfns
    println("Generating plots for objective function $objfn")
    plotfitquality(objfn, [1, 2, 3, 4])
end

# # print model fit statistics when evaluated with psychchronseq1 objective
println("Fitted on psychchron1, evaluated on psychchronseq1")
fitstats, fitstatsall = printcrossfitquality("psychchron1", "psychchronseq1", [1, 2, 3, 4])
fname = "$datapath/fitqual_psychchron1_to_psychchronseq1.mat"
writefitquality(fitstats, fitstatsall, [1, 2, 3, 4], fname)
println("Data written to $fname")
println()
println("Generating plots")
p_abs, p_rel = genfitqualityplots(fitstats, fitstatsall, [1, 2, 3, 4])
draw(PDF("$figpath/fitqual_psychchron1_to_psychchronseq1.pdf", 8.5inch, 11inch), p_abs)
println("Plot written to $figpath/fitqual_psychchron1_to_psychchronseq1.pdf")
draw(PDF("$figpath/fitqual_psychchron1_to_psychchronseq1_relativ.pdf", 8.5inch, 11inch), p_rel)
println("Plot written to $figpath/fitqual_psychchron1_to_psychchronseq1_relativ.pdf")

