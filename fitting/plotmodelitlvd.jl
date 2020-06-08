# script to plot model predictions for the interleaved dataset
#
# The script can be called in multiple ways:
#
# (i) discover the model parameters
#
# plotmodelitlvd.jl [model]
#
# (ii) plot model fits for previously fitted model
#
# plotmodelitlvd.jl [model] [objective]
#
# (iii) plot model fits for given parameter values
#
# plotmodelitlvd.jl [model] [p1] [p2] ...
#
# The plots are written to figs/[model]_[objective]_itlvd.pdf

using Random, Printf, CSV
import Fontconfig, Cairo

include("common.jl")
include("simmodel.jl")
include("behaviorstats.jl")
include("models.jl")

const nrats = 1
const t = Theme(minor_label_font_size=6pt, major_label_font_size=8pt)
const fitpath = "fits"
const datapath = "../fitdata/$(fitpath)"
const figpath = "../fitdata/$(fitpath)"

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


function plotitlvddataset(m_itlvd, plotfilename)
    mixctrst = [0.12, 0.36, 0.6, 1.0]
    nconcentr = 4
    d = load_data(:itlvd)
    # generate plot layers, iterating over concetrations
    lchron = Array{Gadfly.Layer}(undef, nconcentr, 4)
    lpsych = Array{Gadfly.Layer}(undef, nconcentr, 4)
    for i = 1:nconcentr  # iterate over different concentrations
        # stats for given concentration
        _, d_bymag = getpsychchronstats(d[d[:concentr] .== i,:], nrats)
        _, m_bymag = getpsychchronstats(m_itlvd[m_itlvd[:concentr] .== i,:], nrats)
        d_bymag[:tmin] = d_bymag[:tμ] - d_bymag[:tsem]
        d_bymag[:tmax] = d_bymag[:tμ] + d_bymag[:tsem]
        m_bymag[:tmin] = m_bymag[:tμ] - m_bymag[:tsem]
        m_bymag[:tmax] = m_bymag[:tμ] + m_bymag[:tsem]
        d_bymag[:pcorrmin] = d_bymag[:pcorrμ] - d_bymag[:pcorrsem]
        d_bymag[:pcorrmax] = d_bymag[:pcorrμ] + d_bymag[:pcorrsem]
        m_bymag[:pcorrmin] = m_bymag[:pcorrμ] - m_bymag[:pcorrsem]
        m_bymag[:pcorrmax] = m_bymag[:pcorrμ] + m_bymag[:pcorrsem]
        # the 8i-36 turns the stimmag into the (1-4) range
        d_bymag[:stimctrst] = mixctrst[d_bymag[:stimmag] .+ (8i - 36)]
        m_bymag[:stimctrst] = mixctrst[m_bymag[:stimmag] .+ (8i - 36)]
        ti = copy(t)
        ti.default_color = RGB(([1.0, 1.0, 1.0] .- 0.2i)...)
        tr = copy(ti)
        tr.lowlight_color = c->RGBA([[1.0, 1.0, 1.0] .- 0.2i; [0.2]]...)  # shaded color
        lchron[i,1] = layer(d_bymag, x="stimctrst", y="tμ", Geom.point, ti)[1]
        lchron[i,2] = layer(d_bymag, x="stimctrst", y="tμ", ymin="tmin", ymax="tmax",
                            Geom.errorbar, ti)[1]
        lchron[i,3] = layer(m_bymag, x="stimctrst", y="tμ", Geom.line, ti)[1]
        lchron[i,4] = layer(m_bymag, x="stimctrst", ymin="tmin", ymax="tmax",
                            Geom.ribbon, tr)[1]
        lpsych[i,1] = layer(d_bymag, x="stimctrst", y="pcorrμ", Geom.point, ti)[1]
        lpsych[i,2] = layer(d_bymag, x="stimctrst", y="pcorrμ", ymin="pcorrmin", ymax="pcorrmax",
                            Geom.errorbar, ti)[1]
        lpsych[i,3] = layer(m_bymag, x="stimctrst", y="pcorrμ", Geom.line, ti)[1]
        lpsych[i,4] = layer(m_bymag, x="stimctrst", ymin="pcorrmin", ymax="pcorrmax",
                            Geom.ribbon, tr)[1]
    end
    # assemble plots
    pchronid = plot(lchron[:,1], lchron[:,2], lchron[:,3], lchron[:,4],
        Guide.xlabel("concentration contrast", orientation=:horizontal),
        Guide.ylabel("reaction time [s]", orientation=:vertical),
        Guide.title("Chronometric curve, interleaved"),
        Scale.x_continuous(minvalue=0.0, maxvalue=1.0),
        Scale.y_continuous(minvalue=0.26, maxvalue=0.44))
    ppsychid = plot(lpsych[:,1], lpsych[:,2], lpsych[:,3], lpsych[:,4],
        Guide.xlabel("concentration contrast", orientation=:horizontal),
        Guide.ylabel("correct choiced (fraction)", orientation=:vertical),
        Guide.title("Psychometric curve, interleaved"),
        Scale.x_continuous(minvalue=0.0, maxvalue=1.0),
        Scale.y_continuous(minvalue=0.5, maxvalue=1.0))
    p = vstack(pchronid, ppsychid)
    draw(PDF(plotfilename, 8.5inch, 11inch), p)
end


# process arguments
if isempty(ARGS)
    error("Plotting script needs to be called with at least on argument")
end
const modelname = lowercase(ARGS[1])
const m = createmodel(modelname)
const ϕns = ϕnames(m)
const ϕn = length(ϕns)
if length(ARGS) == 1
    println("Model $modelname has $ϕn parameters:")
    println(foldl((a, b) -> a * ", " * b, [string(i) for i in ϕns]))
    quit()

elseif length(ARGS) == 2
    basefilename = "$(modelname)_$(ARGS[2])"
    fitfilename = "$(datapath)/fit_$(basefilename).csv"
    if !isfile(fitfilename)
        error("Fitting data $fitfilename not found")
    end
    println("Loading fitting data from $fitfilename")
    d = CSV.read(fitfilename)
    (maxllh, maxi) = findmax(d[:llh])
    if maxi != size(d,1) && maxllh > d[end,end]
        warn("Maximum llh $maxllh found in line $maxi instead of last line")
    end
    ϕ = Vector(d[maxi,1:end-1])
else
    basefilename = modelname
    if length(ARGS) != ϕn+1
        error("Expected $(ϕn+1) parameters for model $modelname")
    end
    # process and print arguments
    ϕ = [float(s) for s in ARGS[2:end]]
end
println("Plotting model $modelname with parameters")
println(foldl((a, b) -> a * ", " * b, 
        [@sprintf("%s=%.2f", ϕns[i], ϕ[i]) for i in 1:ϕn]))

# simulate model
Random.seed!(0)
const evidence, ids = catitlvdevidence(itlvd_cs, 10, 10^5)
const m_itlvd = simitlvdmodel(m, ϕ, evidence, ids)

# plot itlvd condition
plotitlvddataset(m_itlvd, "$(figpath)/$(basefilename)_itlvd.pdf")
println("Plot written to $(figpath)/$(basefilename)_itlvd.pdf")
