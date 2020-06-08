 # script to plot model predictions
#
# The script can be called in multiple ways:
#
# (i) discover the model parameters
#
# plotmodel.jl [model]
#
# (ii) plot model fits for previously fitted model
#
# plotmodel.jl [model] [objective]
#
# (iii) plot model fits for given parameter values
#
# plotmodel.jl [model] [p1] [p2] ...
#
# The plots are written to figs/[model]_[objective]_[categ/ident].pdf

using Printf, Random, CSV
import Cairo, Fontconfig

include("common.jl")
include("simmodel.jl")
include("behaviorstats.jl")
include("models.jl")

const nrats = 1
const t = Theme(minor_label_font_size=6pt, major_label_font_size=8pt)
const fitpath = "nrats1_0.1tndvar_fits"
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


# plots given dataset
function plotdataset(dataset::Symbol, m_behav, stimx, magctrst,
                     stimlabel, ctrstlabel, ttitle,
                     plotfilename)
    d = load_data(dataset)
    # compute stats
    d_bystim, d_bymag = getpsychchronstats(d, nrats)
    d_seq, d_seqfit, d_biasid, d_biasmag = getseqstats(d, nrats)
    m_bystim, m_bymag = getpsychchronstats(m_behav)
    m_seq, m_seqfit, m_biasid, m_biasmag = getseqstats(m_behav)

    # generate plots
    pchronid, ppsychid = plotchronpsychidm(d_bystim, m_bystim, stimx, stimlabel, ttitle, t)
    pchronmag, ppsychmag = plotchronpsychmagm(d_bymag, m_bymag, magctrst, ctrstlabel, ttitle, t)
    pcondpsych = plotcondpsychm(d_seq, m_seq, m_seqfit, stimx, "prev. $stimlabel", ttitle, t)
    pbiasid = plotbiasidm(d_biasid, m_biasid, stimx, "prev. $stimlabel", ttitle, t)
    pbiasmag = plotbiasmagm(d_biasmag, m_biasmag, magctrst, "prev. $ctrstlabel", ttitle, t)
    # plot all of those onto single "sheet"
    p = vstack(
        hstack(vstack(pchronid, ppsychid), vstack(pchronmag, ppsychmag)),
        vstack(
            hstack(pcondpsych[1], pcondpsych[2]),
            hstack(pcondpsych[3], pcondpsych[4])
        ),
        hstack(pbiasid, pbiasmag)
    )
    # output to file
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
    ϕ = [parse(Float64, s) for s in ARGS[2:end]]
end
println("Plotting model $modelname with parameters")
println(foldl((a, b) -> a * ", " * b, 
        [@sprintf("%s=%.2f", ϕns[i], ϕ[i]) for i in 1:ϕn]))

# simulate model
Random.seed!(0)
const evidence, ids = catevidence(ident_cs, categ_cs, 10, 10^5);
const m_ident, m_categ = simmodel(m, ϕ, evidence, ids)

# plot both conditions
const categ_x = [0, 0.2, 0.32, 0.44, 0.56, 0.68, 0.8, 1]
const categ_ctrst = [0, 0, 0, 0, 0.12, 0.36, 0.6, 1.0]
plotdataset(:categ, m_categ, categ_x, categ_ctrst,
            "odor A (frac)", "concentration contrast", "categorization",
            "$(figpath)/$(basefilename)_categ.pdf")
println("Plot written to $(figpath)/$(basefilename)_categ.pdf")

const ident_x = [-4, -3, -2, -1, 1, 2, 3, 4]
const ident_ctrst = [-5.0, -5, -5, -5, -4, -3, -2, -1]
plotdataset(:ident, m_ident, ident_x, ident_ctrst, 
            "concentration (10^x)", "concentration contrast", "identification",
            "$(figpath)/$(basefilename)_ident.pdf")
println("Plot written to $(figpath)/$(basefilename)_ident.pdf")
