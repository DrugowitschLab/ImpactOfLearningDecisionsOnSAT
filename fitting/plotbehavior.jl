# plots behavioral data

using Printf, Gadfly, Colors
import Cairo, Fontconfig

include("behaviorstats.jl")

const nrats = 1   # treat all 4 rats as one rat
const t = Theme(minor_label_font_size=6pt, major_label_font_size=8pt)
const figpath = "../fitdata/fits"

function plotdataset(dataset::Symbol, stimx, magctrst, stimlabel, ctrstlabel, ttitle, ratid::Int = 0)
    d = load_data(dataset, ratid)
    # compute stats
    d_bystim, d_bymag = getpsychchronstats(d, nrats)
    d_seq, d_seqfit, d_biasid, d_biasmag = getseqstats(d, nrats)
    # generate plots
    pchronid, ppsychid = plotchronpsychid(d_bystim, stimx, stimlabel, ttitle, t)
    pchronmag, ppsychmag = plotchronpsychmag(d_bymag, magctrst, ctrstlabel, ttitle, t)
    pcondpsych = plotcondpsych(d_seq, d_seqfit, stimx, stimlabel, ttitle, t)
    pbiasid = plotbiasid(d_biasid, stimx, "prev. $stimlabel", ttitle, t)
    pbiasmag = plotbiasmag(d_biasmag, magctrst, "prev. $ctrstlabel", ttitle, t)

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
    outfile = ratid == 0 ? @sprintf("%s/behav_%s.pdf", figpath, string(dataset)) : 
                           @sprintf("%s/behav_%s_rat%d.pdf", figpath, string(dataset), ratid)
    draw(PDF(outfile, 8.5inch, 11inch), p)
end


function plotitlvddataset(ratid::Int = 0)
    mixctrst = [0.12, 0.36, 0.6, 1.0]
    nconcentr = 4
    d = load_data(:itlvd, ratid)
    # generate plot layers, iterating over concetrations
    lchron = Array{Gadfly.Layer}(undef, nconcentr, 3)
    lpsych = Array{Gadfly.Layer}(undef, nconcentr, 3)
    for i = 1:nconcentr  # iterate over different concentrations
        # stats for given concentration
        _, d_bymag = getpsychchronstats(d[d[:concentr] .== i,:], nrats)
        d_bymag[:tmin] = d_bymag[:tμ] - d_bymag[:tsem]
        d_bymag[:tmax] = d_bymag[:tμ] + d_bymag[:tsem]
        d_bymag[:pcorrmin] = d_bymag[:pcorrμ] - d_bymag[:pcorrsem]
        d_bymag[:pcorrmax] = d_bymag[:pcorrμ] + d_bymag[:pcorrsem]
        # the 8i-36 turns the stimmag into the (1-4) range
        d_bymag[:stimctrst] = mixctrst[d_bymag[:stimmag] .+ (8i - 36)]
        ti = copy(t)
        ti.default_color = RGB(([1.0, 1.0, 1.0] .- 0.2i)...)
        lchron[i,1] = layer(d_bymag, x="stimctrst", y="tμ", Geom.point, ti)[1]
        lchron[i,2] = layer(d_bymag, x="stimctrst", y="tμ", ymin="tmin", ymax="tmax",
                            Geom.errorbar, ti)[1]
        lchron[i,3] = layer(d_bymag, x="stimctrst", y="tμ", Geom.line, ti)[1]
        lpsych[i,1] = layer(d_bymag, x="stimctrst", y="pcorrμ", Geom.point, ti)[1]
        lpsych[i,2] = layer(d_bymag, x="stimctrst", y="pcorrμ", ymin="pcorrmin", ymax="pcorrmax",
                            Geom.errorbar, ti)[1]
        lpsych[i,3] = layer(d_bymag, x="stimctrst", y="pcorrμ", Geom.line, ti)[1]
    end
    # assemble plots
    pchronid = plot(lchron[:,1], lchron[:,2], lchron[:,3], 
        Guide.xlabel("concentration contrast", orientation=:horizontal),
        Guide.ylabel("reaction time [s]", orientation=:vertical),
        Guide.title("Chronometric curve, interleaved"),
        Scale.x_continuous(minvalue=0.0, maxvalue=1.0),
        Scale.y_continuous(minvalue=0.26, maxvalue=0.44))
    ppsychid = plot(lpsych[:,1], lpsych[:,2], lpsych[:,3],
        Guide.xlabel("concentration contrast", orientation=:horizontal),
        Guide.ylabel("correct choiced (fraction)", orientation=:vertical),
        Guide.title("Psychometric curve, interleaved"),
        Scale.x_continuous(minvalue=0.0, maxvalue=1.0),
        Scale.y_continuous(minvalue=0.5, maxvalue=1.0))
    p = vstack(pchronid, ppsychid)
    draw(PDF("$(figpath)/behav_itlvd.pdf", 8.5inch, 11inch), p)
end


const categ_x = [0, 0.2, 0.32, 0.44, 0.56, 0.68, 0.8, 1]
#const categ_ctrst = categ_x - flipdim(categ_x, 1)
const categ_ctrst = [0, 0, 0, 0, 0.12, 0.36, 0.6, 1.0]
plotdataset(:categ, categ_x, categ_ctrst,
            "odor A (frac)", "concentration contrast", "categorization")
for ratid in [1, 2, 3, 4]
    plotdataset(:categ, categ_x, categ_ctrst,
                "odor A (frac)", "concentration contrast",
                "categorization, rat$(ratid)", ratid)
end


const ident_x = [-4, -3, -2, -1, 1, 2, 3, 4]
const ident_ctrst = [-5.0, -5, -5, -5, -4, -3, -2, -1]
plotdataset(:ident, ident_x, ident_ctrst, 
            "concentration (10^x)", "concentration contrast", "identification")
for ratid in [1, 2, 3, 4]
    plotdataset(:ident, ident_x, ident_ctrst, 
                "concentration (10^x)", "concentration contrast",
                "identification, rat$(ratid)", ratid)
end


plotitlvddataset()