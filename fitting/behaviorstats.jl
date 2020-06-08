# scripts to analyse behavioral data in various ways

using SpecialFunctions, Distributions, Optim, Calculus, MAT, DataFrames, Gadfly, Colors

# constants and common functions
const sqrttwo = √(2)
const sqrttwoπ = √(2 * π)
Φ(x) = (1 + erf(x / sqrttwo)) / 2


###############################################################################
# loading experimental data
#
# Most of the dataset is self-explanatory.
# Stimulus ranges from 1 to 8 (Stimulus = 1..4 -> right, 5..8 -> left). OSD is
# odor sampling duration [s]. Outcome is 1 (rewarded), 0 (not rewarded due to
# wrong choice), or NaN (not rewarded due to too fast choice).
#
# For the interleaved :itlvd dataset, the format is slightly different, in
# that Stimulus ranges from 1 to 32, with the following ordering:
#  1.. 4 (right easy->hard)  8.. 5 (left hard->easy) concentration 0.1
#  9..12 (right easy->hard) 16..13 (left hard->easy) concentration 0.01
# 17..20                    24..21                   concentration 0.001
# 25..28                    32..29                   concentration 0.0001
# In the below, the Id's for left will be inverted to recover the 1-8, 9-16, ...
# ordering.
#
# On top of the stimulus id, the method adds the following stimulus identifier
# mixw 1  mixmag 4  for 100/  0 mix | concentr 4  for 0.1     concentration
# mixw 2  mixmag 3  for  80/ 20 mix | concentr 3  for 0.01    concentration
# mixw 3  mixmag 2  for  68/ 32 mix | concentr 2  for 0.001   concentration 
# mixw 4  mixmag 1  for  56/ 44 mix | concentr 1  for 0.0001  concentration
# mixw 5  mixmag 1  for  44/ 56 mix
# mixw 6  mixmag 2  for  32/ 68 mix
# mixw 7  mixmag 3  for  20/ 80 mix
# mixw 8  mixmag 4  for   0/100 mix

# dataset is :categ, :ident, :itlvd
# a ratid of 0 causes the function to return all rats
function load_data(dataset::Symbol, ratid::Int = 0)
    @assert dataset == :categ || dataset == :ident || dataset == :itlvd
    @assert ratid ∈ [0, 1, 2, 3, 4]  # only 4 rats
    # load MAT file
    d = dataset == :categ ?
        matread("../data/DATA CAT.mat")["DATA_cat"] :
        (dataset == :ident ? 
         matread("../data/DATA ID.mat")["DATA_id"] :
         matread("../data/DATA INT.mat")["DATA_int"])
    stimid = round.(Int, vec(d["Stimulus"]))
    resp = round.(Int, vec(d["ChoiceDir"]))
    trials = length(stimid)
    if dataset == :itlvd
        # :itlvd dataset
        @assert minimum(stimid) == 1 && maximum(stimid) == 32
        corrresp = zeros(Int, trials)
        stimmag = zeros(Int, trials)
        corr = zeros(Bool, trials)
        mixw = zeros(Int, trials)
        mixmag = zeros(Int, trials)
        concentr = zeros(Int, trials)
        # process stimulus properties separately per concentration (1-8, 9-16, ...)
        for i = 1:4
            # reverse stimulus ordering for 5 -> 8, 13 -> 16, etc.
            righttrials = (stimid .>= 8i-3) .& (stimid .<= 8i)
            lefttrials = (stimid .>= 8i-7) .& (stimid .<= 8i-4)
            alltrials = righttrials .| lefttrials
            stimid[righttrials] = (16i - 3) .- stimid[righttrials]
            corrresp[righttrials] .= 1 # left = 0, right = 1
            stimmag[righttrials] .= stimid[righttrials]
            stimmag[lefttrials] .= (16i - 7) .- stimid[lefttrials]
            corr[lefttrials] = resp[lefttrials] .== 0
            corr[righttrials] = resp[righttrials] .== 1
            mixw[alltrials] = stimid[alltrials] .- 8(i-1)       # to 1-8 range
            mixmag[alltrials] = stimmag[alltrials] .- (8i - 4)  # to 1-4 range
            concentr[alltrials] .= 5-i
        end
    else
        # :categ or :ident dataset - those are simpler
        @assert minimum(stimid) == 1 && maximum(stimid) == 8
        corrresp = Int[i < 5 ? 0 : 1 for i in stimid]
        stimmag = Int[i < 5 ? 9 - i : i for i in stimid]
        corr = (stimid .>= 5) .& (resp .== 1) .| (stimid .<= 4) .& (resp .== 0)
        if dataset == :categ
            mixw = copy(stimid)
            mixmag = stimmag .- 4
            concentr = 4 * ones(Int, trials)
        else
            mixw = 7corrresp .+ 1  # either 1 or 8
            mixmag = 4 * ones(Int, trials)
            concentr = stimmag .- 4
        end
    end
    rew = [!isnan(i) && round(Int, i) == 1 for i in vec(d["Outcome"])]
    # turn into DataFrame
    df = DataFrame(rat = round.(Int, vec(d["RAT"])),
                   session = round.(Int, vec(d["Session"])),
                   trial = round.(Int, vec(d["Trial"])),
                   stimid = stimid,
                   stimmag = stimmag,
                   mixw = mixw,
                   mixmag = mixmag,
                   concentr = concentr,
                   corrresp = corrresp,
                   previd = [stimid[end]; stimid[1:end-1]],    # ign. boundaries
                   prevmag = [stimmag[end]; stimmag[1:end-1]], # "
                   resp = resp,
                   t = vec(d["OSD"]),
                   corr = corr,
                   rew = rew,
                   prevcorr = [corr[end]; corr[1:end-1]],      # ign. boundaries
                   prevrew = [rew[end]; rew[1:end-1]])
    if ratid != 0
        df = df[df[:rat] .== ratid, :]
    end
    return df
end


###############################################################################
# fitting psychometric function, assuming the probabilistic model
#
# p(y = 1 | x) = (1-pl) Phi((x - μ) / σ) + pl pb
#
# with mean μ, standard deviation σ, lapse rate pl and lapse bias bp.
# THe log-likelihood is defined over J xj's, nj's and pj's, where nj is the
# number of observations for the corresponding xj, and pj = sum yj / nj is the
# fraction of y's that are 1 at this xj.
#
# The parameter vector is θ = [μ, σ, pl, pb]

# p(y = 1 | x) for given parameters
psychpy1(θ::AbstractVector, x) = (1 - θ[3]) .* Φ.((x .- θ[1]) ./ θ[2]) .+ θ[3] * θ[4]

# log-likelihood
function psychllh(θ::AbstractVector,
                  x::AbstractVector, n::AbstractVector, p::AbstractVector)
    @assert length(x) == length(n) == length(p)
    @assert length(θ) == 4

    # parameter bound checking
    μ, σ, pl, pb = θ
    σ > 0.0 && 0.0 <= pl <= 1.0 && 0.0 <= pb <= 1.0 || return -Inf
    # compute llh
    llh = 0.0
    for j = 1:length(x)
        res = (x[j] - μ) / σ
        llh += n[j] * (p[j] * log((1-pl) * Φ(res) + pl * pb) + 
                       (1-p[j]) * log((1-pl) * Φ(-res) + pl * (1-pb)))
    end
    return llh
end

# gradient g
function psychgrad(θ::AbstractVector,
                   x::AbstractVector, n::AbstractVector, p::AbstractVector)
    @assert length(x) == length(n) == length(p)
    @assert length(θ) == 4
    g = Vector{Float64}(undef, 4)

    # parameter bound checking
    μ, σ, pl, pb = θ
    if !(σ > 0.0 && 0.0 <= pl <= 1.0 && 0.0 <= pb <= 1.0)
        g[1] = 0.0
        g[2] = σ > 0.0 ? 0.0 : Inf
        g[3] = 0.0 <= pl <= 1.0 ? 0.0 : (pl < 0.0 ? Inf : -Inf)
        g[4] = 0.0 <= pb <= 1.0 ? 0.0 : (pb < 0.0 ? Inf : -Inf)
        return g
    end

    # compute gradient
    g[:] .= 0.0
    for j = 1:length(x)
        res = (x[j] - μ) / σ
        Nres = exp(- 0.5 * abs2(res)) / (sqrttwoπ * σ)
        Φres, Φnres = Φ(res), Φ(-res)
        plΦres, plΦnres = (1-pl) * Φres + pl * pb, (1-pl) * Φnres + pl * (1-pb)
        plfrac = (1-p[j]) / plΦnres - p[j] / plΦres

        g[1] += n[j] * (1-pl) * Nres * plfrac
        g[2] += n[j] * (1-pl) * Nres * res * plfrac
        g[3] += n[j] * (p[j] * (pb - Φres) / plΦres + 
                        (1 - p[j]) * (1 - Φnres - pb) / plΦnres)
        g[4] -= n[j] * pl * plfrac
    end
    return g
end
# returing llh, modifying gradient, to be used by NLopt.jl
function psychllhgrad!(θ::AbstractVector, g::AbstractVector, 
                       x::AbstractVector, n::AbstractVector, p::AbstractVector)
    if !isempty(g)
        g[:] = psychgrad(θ, x, n, p)
    end
    return psychllh(θ, x, n, p)
end
# modifying negative gradient, to be used by Optim.jl
function negpsychgrad!(θ::AbstractVector, g::AbstractVector, 
                       x::AbstractVector, n::AbstractVector, p::AbstractVector)
    g[:] = -psychgrad(θ, x, n, p)
    return g
end

# Hessian h, by finite differences
psychhess(θ, x, n, p) = hessian(θt -> psychllh(θt, x, n, p), θ)
# Hessian diagonal by taking (finite difference) derivative of gradient
function psychhessdiag(θ, x, n, p)
    @assert length(θ) == 4
    psychgradi(i, θi) = (θtmp = copy(θ); θtmp[i] = θi; psychgrad(θtmp, x, n, p)[i])
    return Float64[derivative(θi -> psychgradi(i, θi), θ[i]) for i in 1:4]
end


###############################################################################
# functions analyzing behavioral data
#
# For psycometric/chronometric functions, objective correctness of the choices
# is used. For sequential dependencies, the measure of 'correctness' is if
# reward has been given in the previous trial.

# statistics that also work for empty lists
_0mean(x) = isempty(x) ? zero(eltype(x)) : mean(x)
_0var(x) = isempty(x) ? Inf : var(x)
_0sem(x, n) = n == 0 ? Inf : √(_0var(x) / n)
_0sem(x) = _0sem(x, length(x))

# returns various RT / choice statistics over the whole set
_tcstats(df, datasets::Int=1) = DataFrame(
    n = size(df, 1) / datasets,
    nr = sum(df[:resp] .== 1) / datasets,
    ncorr = sum(df[:corr]) / datasets,
    pcorrμ = _0mean(df[:corr]),
    pcorrsem = √(datasets * _0mean(df[:corr]) .* (1 - _0mean(df[:corr])) ./ size(df, 1)),
    prμ = _0mean(df[:resp] .== 1),
    prsem = √(datasets * _0mean(df[:resp] .== 1) .* (1 - _0mean(df[:resp] .== 1)) ./ size(df, 1)),
    tμ = _0mean(df[:t]),
    tσ = √(_0var(df[:t])),
    tsem = _0sem(df[:t], size(df, 1) / datasets),
    tcorrμ = _0mean(df[df[:corr], :t]),
    tcorrσ = √(_0var(df[df[:corr], :t])),
    tcorrsem = _0sem(df[df[:corr], :t], sum(df[:corr]) / datasets),
    tincorrμ = _0mean(df[.!df[:corr], :t]),
    tincorrσ = √(_0var(df[.!df[:corr], :t])),
    tincorrsem = _0sem(df[.!df[:corr], :t], sum(.!df[:corr]) / datasets)
    )

# returns psychometric/chronometric statistics, once ordered by stimid, once by
# stimmag.
# If datasets > 1, the returned sem's are average sem's per dataset, assuming
# that the provided data is from the given number of datasets.
getpsychchronstats(df, datasets::Int=1) = (
    by(df, :stimid, df -> _tcstats(df, datasets)),
    by(df, :stimmag, df -> _tcstats(df, datasets)))

# returns sequential statistics, seq_df, fit_df, biasid_df, biasmag_df
# seq_df returns sequential choice statistics, conditional on previous reward,
# previous stimulus id, and current stimulus id.
# fit_df fits these sequential psychometric curve, with one fit per
# (pre. rew, prev. stim) combination. The function first performs a global
# fit (stimid/stimmag = 0) to determine the slope of the psychometric curve,
# and fixes this slope in all subsequent (conditional) fits.
# biasid_df and biasmag_df use the fits to compute the biases, either per
# id or magnitude of the previous stimulus.
function getseqstats(df, datasets::Int=1)
    seq_df = by(df, [:stimid, :previd, :prevrew], df -> _tcstats(df, datasets))

    # find psychometric curve fit parameters
    fit_df = DataFrame(previd = Int[], prevrew = Bool[],
                       μ = Float64[], σ = Float64[], pl = Float64[], pb = Float64[],
                       μsd = Float64[], σsd = Float64[], plsd = Float64[], pbsd = Float64[])
    # fit overall psychometric curve
    psych_df = by(df, :stimid,
                  df -> DataFrame(n = size(df, 1), pr = _0mean(df[:resp])))
    x, n, p = psych_df[:stimid], psych_df[:n] ./ datasets, psych_df[:pr]
    θini = [mean(x), 1.0, 0.1, 0.5]
    θmin = [-Inf, 0.0, 0.0, 0.0]
    θmax = [Inf, Inf, 1.0, 1.0]
    res = optimize(
        OnceDifferentiable(
            θ -> -psychllh(θ, x, n, p),
            (g, θ) -> negpsychgrad!(θ, g, x, n, p), θini),
        θmin, θmax, θini, Fminbox(LBFGS()))
    globalθ = Optim.minimizer(res)
    globalθsd = sqrt.(max.(0.0, -1 ./ psychhessdiag(globalθ, x, n, p)))
    push!(fit_df, [0, true, 
                   globalθ[1], globalθ[2], globalθ[3], globalθ[4],
                   globalθsd[1], globalθsd[2], globalθsd[3], globalθsd[4]])

    # fit per-condition psychometric curve, fixing slope to above slope
    stimids = sort(unique(psych_df[:stimid]))
    for prevrew in [true, false], sid in stimids
        psych_df = by(df[(df[:prevrew] .== prevrew) .& (df[:previd] .== sid), :], :stimid,
                      df -> DataFrame(n = size(df, 1), pr = _0mean(df[:resp])))
        x, n, p = psych_df[:stimid], psych_df[:n] ./ datasets, psych_df[:pr]
        # optimize over [μ, pl, pb], while using σ = globalθ[2]
        function psychllhgradσ!(subθ, subg)
            θ = [subθ[1], globalθ[2], subθ[2], subθ[3]]
            if !isempty(subg)
                subg[:] = -psychgrad(θ, x, n, p)[[1, 3, 4]] # ignore σ gradient
            end
        end
        res = optimize(
            OnceDifferentiable(
                subθ -> -psychllh([subθ[1], globalθ[2], subθ[2], subθ[3]], x, n, p),
                (subg, subθ) -> psychllhgradσ!(subθ, subg), θini[[1, 3, 4]]),
            θmin[[1, 3, 4]], θmax[[1, 3, 4]], θini[[1, 3, 4]], Fminbox(LBFGS()))
        θ = Optim.minimizer(res)
        θ = Float64[θ[1], globalθ[2], θ[2], θ[3]]
        # approximation, as off-diagonal terms might influence inv. Hessian
        θsd = sqrt.(max.(0.0, -1 ./ psychhessdiag(θ, x, n, p)))
        # ensure σ^2 < μ (1 - μ)
        θsd[3] = min(θsd[3], 0.999999 * sqrt.(θ[3] * (1 - θ[3])))
        θsd[4] = min(θsd[4], 0.999999 * sqrt.(θ[4] * (1 - θ[4])))        
        push!(fit_df, [sid, prevrew, 
                       θ[1], θ[2], θ[3], θ[4], θsd[1], θsd[2], θsd[3], θsd[4]])
    end
    fit_df[:prevmag] = [id == 0 ? 0 : (id < 5 ? 9 - id : id) for id in fit_df[:previd]]

    # compute biases
    bootstrapn = 1000
    # draw n samples from Beta distribution with mean μ, standard deviation σ
    samplebeta(μ, σ, n) = σ <= 1e-30 ? fill(μ, n) :
        (x = (μ - abs2(μ)) / abs2(σ) - 1; rand(Beta(μ * x, (1 - μ) * x), n))
    biasid_df = DataFrame(previd = Int[], prevrew = Bool[], 
                          biasμ = Float64[], biasσ = Float64[])
    for prevrew in [true, false], sid in stimids
        μ, σ, pl, pb, μsd, σsd, plsd, pbsd = Vector{Float64}(
            fit_df[(fit_df[:previd] .== sid) .& (fit_df[:prevrew] .== prevrew), 
                   [:μ, :σ, :pl, :pb, :μsd, :σsd, :plsd, :pbsd]][1,:])
        # bias = intersection of psychometric curve with 4.5
        biases = Float64[psychpy1([μ + μsd * randn(), 
                                   σ + σsd * randn(),
                                   samplebeta(pl, plsd, 1)[1],
                                   samplebeta(pb, pbsd, 1)[1]], 4.5) - 0.5
                         for n in 1:bootstrapn]
        push!(biasid_df, [sid, prevrew, mean(biases), √(var(biases))])
    end
    # biases over magnitudes -> flip for id < 5, and combine by variance
    stimmags = sort(unique(fit_df[fit_df[:prevmag] .!= 0, :prevmag]))
    biasmag_df = DataFrame(prevmag = Int[], prevrew = Bool[], 
                           biasμ = Float64[], biasσ = Float64[])
    for prevrew in [true, false], smag in stimmags
        μs = [-biasid_df[(biasid_df[:prevrew] .== prevrew) .& 
                         (biasid_df[:previd] .== 9 - smag), :biasμ][1],
              biasid_df[(biasid_df[:prevrew] .== prevrew) .& 
                         (biasid_df[:previd] .== smag), :biasμ][1]]
        σs = [biasid_df[(biasid_df[:prevrew] .== prevrew) .& 
                        (biasid_df[:previd] .== 9 - smag), :biasσ][1],
              biasid_df[(biasid_df[:prevrew] .== prevrew) .& 
                         (biasid_df[:previd] .== smag), :biasσ][1]]
        ws = 1 ./ abs2.(σs)
        push!(biasmag_df, [smag, prevrew, sum(μs .* ws) / sum(ws), √(1 / sum(ws))])
    end

    return seq_df, fit_df, biasid_df, biasmag_df
end

###############################################################################
# functions plotting behavioral data
#
# These functions use the above analysis functions to perform the plotting.

# returns chron/psych plot over stimulus id
function plotchronpsychid(d_bystim, stimx, xlabel, ttitle, t=Theme())
    d_bystim[:stimx] = stimx[d_bystim[:stimid]]
    xmin, xmax = minimum(stimx), maximum(stimx)
    tp = copy(t)
    tp.default_color = colorant"black"
    # chronometric curve
    d_bystim[:tmin] = d_bystim[:tμ] - d_bystim[:tsem]
    d_bystim[:tmax] = d_bystim[:tμ] + d_bystim[:tsem]
    pchronid = plot(d_bystim, x="stimx", y="tμ", ymin="tmin", ymax="tmax",
        Geom.point, Geom.errorbar,
        Guide.xlabel(xlabel, orientation=:horizontal),
        Guide.ylabel("reaction time [s]", orientation=:vertical),
        Guide.title("Chronometric curve, $ttitle"),
        Scale.x_continuous(minvalue=xmin, maxvalue=xmax),
        Scale.y_continuous(minvalue=0.26, maxvalue=0.44),
        tp)
    # psychometric curve
    d_bystim[:prmin] = d_bystim[:prμ] - d_bystim[:prsem]
    d_bystim[:prmax] = d_bystim[:prμ] + d_bystim[:prsem]
    ppsychid = plot(d_bystim, x="stimx", y="prμ", ymin="prmin", ymax="prmax",
        Geom.point, Geom.errorbar,
        Guide.xlabel(xlabel, orientation=:horizontal),
        Guide.ylabel("left choices (fraction)", orientation=:vertical),
        Guide.title("Psychometric curve, $ttitle"),
        Scale.x_continuous(minvalue=xmin, maxvalue=xmax),
        Scale.y_continuous(minvalue=0.0, maxvalue=1.0),
        tp)
    return pchronid, ppsychid
end
function plotchronpsychidm(d_bystim, m_bystim, stimx, xlabel, ttitle, t=Theme())
    d_bystim[:stimx] = stimx[d_bystim[:stimid]]
    m_bystim[:stimx] = stimx[m_bystim[:stimid]]
    xmin, xmax = minimum(stimx), maximum(stimx)
    tp = copy(t)
    tp.default_color = colorant"black"
    # chronometric curve
    d_bystim[:tmin] = d_bystim[:tμ] - d_bystim[:tsem]
    d_bystim[:tmax] = d_bystim[:tμ] + d_bystim[:tsem]
    pchronid = plot(
        layer(d_bystim, x="stimx", y="tμ", ymin="tmin", ymax="tmax",
            Geom.point, Geom.errorbar, order=1),
        layer(m_bystim, x="stimx", y="tμ", Geom.line, order=0),
        Guide.xlabel(xlabel, orientation=:horizontal),
        Guide.ylabel("reaction time [s]", orientation=:vertical),
        Guide.title("Chronometric curve, $ttitle"),
        Scale.x_continuous(minvalue=xmin, maxvalue=xmax),
        Scale.y_continuous(minvalue=0.26, maxvalue=0.44),
        tp)
    # psychometric curve
    d_bystim[:prmin] = d_bystim[:prμ] - d_bystim[:prsem]
    d_bystim[:prmax] = d_bystim[:prμ] + d_bystim[:prsem]
    ppsychid = plot(
        layer(d_bystim, x="stimx", y="prμ", ymin="prmin", ymax="prmax",
            Geom.point, Geom.errorbar, order=1),
        layer(m_bystim, x="stimx", y="prμ", Geom.line, order=0),
        Guide.xlabel(xlabel, orientation=:horizontal),
        Guide.ylabel("left choices (fraction)", orientation=:vertical),
        Guide.title("Psychometric curve, $ttitle"),
        Scale.x_continuous(minvalue=xmin, maxvalue=xmax),
        Scale.y_continuous(minvalue=0.0, maxvalue=1.0),
        tp)
    return pchronid, ppsychid
end

# returns psych/chron over stimulus magnitude
function plotchronpsychmag(d_bymag, magctrst, xlabel, ttitle, t=Theme())
    d_bymag[:stimctrst] = magctrst[d_bymag[:stimmag]]
    xmin, xmax = minimum(magctrst), maximum(magctrst)
    tp = copy(t)
    tp.default_color = colorant"black"
    # chronometric curve
    d_bymag[:tmin] = d_bymag[:tμ] - d_bymag[:tsem]
    d_bymag[:tmax] = d_bymag[:tμ] + d_bymag[:tsem]
    pchronmag = plot(d_bymag, x="stimctrst", y="tμ", ymin="tmin", ymax="tmax",
        Geom.point, Geom.errorbar,
        Guide.xlabel(xlabel, orientation=:horizontal),
        Guide.ylabel("reaction time [s]", orientation=:vertical),
        Guide.title("Chronometric curve, $ttitle"),
        Scale.x_continuous(minvalue=xmin, maxvalue=xmax),
        Scale.y_continuous(minvalue=0.26, maxvalue=0.44),
        tp)
    # psychometric curve
    d_bymag[:pcorrmin] = d_bymag[:pcorrμ] - d_bymag[:pcorrsem]
    d_bymag[:pcorrmax] = d_bymag[:pcorrμ] + d_bymag[:pcorrsem]
    ppsychmag = plot(d_bymag, x="stimctrst", y="pcorrμ", ymin="pcorrmin", ymax="pcorrmax",
        Geom.point, Geom.errorbar,
        Guide.xlabel(xlabel, orientation=:horizontal),
        Guide.ylabel("correct choices (fraction)", orientation=:vertical),
        Guide.title("Psychometric curve, $ttitle"),
        Scale.x_continuous(minvalue=xmin, maxvalue=xmax),
        Scale.y_continuous(minvalue=0.5, maxvalue=1.0),
        tp)
    return pchronmag, ppsychmag
end
function plotchronpsychmagm(d_bymag, m_bymag, magctrst, xlabel, ttitle, t=Theme())
    d_bymag[:stimctrst] = magctrst[d_bymag[:stimmag]]
    m_bymag[:stimctrst] = magctrst[m_bymag[:stimmag]]
    xmin, xmax = minimum(magctrst), maximum(magctrst)
    tp = copy(t)
    tp.default_color = colorant"black"
    # chronometric curve
    d_bymag[:tmin] = d_bymag[:tμ] - d_bymag[:tsem]
    d_bymag[:tmax] = d_bymag[:tμ] + d_bymag[:tsem]
    pchronmag = plot(
        layer(d_bymag, x="stimctrst", y="tμ", ymin="tmin", ymax="tmax",
            Geom.point, Geom.errorbar, order=1),
        layer(m_bymag, x="stimctrst", y="tμ", Geom.line, order=0),
        Guide.xlabel(xlabel, orientation=:horizontal),
        Guide.ylabel("reaction time [s]", orientation=:vertical),
        Guide.title("Chronometric curve, $ttitle"),
        Scale.x_continuous(minvalue=xmin, maxvalue=xmax),
        Scale.y_continuous(minvalue=0.26, maxvalue=0.44),
        tp)
    # psychometric curve
    d_bymag[:pcorrmin] = d_bymag[:pcorrμ] - d_bymag[:pcorrsem]
    d_bymag[:pcorrmax] = d_bymag[:pcorrμ] + d_bymag[:pcorrsem]
    ppsychmag = plot(
        layer(d_bymag, x="stimctrst", y="pcorrμ", ymin="pcorrmin", ymax="pcorrmax",
            Geom.point, Geom.errorbar, order=1),
        layer(m_bymag, x="stimctrst", y="pcorrμ", Geom.line, order=0),
        Guide.xlabel(xlabel, orientation=:horizontal),
        Guide.ylabel("correct choices (fraction)", orientation=:vertical),
        Guide.title("Psychometric curve, $ttitle"),
        Scale.x_continuous(minvalue=xmin, maxvalue=xmax),
        Scale.y_continuous(minvalue=0.5, maxvalue=1.0),
        tp)
    return pchronmag, ppsychmag
end

# returns array of conditional psychometric curves/fits
function plotcondpsych(d_seq, d_seqfit, stimx, xlabel, ttitle, t=Theme())
    # ignore the actual x stimulus distribution, plot on [0..1] scale
    stimx = range(0, stop=1, length=8)
    d_seq[:stimx] = stimx[d_seq[:stimid]]
    stimidcol(stimid) = RGB(0.0, 1 - 0.125stimid, 0.125stimid)
    pcondpsych = Vector{Any}(undef, 4)
    xcont = range(0.0, stop=1.0, length=100)
    globalθ = Vector{Float64}(
        d_seqfit[(d_seqfit[:prevrew] .== true) .& (d_seqfit[:previd] .== 0), 
                 [:μ, :σ, :pl, :pb]][1,:])
    for stimmag = 1:4
        id1, id2 = stimmag, 9-stimmag
        pr1 = [d_seq[(d_seq[:prevrew] .== true) .& (d_seq[:previd] .== id1) .&
                     (d_seq[:stimid] .== id), :prμ][1] for id = 1:8]
        pr1sem = [d_seq[(d_seq[:prevrew] .== true) .& (d_seq[:previd] .== id1) .&
                        (d_seq[:stimid] .== id), :prsem][1] for id = 1:8]
        θ1 = Vector{Float64}(
            d_seqfit[(d_seqfit[:prevrew] .== true) .& (d_seqfit[:previd] .== id1), 
                     [:μ, :σ, :pl, :pb]][1,:])
        pr2 = [d_seq[(d_seq[:prevrew] .== true) .& (d_seq[:previd] .== id2) .&
                     (d_seq[:stimid] .== id), :prμ][1] for id = 1:8]
        pr2sem = [d_seq[(d_seq[:prevrew] .== true) .& (d_seq[:previd] .== id2) .&
                        (d_seq[:stimid] .== id), :prsem][1] for id = 1:8]
        θ2 = Vector{Float64}(
            d_seqfit[(d_seqfit[:prevrew] .== true) .& (d_seqfit[:previd] .== id2), 
                     [:μ, :σ, :pl, :pb]][1,:])

        pcondpsych[stimmag] = plot(
            Scale.x_continuous(minvalue=0.0, maxvalue=1.0),
            Scale.y_continuous(minvalue=0.0, maxvalue=1.0),
            layer(x = xcont, y = psychpy1(globalθ, 1 .+ 7xcont),
                  Geom.line, Theme(default_color=colorant"black"), order=0),
            layer(x = xcont, y = psychpy1(θ1, 1 .+ 7xcont),
                  Geom.line, Theme(default_color=stimidcol(id1)), order=1), 
            layer(x = xcont, y = psychpy1(θ2, 1 .+ 7xcont),
                  Geom.line, Theme(default_color=stimidcol(id2)), order=2),     
            layer(x = stimx, y = pr1, ymin = max.(0.0, pr1-pr1sem), ymax = min.(1.0, pr1+pr1sem), 
                  Geom.point, Geom.errorbar, Theme(default_color=stimidcol(id1)), order=3),
            layer(x = stimx, y = pr2, ymin = max.(0.0, pr2-pr2sem), ymax = min.(1.0, pr2+pr2sem), 
                  Geom.point, Geom.errorbar, Theme(default_color=stimidcol(id2)), order=4),
            Guide.xlabel(xlabel, orientation=:horizontal),
            Guide.ylabel("left choices (fraction)", orientation=:vertical),
            t)
    end

    return pcondpsych
end
# returns array of conditional psychometric curves/fits
function plotcondpsychm(d_seq, m_seq, m_seqfit, stimx, xlabel, ttitle, t=Theme())
    # ignore the actual x stimulus distribution, plot on [0..1] scale
    stimx = range(0, stop=1, length=8)
    d_seq[:stimx] = stimx[d_seq[:stimid]]
    stimidcol(stimid) = RGB(0.0, 1 - 0.125stimid, 0.125stimid)
    fitidcol(stimid) = RGB(0.5, 1 - 0.0625stimid, 0.5 + 0.0625stimid)
    pcondpsych = Vector{Any}(undef, 4)
    xcont = range(0.0, stop=1.0, length=100)
    for stimmag = 1:4
        id1, id2 = stimmag, 9-stimmag
        pr1 = [d_seq[(d_seq[:prevrew] .== true) .& (d_seq[:previd] .== id1) .&
                     (d_seq[:stimid] .== id), :prμ][1] for id = 1:8]
        pr1sem = [d_seq[(d_seq[:prevrew] .== true) .& (d_seq[:previd] .== id1) .&
                        (d_seq[:stimid] .== id), :prsem][1] for id = 1:8]
        mpr1 = [m_seq[(m_seq[:prevrew] .== true) .& (m_seq[:previd] .== id1) .&
                      (m_seq[:stimid] .== id), :prμ][1] for id = 1:8]
        θ1 = Vector{Float64}(
            m_seqfit[(m_seqfit[:prevrew] .== true) .& (m_seqfit[:previd] .== id1), 
                     [:μ, :σ, :pl, :pb]][1,:])
        pr2 = [d_seq[(d_seq[:prevrew] .== true) .& (d_seq[:previd] .== id2) .&
                     (d_seq[:stimid] .== id), :prμ][1] for id = 1:8]
        pr2sem = [d_seq[(d_seq[:prevrew] .== true) .& (d_seq[:previd] .== id2) .&
                        (d_seq[:stimid] .== id), :prsem][1] for id = 1:8]
        mpr2 = [m_seq[(m_seq[:prevrew] .== true) .& (m_seq[:previd] .== id2) .&
                      (m_seq[:stimid] .== id), :prμ][1] for id = 1:8]
        θ2 = Vector{Float64}(
            m_seqfit[(m_seqfit[:prevrew] .== true) .& (m_seqfit[:previd] .== id2), 
                     [:μ, :σ, :pl, :pb]][1,:])

        pcondpsych[stimmag] = plot(
            Scale.x_continuous(minvalue=0.0, maxvalue=1.0),
            Scale.y_continuous(minvalue=0.0, maxvalue=1.0),
            layer(x = xcont, y = psychpy1(θ1, 1 .+ 7xcont),
                  Geom.line, Theme(default_color=fitidcol(id1)), order=0), 
            layer(x = xcont, y = psychpy1(θ2, 1 .+ 7xcont),
                  Geom.line, Theme(default_color=fitidcol(id2)), order=1),     
            layer(x = stimx, y = mpr1, Geom.line,
                  Theme(default_color=stimidcol(id1)), order=2),
            layer(x = stimx, y = mpr2, Geom.line,
                  Theme(default_color=stimidcol(id2)), order=3),
            layer(x = stimx, y = pr1, ymin = max.(0.0, pr1-pr1sem), ymax = min.(1.0, pr1+pr1sem), 
                  Geom.point, Geom.errorbar, Theme(default_color=stimidcol(id1)), order=4),
            layer(x = stimx, y = pr2, ymin = max.(0.0, pr2-pr2sem), ymax = min.(1.0, pr2+pr2sem), 
                  Geom.point, Geom.errorbar, Theme(default_color=stimidcol(id2)), order=5),
            Guide.xlabel(xlabel, orientation=:horizontal),
            Guide.ylabel("left choices (fraction)", orientation=:vertical),
            t)
    end

    return pcondpsych
end

# returns bias plot over simulus id
function plotbiasid(d_biasid, stimx, xlabel, ttitle, t=Theme())
    d_biasid[:prevx] = stimx[d_biasid[:previd]]
    d_biasid[:biasmin] = d_biasid[:biasμ] - d_biasid[:biasσ]
    d_biasid[:biasmax] = d_biasid[:biasμ] + d_biasid[:biasσ]
    tp = copy(t)
    tp.default_color = colorant"red"
    pbiasid = plot(d_biasid[d_biasid[:prevrew],:], 
        x="prevx", y="biasμ", ymin="biasmin", ymax="biasmax",
        Geom.point, Geom.errorbar,
        Guide.xlabel(xlabel, orientation=:horizontal),
        Guide.ylabel("choice bias", orientation=:vertical),
        Guide.title("Choice biases, $ttitle"), 
        Scale.x_continuous(minvalue=minimum(stimx), maxvalue=maximum(stimx)),
        Scale.y_continuous(minvalue=-0.5, maxvalue=0.5),
        tp)
    return pbiasid
end
function plotbiasidm(d_biasid, m_biasid, stimx, xlabel, ttitle, t=Theme())
    d_biasid[:prevx] = stimx[d_biasid[:previd]]
    m_biasid[:prevx] = stimx[m_biasid[:previd]]
    d_biasid[:biasmin] = d_biasid[:biasμ] - d_biasid[:biasσ]
    d_biasid[:biasmax] = d_biasid[:biasμ] + d_biasid[:biasσ]
    tp = copy(t)
    tp.default_color = colorant"red"
    pbiasid = plot(
        layer(d_biasid[d_biasid[:prevrew],:], 
            x="prevx", y="biasμ", ymin="biasmin", ymax="biasmax",
            Geom.point, Geom.errorbar),
        layer(m_biasid[m_biasid[:prevrew],:],
            x="prevx", y="biasμ", Geom.line),
        Guide.xlabel(xlabel, orientation=:horizontal),
        Guide.ylabel("choice bias", orientation=:vertical),
        Guide.title("Choice biases, $ttitle"), 
        Scale.x_continuous(minvalue=minimum(stimx), maxvalue=maximum(stimx)),
        Scale.y_continuous(minvalue=-0.5, maxvalue=0.5),
        tp)
    return pbiasid
end

# returns bias plot over stimulus magnitude
function plotbiasmag(d_biasmag, magctrst, xlabel, ttitle, t=Theme())
    d_biasmag[:prevctrst] = magctrst[d_biasmag[:prevmag]]
    d_biasmag[:biasmin] = d_biasmag[:biasμ] - d_biasmag[:biasσ]
    d_biasmag[:biasmax] = d_biasmag[:biasμ] + d_biasmag[:biasσ]
    tp = copy(t)
    tp.default_color = colorant"red"
    pbiasmag = plot(d_biasmag[d_biasmag[:prevrew],:],
        x="prevctrst", y="biasμ", ymin="biasmin", ymax="biasmax",
        Geom.point, Geom.errorbar,
        Guide.xlabel(xlabel, orientation=:horizontal),
        Guide.ylabel("choice bias", orientation=:vertical),
        Guide.title("Choice biases, $ttitle"),
        Scale.x_continuous(minvalue=minimum(magctrst), maxvalue=maximum(magctrst)),
        Scale.y_continuous(minvalue=-0.05, maxvalue=0.25),
        tp)
    return pbiasmag
end
function plotbiasmagm(d_biasmag, m_biasmag, magctrst, xlabel, ttitle, t=Theme())
    d_biasmag[:prevctrst] = magctrst[d_biasmag[:prevmag]]
    m_biasmag[:prevctrst] = magctrst[m_biasmag[:prevmag]]
    d_biasmag[:biasmin] = d_biasmag[:biasμ] - d_biasmag[:biasσ]
    d_biasmag[:biasmax] = d_biasmag[:biasμ] + d_biasmag[:biasσ]
    tp = copy(t)
    tp.default_color = colorant"red"
    pbiasmag = plot(
        layer(d_biasmag[d_biasmag[:prevrew],:],
            x="prevctrst", y="biasμ", ymin="biasmin", ymax="biasmax",
            Geom.point, Geom.errorbar),
        layer(m_biasmag[m_biasmag[:prevrew],:],
            x="prevctrst", y="biasμ", Geom.line),
        Guide.xlabel(xlabel, orientation=:horizontal),
        Guide.ylabel("choice bias", orientation=:vertical),
        Guide.title("Choice biases, $ttitle"),
        Scale.x_continuous(minvalue=minimum(magctrst), maxvalue=maximum(magctrst)),
        Scale.y_continuous(minvalue=-0.05, maxvalue=0.25),
        tp)
    return pbiasmag
end