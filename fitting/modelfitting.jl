# functions to fit models to data

using NLopt, DataFrames, Random, Printf

const tnd_sd_scale = 0.1 # SD(tnd) = tnd_sd_scale * <tnd> (as in Palmer, Huk & Shadlen, 2005)
#const tnd_sd_scale = 0 # SD(tnd) = tnd_sd_scale * <tnd>
const simsrand = 0


###############################################################################
# objective functions for fitting model parameters

# Bernoulli likelihood
plogq(p, q) = 0 .<= p .< 1e-300 ? -500.0p : p .* log.(max.(pmin, q))  # ensure 0 * log 0 = 0
#bernllh(m_pr, d_pr, d_n) = d_n .* (plogq(d_pr, m_pr) + plogq(1 - d_pr, 1 - m_pr))
bernllh(m_pr, d_pr, d_n) = d_n .* (d_pr .* log.(max.(pmin, m_pr)) .+ (1 .- d_pr) .* log.(max.(pmin, 1-m_pr)))
# Gaussian likelihood for data drawn from N(m_tμ, m_tσ^2)
gaussfullllh(m_tμ, m_tσ, d_tμ, d_tσ, d_n) = d_n .* (-0.5logtwoπ .- log.(m_tσ) .- 
    (abs2.(d_tσ) .+ abs2.(d_tμ .- m_tμ)) ./ (2abs2.(m_tσ)))
# Gaussian likelihood for data mean drawn d_n times from N(m_tμ, d_tsem^2)
gausssemllh(m_tμ, d_tμ, d_tsem, d_n) = d_n .* (-0.5logtwoπ .- log.(d_tsem) .- 
    abs2.(d_tμ .- m_tμ)./(2abs2.(d_tsem)))


# the objective function
abstract type FittingObjective end

###############################################################################
# objective functions that don't model sequential choices

# - fit choices by Bernoulli likelihood
# - fit RTs with Gaussian likelihood for correct / incorrect choices combined
# - only fit a subset or both conditions (identified at initialization)
# - the RT variance is augmented by (tnd_sd_scale * tnd)^2
struct PsychChron1ObjFn <: FittingObjective
    m::ModelBase
    evidence::Matrix{Float64}
    ids::Vector{Int}
    uniqueids::Vector{Int}
    burnin::Int
    d_psych::Vector{DataFrame}
    conditions::Vector{Int}

    PsychChron1ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets,
                     conds::AbstractVector{Symbol}) =
        new(m, evidence, ids, sort(unique(ids)), burnin, 
            DataFrame[by(df, :stimid, df -> _tcstats(df, datasets)) 
                      for df in [d_ident, d_categ]],
            map(x -> Dict(:ident => 1, :categ => 2)[x], conds))
    # defaults to fitting both conditions simultaneously
    PsychChron1ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets) =
        PsychChron1ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets,
                         [:ident, :categ])
end
# function returns avg. llh across modeled trials
function llh(o::PsychChron1ObjFn, ϕ::Vector)
    # simulate model and gather summary stats
    Random.seed!(simsrand)
    ms = simmodel(o.m, ϕ, o.evidence, o.ids)
    tndvar = abs2(tnd_sd_scale * gettnd(o.m, ϕ))
    # assemble likelihood
    llh, n = 0.0, 0.0
    for i in o.conditions
        d = o.d_psych[i]
        m = by(ms[i][(o.burnin+1):end, :], :stimid, _tcstats)
        for id in o.uniqueids
            m_prμ, m_tμ = Vector{Float64}(m[m[:stimid] .== id, [:prμ, :tμ]][1,:])
            d_n, d_prμ, d_tμ, d_tsem =
                Vector{Float64}(d[d[:stimid] .== id, [:n, :prμ, :tμ, :tsem]][1,:])
            llh += bernllh(m_prμ, d_prμ, d_n) + 
                # we use d_n = 1, as we are already using the SEM
                gausssemllh(m_tμ, d_tμ, √(abs2(d_tsem) + tndvar), 1)
            n += d_n
        end
    end
    return llh / n
end
# alternatives: fit single condition
PsychChron1IdentObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets) =
    PsychChron1ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets, [:ident])
PsychChron1CategObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets) =
    PsychChron1ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets, [:categ])


# - fit choices by Bernoulli likelihood
# - fit RTs for Gaussian likelihood for correct / incorrect choices separately
# - the RT variance is augmented by (tnd_sd_scale * tnd)^2
struct PsychChron2ObjFn <: FittingObjective
    m::ModelBase
    evidence::Matrix{Float64}
    ids::Vector{Int}
    uniqueids::Vector{Int}
    burnin::Int
    d_psych::Vector{DataFrame}

    PsychChron2ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets) =
        new(m, evidence, ids, sort(unique(ids)), burnin, 
            DataFrame[by(df, :stimid, df -> _tcstats(df, datasets)) 
                      for df in [d_ident, d_categ]])
end
# function returns avg. llh across modeled trials
function llh(o::PsychChron2ObjFn, ϕ::Vector)
    # simulate model
    Random.seed!(simsrand)
    ms = simmodel(o.m, ϕ, o.evidence, o.ids)
    tndvar = abs2(tnd_sd_scale * gettnd(o.m, ϕ))
    # assemble likelihood
    llh, n = 0.0, 0.0
    for i in [1, 2]
        d = o.d_psych[i]
        m = by(ms[i][(o.burnin+1):end, :], :stimid, _tcstats)
        for id in o.uniqueids
            m_prμ, m_tcorrμ, m_tincorrμ = Vector{Float64}( 
                m[m[:stimid] .== id, [:prμ, :tcorrμ, :tincorrμ]][1,:])
            d_n, d_ncorr, d_prμ, d_tcorrμ, d_tincorrμ, d_tcorrsem, d_tincorrsem =
                Vector{Float64}( 
                    d[d[:stimid] .== id, 
                      [:n, :ncorr, :prμ, :tcorrμ, :tincorrμ, :tcorrsem, :tincorrsem]][1,:])
            llh += bernllh(m_prμ, d_prμ, d_n) + 
                # we use d_n = 1, as we are already using the SEM
                gausssemllh(m_tcorrμ, d_tcorrμ, √(abs2(d_tcorrsem) + tndvar), 1) + 
                gausssemllh(m_tincorrμ, d_tincorrμ, √(abs2(d_tincorrsem) + tndvar), 1)
            n += d_n
        end
    end
    return llh / n
end


# - fit choices by Bernoulli likelihood
# - fit RTs with Gaussian SD likelihood for correct / incorrect choices combined
# - only fit a subset or both conditions (identified at initialization)
# (same as PsychChron2ObjFn, only that RTs are modelled by SD, not SEM)
struct PsychChron3ObjFn <: FittingObjective
    m::ModelBase
    evidence::Matrix{Float64}
    ids::Vector{Int}
    uniqueids::Vector{Int}
    burnin::Int
    d_psych::Vector{DataFrame}
    conditions::Vector{Int}

    PsychChron3ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets,
                     conds::AbstractVector{Symbol}) =
        new(m, evidence, ids, sort(unique(ids)), burnin, 
            DataFrame[by(df, :stimid, df -> _tcstats(df, datasets)) 
                      for df in [d_ident, d_categ]],
            map(x -> Dict(:ident => 1, :categ => 2)[x], conds))
    # defaults to fitting both conditions simultaneously
    PsychChron3ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets) =
        PsychChron3ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets,
                         [:ident, :categ])
end
# function returns avg. llh across modeled trials
function llh(o::PsychChron3ObjFn, ϕ::Vector)
    # simulate model and gather summary stats
    Random.seed!(simsrand)
    ms = simmodel(o.m, ϕ, o.evidence, o.ids)
    # assemble likelihood
    llh, n = 0.0, 0.0
    for i in o.conditions
        d = o.d_psych[i]
        m = by(ms[i][(o.burnin+1):end, :], :stimid, _tcstats)
        for id in o.uniqueids
            m_prμ, m_tμ, m_tσ =
                Vector{Float64}(m[m[:stimid] .== id, [:prμ, :tμ, :tσ]][1,:])
            d_n, d_prμ, d_tμ, d_tσ =
                Vector{Float64}(d[d[:stimid] .== id, [:n, :prμ, :tμ, :tσ]][1,:])
            llh += bernllh(m_prμ, d_prμ, d_n) + 
                gaussfullllh(m_tμ, m_tσ, d_tμ, d_tσ, d_n)
            n += d_n
        end
    end
    return llh / n
end
# alternatives: fit single condition
PsychChron3IdentObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets) =
    PsychChron3ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets, [:ident])
PsychChron3CategObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets) =
    PsychChron3ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets, [:categ])


###############################################################################
# objective functions that model sequential choices

# - after correct choices, fit choices by Bernoulli likelihood, condition on
#   previous stimulus id
# - after incorrect choices, fit choices by Bernoulli likelihood, independent
#   of pervious stimulus
# - fit RTs with Gaussian likelihood for correct / incorrect choices combined
# - the RT variance is augmented by (tnd_sd_scale * tnd)^2
struct PsychChronSeq1ObjFn <: FittingObjective
    m::ModelBase
    evidence::Matrix{Float64}
    ids::Vector{Int}
    uniqueids::Vector{Int}
    burnin::Int
    d_nonseq::Vector{DataFrame}
    d_seq_corr::Vector{DataFrame}
    d_seq_incorr::Vector{DataFrame}

    PsychChronSeq1ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets) = 
        new(m, evidence, ids, sort(unique(ids)), burnin,
            DataFrame[by(df, :stimid, df -> _tcstats(df, datasets))
                      for df in [d_ident, d_categ]],
            DataFrame[by(df[df[:prevrew] .== true, :], [:stimid, :previd], 
                         df -> _tcstats(df, datasets)) for df in [d_ident, d_categ]],
            DataFrame[by(df[df[:prevrew] .== false, :], :stimid, 
                         df -> _tcstats(df, datasets)) for df in [d_ident, d_categ]])
end
function llh(o::PsychChronSeq1ObjFn, ϕ::Vector)
    Random.seed!(simsrand)
    ms = simmodel(o.m, ϕ, o.evidence, o.ids)
    tndvar = abs2(tnd_sd_scale * gettnd(o.m, ϕ))
    # assemble likelihood
    llh, n = 0.0, 0.0
    for i in [1, 2]
        mi = ms[i][(o.burnin+1):end, :]
        m = by(mi, :stimid, _tcstats)
        mcorr = by(mi[mi[:prevrew] .== true, :], [:stimid, :previd], _tcstats)
        if any(mi[:prevrew] .== false)
            mincorr = by(mi[mi[:prevrew] .== false, :], :stimid, _tcstats)
        else
            # create dataframe that violates any(mincorr[:stimid] .== id) below
            mincorr = DataFrame(stimid = Int[0])
        end
        d = o.d_nonseq[i]
        dcorr = o.d_seq_corr[i]
        dincorr = o.d_seq_incorr[i]
        # in all cases, iterate over current stimulus id
        for id in o.uniqueids
            # reaction times
            m_tμ = m[m[:stimid] .== id, :tμ][1]
            d_n, d_tμ, d_tsem = 
                Vector{Float64}(d[d[:stimid] .== id, [:n, :tμ, :tsem]][1,:])
            # we use d_n = 1, as we are already using the SEM
            llh += gausssemllh(m_tμ, d_tμ, √(abs2(d_tsem) + tndvar), 1)
            # choices after incorrect trials
            if any(mincorr[:stimid] .== id)
                m_prμ = mincorr[mincorr[:stimid] .== id, :prμ][1]
            else
                m_prμ = id > 4 ? 1.0 : 0.0    # can happen for too high 'gain'
            end
            d_nincorr, d_prμ =
                Vector{Float64}(dincorr[dincorr[:stimid] .== id, [:n, :prμ]][1,:])
            llh += bernllh(m_prμ, d_prμ, d_nincorr)
            # choices after correct trials
            d_ncorr = 0.0
            for previd in o.uniqueids
                m_prμ = mcorr[(mcorr[:previd] .== previd) .& 
                              (mcorr[:stimid] .== id), :prμ][1]
                d_ni, d_prμ = Vector{Float64}(
                    dcorr[(dcorr[:previd] .== previd) .& 
                          (dcorr[:stimid] .== id), [:n, :prμ]][1,:])
                llh += bernllh(m_prμ, d_prμ, d_ni)
                d_ncorr += d_ni
            end
            # check if n is consistent
            @assert d_n == d_ncorr + d_nincorr
            n += d_n
        end
    end
    return llh / n
end

# - after correct choices, fit choices by Bernoulli likelihood and RTs with
#   Gaussian likelhood (indep. of corr/incorr), conditioned on previous stimid
# - after incorrect choices, do the same without conditioning on prev. stimid
# - the RT variance is augmented by (tnd_sd_scale * tnd)^2
#
# This model provides larger llh's than the above models, due to the scaling of
# the SEM with n. For this model, the SEM's will be unproportionally larger
# (based on less data) than for the above models, causing the llh's to be larger.
# Overall, this causes a stronger focus on p(choice) than RTs.
struct PsychChronSeq2ObjFn <: FittingObjective
    m::ModelBase
    evidence::Matrix{Float64}
    ids::Vector{Int}
    uniqueids::Vector{Int}
    burnin::Int
    d_seq_corr::Vector{DataFrame}
    d_seq_incorr::Vector{DataFrame}

    PsychChronSeq2ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets) =
        new(m, evidence, ids, sort(unique(ids)), burnin, 
            DataFrame[by(df[df[:prevrew] .== true, :], [:stimid, :previd], 
                         df -> _tcstats(df, datasets)) for df in [d_ident, d_categ]],
            DataFrame[by(df[df[:prevrew] .== false, :], :stimid, 
                         df -> _tcstats(df, datasets)) for df in [d_ident, d_categ]])
end
function llh(o::PsychChronSeq2ObjFn, ϕ::Vector)
    Random.seed!(simsrand)
    ms = simmodel(o.m, ϕ, o.evidence, o.ids)
    tndvar = abs2(tnd_sd_scale * gettnd(o.m, ϕ))
    # assemble likelihood
    llh, n = 0.0, 0.0
    for i in [1, 2]
        mi = ms[i][(o.burnin+1):end, :]
        mcorr = by(mi[mi[:prevrew] .== true, :], [:stimid, :previd], _tcstats)
        if any(mi[:prevrew] .== false)
            mincorr = by(mi[mi[:prevrew] .== false, :], :stimid, _tcstats)
        else
            # create dataframe that violates any(mincorr[:stimid] .== id) below
            mincorr = DataFrame(stimid = Int[0], tμ = Float64[mean(mcorr[:tμ])])
        end
        # llh for trials after correct choices
        dcorr = o.d_seq_corr[i]
        for previd in o.uniqueids
            for id in o.uniqueids
                m_prμ, m_tμ = Vector{Float64}( 
                    mcorr[(mcorr[:previd] .== previd) .& 
                          (mcorr[:stimid] .== id), [:prμ, :tμ]][1,:])
                d_n, d_prμ, d_tμ, d_tsem = Vector{Float64}(
                    dcorr[(dcorr[:previd] .== previd) .& 
                          (dcorr[:stimid] .== id), [:n, :prμ, :tμ, :tsem]][1,:])
                llh += bernllh(m_prμ, d_prμ, d_n) +
                    # we use d_n = 1, as we are already using the SEM
                    gausssemllh(m_tμ, d_tμ, √(abs2(d_tsem) + tndvar), 1)
                n += d_n
            end
        end
        # llh for trials after incorrect choices
        dincorr = o.d_seq_incorr[i]
        for id in o.uniqueids
            if any(mincorr[:stimid] .== id)
                m_prμ, m_tμ = Vector{Float64}( 
                    mincorr[mincorr[:stimid] .== id, [:prμ, :tμ]][1,:])
            else
                # for large p(correct), there might have been too few previous
                # incorrect choices - in this case, take the average tμ, and the
                # correct pr
                m_prμ = id > 4 ? 1.0 : 0.0
                m_tμ = mean(mincorr[:tμ])
            end
            d_n, d_prμ, d_tμ, d_tsem = Vector{Float64}(
                dincorr[dincorr[:stimid] .== id, [:n, :prμ, :tμ, :tsem]][1,:])
            llh += bernllh(m_prμ, d_prμ, d_n) +
                # we use d_n = 1, as we are already using the SEM
                gausssemllh(m_tμ, d_tμ, √(abs2(d_tsem) + tndvar), 1)
            n += d_n
        end
    end
    return llh / n
end


# - after correct choices, fit choices by Bernoulli likelihood, condition on
#   previous stimulus id
# - after incorrect choices, fit choices by Bernoulli likelihood, independent
#   of pervious stimulus
# - fit RTs with Gaussian likelihood for correct / incorrect choices combined
# (same as PsychChronSeq1ObjFn, only that RTs are modelled by SD, not SEM)
struct PsychChronSeq3ObjFn <: FittingObjective
    m::ModelBase
    evidence::Matrix{Float64}
    ids::Vector{Int}
    uniqueids::Vector{Int}
    burnin::Int
    d_nonseq::Vector{DataFrame}
    d_seq_corr::Vector{DataFrame}
    d_seq_incorr::Vector{DataFrame}

    PsychChronSeq3ObjFn(m, evidence, ids, burnin, d_ident, d_categ, datasets) = 
        new(m, evidence, ids, sort(unique(ids)), burnin,
            DataFrame[by(df, :stimid, df -> _tcstats(df, datasets))
                      for df in [d_ident, d_categ]],
            DataFrame[by(df[df[:prevrew] .== true, :], [:stimid, :previd], 
                         df -> _tcstats(df, datasets)) for df in [d_ident, d_categ]],
            DataFrame[by(df[df[:prevrew] .== false, :], :stimid, 
                         df -> _tcstats(df, datasets)) for df in [d_ident, d_categ]])
end
function llh(o::PsychChronSeq3ObjFn, ϕ::Vector)
    Random.seed!(simsrand)
    ms = simmodel(o.m, ϕ, o.evidence, o.ids)
    # assemble likelihood
    llh, n = 0.0, 0.0
    for i in [1, 2]
        mi = ms[i][(o.burnin+1):end, :]
        m = by(mi, :stimid, _tcstats)
        mcorr = by(mi[mi[:prevrew] .== true, :], [:stimid, :previd], _tcstats)
        if any(mi[:prevrew] .== false)
            mincorr = by(mi[mi[:prevrew] .== false, :], :stimid, _tcstats)
        else
            # create dataframe that violates any(mincorr[:stimid] .== id) below
            mincorr = DataFrame(stimid = Int[0])
        end
        d = o.d_nonseq[i]
        dcorr = o.d_seq_corr[i]
        dincorr = o.d_seq_incorr[i]
        # in all cases, iterate over current stimulus id
        for id in o.uniqueids
            # reaction times
            m_tμ, m_tσ = Vector{Float64}(m[m[:stimid] .== id, [:tμ, :tσ]][1,:])
            d_n, d_tμ, d_tσ = Vector{Float64}(d[d[:stimid] .== id, [:n, :tμ, :tσ]][1,:])
            llh += gaussfullllh(m_tμ, m_tσ, d_tμ, d_tσ, d_n)
            # choices after incorrect trials
            if any(mincorr[:stimid] .== id)
                m_prμ = mincorr[mincorr[:stimid] .== id, :prμ][1]
            else
                m_prμ = id > 4 ? 1.0 : 0.0    # can happen for too high 'gain'
            end
            d_nincorr, d_prμ =
                Vector{Float64}(dincorr[dincorr[:stimid] .== id, [:n, :prμ]][1,:])
            llh += bernllh(m_prμ, d_prμ, d_nincorr)
            # choices after correct trials
            d_ncorr = 0.0
            for previd in o.uniqueids
                m_prμ = mcorr[(mcorr[:previd] .== previd) .& 
                              (mcorr[:stimid] .== id), :prμ][1]
                d_ni, d_prμ = Vector{Float64}(
                    dcorr[(dcorr[:previd] .== previd) .& 
                          (dcorr[:stimid] .== id), [:n, :prμ]][1,:])
                llh += bernllh(m_prμ, d_prμ, d_ni)
                d_ncorr += d_ni
            end
            # check if n is consistent
            @assert d_n == d_ncorr + d_nincorr
            n += d_n
        end
    end
    return llh / n
end

###############################################################################
# objective functions for the interleaved dataset

# - fit choices by Bernoulli likelihood
# - fit RTs with Gaussian likelihood for correct / incorrect choices combined
# - only fit a subset or both conditions (identified at initialization)
# - the RT variance is augmented by (tnd_sd_scale * tnd)^2
struct PsychChron1ItlvdObjFn <: FittingObjective
    m::ModelBase
    evidence::Matrix{Float64}
    ids::Vector{Int}
    burnin::Int
    d_psych::DataFrame
    modeledids::Vector{Int}

    PsychChron1ItlvdObjFn(m, evidence, ids, burnin, d_itlvd, datasets, modeledids) =
        new(m, evidence, ids, burnin, 
            by(d_itlvd, :stimid, df -> _tcstats(df, datasets)), modeledids)
    # defaults to fitting all stimulus ids simultaneously
    PsychChron1ItlvdObjFn(m, evidence, ids, burnin, d_itlvd, datasets) =
        PsychChron1ItlvdObjFn(m, evidence, ids, burnin, d_itlvd, datasets,
                               collect(1:32))
end
# function returns avg. llh across modeled trials
function llh(o::PsychChron1ItlvdObjFn, ϕ::Vector)
    # simulate model and gather summary stats
    Random.seed!(simsrand)
    ms = simitlvdmodel(o.m, ϕ, o.evidence, o.ids)
    tndvar = abs2(tnd_sd_scale * gettnd(o.m, ϕ))
    # assemble likelihood
    llh, n = 0.0, 0.0
    d = o.d_psych
    m = by(ms[(o.burnin+1):end, :], :stimid, _tcstats)
    for id in o.modeledids
        m_prμ, m_tμ = Vector{Float64}(m[m[:stimid] .== id, [:prμ, :tμ]][1,:])
        d_n, d_prμ, d_tμ, d_tsem =
            Vector{Float64}(d[d[:stimid] .== id, [:n, :prμ, :tμ, :tsem]][1,:])
        #println("id $id : mprμ = $m_prμ  dprμ = $d_prμ    mtμ = $m_tμ  dtμ = $d_tμ")
        llh += bernllh(m_prμ, d_prμ, d_n) + 
            # we use d_n = 1, as we are already using the SEM
            gausssemllh(m_tμ, d_tμ, √(abs2(d_tsem) + tndvar), 1)
        n += d_n
    end
    return llh / n
end


###############################################################################
# model fitting function

# returns all tested and best-fitting model parameters and objective value for given model
function fitmodel(m::ModelBase, o::FittingObjective, ϕini::AbstractVector; verbose=true)
    maxevals = 5000
    ϕmin, ϕmax = minimumϕ(m), maximumϕ(m)
    ϕn = length(ϕini)
    # store tested parameters + LLHs
    ϕs = Array{Float64}(undef, maxevals+1, ϕn+1)

    # formats output vector of floats
    fmtvec(x) = foldl((a, b) -> a * ", " * b, [@sprintf("%.2f", i) for i in x])

    # verbose likelihood function
    evalcount = -1  # start at -1 to not record first llhv(.) test call
    function llhv(ϕ, g)
        boundllh = llh(o, max.(ϕmin, min.(ϕmax, ϕ)))
        if verbose && mod(evalcount, 20) == 0
            @printf("%8d  ϕ = [%s]   llh = %f\n", evalcount, fmtvec(ϕ), boundllh)
        end
        evalcount += 1
        if evalcount >= 1
            ϕs[evalcount,:] = [ϕ; boundllh]
        end
        return boundllh
    end

    # Call the objective function once, to make sure it doesn't bug out.
    # We do this here, as within NLopt.optimize, it doesn't generate an stack
    # trace, which makes debugging impossible.
    !verbose || print("Testing llh function...")
    llhv(ϕini, Nothing)
    !verbose || println(" passed")

    # use subplex method, which is able to cope with noisy objective functions
    opt = Opt(:LN_SBPLX, ϕn)
    max_objective!(opt, llhv)
    lower_bounds!(opt, ϕmin)
    upper_bounds!(opt, ϕmax)
    ftol_rel!(opt, 1e-8)
    xtol_rel!(opt, 1e-32)
    maxtime!(opt, 11 * 60 * 60)  # 11h max
    maxeval!(opt, maxevals)     # max evaluations

    !verbose || println("    iter  result")
    optllh, optϕ, ret = NLopt.optimize(opt, ϕini)
    # add final value as last element
    evalcount += 1
    ϕs[evalcount,:] = [optϕ; optllh]

    # return dataframe
    return DataFrame(Any[ϕs[1:evalcount,i] for i in 1:ϕn+1], [ϕnames(m); :llh]), ret
end
fitmodel(m::ModelBase, o::FittingObjective; verbose=true) = fitmodel(m, o, initialϕ(m); verbose=verbose)
