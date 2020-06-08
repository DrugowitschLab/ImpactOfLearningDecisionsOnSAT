# scripts with the different models

import LinearAlgebra.BLAS: symv, syr!
using LinearAlgebra

# simulation step size and length for diffusion models
const simδt = 0.1 * 1e-3
const simmaxt = 1  # max stimulus presentation time

# generic model functions

# parameter limits
const paramlims = Dict(
    :θ => (1e-10, Inf), :θslo => (-5.0, 1.0), :θdec => (0.0, 10.0),  # bound parameters
    :tnd => (0.0, 1.0),                           # non-decision time
    :k => (0.0, Inf), :β => (0.0, 30.0),          # input scaling
    :λ => (0.0, 1.0), :σd => (0.0, Inf), :σb => (0.0, Inf),  # ADF diffusion
    :σw => (0.0, 2.0), :σwb => (0.0, 2.0),        # random weight/bias SDs
    :αw => (0.0, 1.0), :αz => (0.0, 1.0),         # delta-rule learning rates
    :α => (0.0, 1.0),                             # gamma-rule learning rates
    :τinv => (0.0, 100.0), :winh => (0.0, 10.0),  # LCA inverse leak and inhibition
    :pl => (0.0, 1.0), :pb => (0.0, 1.0))         # lapse model
minimumϕ(m::ModelBase) = [paramlims[ϕname][1] for ϕname in ϕnames(m)]
maximumϕ(m::ModelBase) = [paramlims[ϕname][2] for ϕname in ϕnames(m)]
getϕ(m::ModelBase, ϕ::AbstractVector, ϕname::Symbol) = ϕ[findfirst(ϕnames(m) .== ϕname)]
gettnd(m::ModelBase, ϕ::AbstractVector) = getϕ(m, ϕ, :tnd)

# simulate choice sequence, returning choices, RTs, (confidence)
function sim(m::ModelBase, ϕ::Vector{Float64}, evidence::Matrix{Float64})
    trials = size(evidence, 1)
    @assert size(evidence, 2) == 2
    @assert length(ϕ) == length(ϕnames(m))
    corrresps = convert(Vector{Int}, evidence[:,2] .> evidence[:,1])
    # initialize learner
    l = initlearner(m, ϕ, evidence)
    # get per-trial stats and return them
    cs = Vector{Int}(undef, trials)
    ts = Vector{Float64}(undef, trials)
    gs = Vector{Float64}(undef, trials)  # gs is only here for legacy reasons
    fill!(gs, NaN)
    for n in 1:trials
        cs[n], ts[n] = performtrial!(m, l, ϕ, view(evidence, n, :), corrresps[n])
    end
    return cs, ts, gs
end
# simulate choice sequence, returning more extensive stats than sim(.); only
# supported for model that provide performtrialwithstats!(.) function
function simwithstats(m::ModelBase, ϕ::Vector{Float64}, evidence::Matrix{Float64})
    trials = size(evidence, 1)
    @assert size(evidence, 2) == 2
    @assert length(ϕ) == length(ϕnames(m))
    corrresps = convert(Vector{Int}, evidence[:,2] .> evidence[:,1])
    # initialize learner
    l = initlearner(m, ϕ, evidence)
    # get per-trial stats and return them
    cs = Vector{Int}(undef, trials)
    ts = Vector{Float64}(undef, trials)
    gs = Vector{Float64}(undef, trials)  # gs is only here for legacy reasons
    fill!(gs, NaN)
    dts = Vector{Float64}(undef, trials)
    ws = Matrix{Float64}(undef, trials, 2)
    biases = Vector{Float64}(undef, trials)
    xs = Matrix{Float64}(undef, trials, 2)
    inputs = Matrix{Float64}(undef, trials, 2)
    for n in 1:trials
        cs[n], ts[n], dts[n], ws[n,:], biases[n], xs[n,:], inputs[n,:] = 
            performtrialwithstats!(m, l, ϕ, view(evidence, n, :), corrresps[n])
    end
    return cs, ts, gs, dts, ws, biases, xs, inputs
end
# add lapses with probability pl and bias pb
addlapse(choice::Int, pl::Real, pb::Real) = rand() < pl ? (rand() < pb ? 1 : 0) : choice
# compute learning rate
learningrate(wpre, wpost, x, t, g) = norm((wpre - wpost) ./ x)


# generic learning methods

# generic Assumed Density Filtering for probit regression, assuming pre-update
# w ~ N(μ, Σ), and post-update μ change to A μ + b, and Σ diffuse by Σd
mutable struct ADF
    μ::Vector{Float64}
    Σ::Matrix{Float64}  # only upper-triagonal part is accurate/updated
    A::Matrix{Float64}
    b::Vector{Float64}
    Σd::Matrix{Float64}
end
getμ(m::ADF) = m.μ
getΣ(m::ADF) = full(Symmetric(m.Σ, :U))
# update with observation x, outcome y in {-1, 1}
function update!(m::ADF, x::Vector{Float64}, y::Int)
    # use BLAS functions for stability / preserving symmetry
    Σx = symv('U', m.Σ, x)  # = Σ * x
    xΣx = √(1 + dot(x, Σx))  # = √(1 + x' * Σ * x)
    ywx = y * dot(x, m.μ) / xΣx
    Cw = NoverΦ(ywx)
    Ccov = abs2(Cw) + Cw * ywx
    m.μ += (y * Cw / xΣx) * Σx
    syr!('U', -Ccov / abs2(xΣx), Σx, m.Σ) # m.Σ -= (Ccov / abs2(xΣx)) * Σx * Σx'
end
function diffuse!(m::ADF)
    m.μ = m.A * m.μ + m.b
    m.Σ += m.Σd
end
# decision confidence when choosing y ∈ {-1, 1} for 'input' x at time t
function confidence(m::ADF, x::Vector{Float64}, y::Int, t::Float64, σμm2::Float64)
    xΣx = √(dot(x, symv('U', m.Σ, x)) + t + σμm2)  # = √(x' Σ x + t + 1/σμ2)
    return Φ(y * dot(x, m.μ) / xΣx)
end


###############################################################################
# Stupid, optimal model that doesn't learn anything, but always uses the
# optimal weights
#
# Parameters are
# 1 - θ, bound height
# 2 - tnd, non-decision time
# 3 - k, proportionality factor
# 4 - β, power to evidence

struct OptimalModel <: ModelBase end
ϕnames(::OptimalModel) =   [   :θ,  :tnd,    :k,    :β]
initialϕ(::OptimalModel) = [ 0.57,   0.8,    40,   1.3]
initlearner(::OptimalModel, ϕ, evidence) = [-1, 1] / √(2)
# returns choice ∈ {0, 1} and reaction time (i.e., decision time + tnd)
function performtrial!(::OptimalModel, l, ϕ, evidence, corrresp)
    c, t, x = simdiffusionconst(ϕ[3] * evidence .^ ϕ[4], l, 0, ϕ[1], simmaxt)
    return c, t + ϕ[2]
end
# with additional σμ2 arg. returns additionally confidence & learning 'rate'
function performtrial!(::OptimalModel, l, ϕ, evidence, corrresp, σμ2)
    c, t, x = simdiffusionconst(ϕ[3] * evidence .^ ϕ[4], l, 0, ϕ[1], simmaxt)
    return c, t + ϕ[2], Φ(ϕ[1] / √(t + 1 / σμ2)), 0.0
end


###############################################################################
# Stupid, optimal model that doesn't learn anything, but always uses the
# optimal weights. This variant includes lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - tnd, non-decision time
# 3 - k, proportionality factor
# 4 - β, power to evidence
# 5 - pl, lapse probability
# 6 - pb, bias probability

struct OptimalLapseModel <: ModelBase end
ϕnames(::OptimalLapseModel) =   [   :θ,  :tnd,    :k,    :β,   :pl,   :pb]
initialϕ(::OptimalLapseModel) = [  0.3,   0.3,   700,   0.8,  0.05,   0.5]
initlearner(::OptimalLapseModel, ϕ, evidence) = [-1, 1] / √(2)
function performtrial!(::OptimalLapseModel, l, ϕ, evidence, corrresp)
    c, t, x = simdiffusionconst(ϕ[3] * evidence .^ ϕ[4], l, 0, ϕ[1], simmaxt)
    return addlapse(c, ϕ[5], ϕ[6]), t + ϕ[2]
end
function performtrial!(::OptimalLapseModel, l, ϕ, evidence, corrresp, σμ2)
    c, t, x = simdiffusionconst(ϕ[3] * evidence .^ ϕ[4], l, 0, ϕ[1], simmaxt)
    return addlapse(c, ϕ[5], ϕ[6]), t + ϕ[2], Φ(ϕ[1] / √(t + 1 / σμ2)), 0.0
end


###############################################################################
# Stupid, optimal model that doesn't learn anything, but always uses the
# optimal weights. This variant includes a collapsing bound, lapses and a
# choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - θslo, slope of boundary, negative = collapsing
# 3 - tnd, non-decision time
# 4 - k, proportionality factor
# 5 - β, power to evidence
# 6 - pl, lapse probability
# 7 - pb, bias probability

struct OptimalColLapseModel <: ModelBase end
ϕnames(::OptimalColLapseModel) =   [   :θ, :θslo,  :tnd,    :k,    :β,   :pl,   :pb]
initialϕ(::OptimalColLapseModel) = [ 0.16,  0.00,  0.31,   152,  0.43,  0.10,   0.5]
initlearner(::OptimalColLapseModel, ϕ, evidence) = [-1, 1] / √(2)
function performtrial!(::OptimalColLapseModel, l, ϕ, evidence, corrresp)
    c, t, x = simdiffusionlin(ϕ[4] * evidence .^ ϕ[5], l, 0, ϕ[1], ϕ[2], simδt, simmaxt)
    return addlapse(c, ϕ[6], ϕ[7]), t + ϕ[3]
end
function performtrial!(::OptimalColLapseModel, l, ϕ, evidence, corrresp, σμ2)
    c, t, x = simdiffusionlin(ϕ[4] * evidence .^ ϕ[5], l, 0, ϕ[1], ϕ[2], simδt, simmaxt)
    return addlapse(c, ϕ[6], ϕ[7]), t + ϕ[3], Φ(ϕ[1] / √(t + 1 / σμ2)), 0.0
end

###############################################################################
# Optimal model that uses the optimal weights but learns the bias using ADF.
# This variant includes lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - tnd, non-decision time
# 3 - k, proportionality factor
# 4 - β, power to evidence
# 5 - λ, assumed across-trials weight decay
# 6 - σb, assumed bias diffusion SD
# 7 - pl, lapse probability
# 8 - pb, bias probability

struct OptimalBiasLapseModel <: ModelBase end
ϕnames(::OptimalBiasLapseModel) =   [   :θ,  :tnd,    :k,    :β,    :λ,   :σb,   :pl,   :pb]
initialϕ(::OptimalBiasLapseModel) = [ 0.16,  0.31,   152,  0.43,  0.91,  80.0,  0.10,   0.5]
function initlearner(::OptimalBiasLapseModel, ϕ, evidence)
    θ, tnd, k, β, λ, σb, pl, pb = ϕ  
    σμm2 = 1 / var(k * evidence.^β * [-1, 1] / √(2))
    return ADF(zeros(1), fill(3abs2(σb), 1, 1), fill(λ, 1, 1),
               [0.0], fill(abs2(σb), 1, 1)), σμm2
end
function performtrial!(::OptimalBiasLapseModel, l, ϕ, evidence, corrresp)
    θ, tnd, k, β, λ, σb, pl, pb = ϕ  
    adfm, σμm2 = l
    wz0 = getμ(adfm)
    z0scaled = wz0[1] * σμm2  # can be NaN if wz0[1] = 0, σμm2 = Inf
    c, t, x = simdiffusionconst(k * evidence.^β, [-1, 1] / √(2),
                                isnan(z0scaled) ? 0.0 : z0scaled, θ, simmaxt)
    σμm2 = min(σμm2, 1e60) # avoid NaN and Inf
    update!(adfm, [σμm2] / √(t + σμm2), 2corrresp - 1)
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd
end
function performtrial!(::OptimalBiasLapseModel, l, ϕ, evidence, corrresp, σμ2)
    θ, tnd, k, β, λ, σb, pl, pb = ϕ
    adfm, σμm2 = l
    wz0 = getμ(adfm)
    z0scaled = wz0[1] * σμm2  # can be NaN if wz0[3] = 0, σμm2 = Inf
    c, t, x = simdiffusionconst(k * evidence.^β, [-1, 1] / √(2),
                                isnan(z0scaled) ? 0.0 : z0scaled, θ, simmaxt)
    σμm2 = min(σμm2, 1e60) # avoid NaN and Inf
    g = confidence(adfm, [σμm2], 2c - 1, t, σμm2)
    update!(adfm, [σμm2] / √(t + σμm2), 2corrresp - 1)
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd, g, 0.0
end


###############################################################################
# Optimal model that uses the optimal weights but learns the bias using ADF.
# This variant includes a collapsing bound, lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - θslo, slope of boundary, negative = collapsing
# 3 - tnd, non-decision time
# 4 - k, proportionality factor
# 5 - β, power to evidence
# 6 - λ, assumed across-trials weight decay
# 7 - σb, assumed bias diffusion SD
# 8 - pl, lapse probability
# 9 - pb, bias probability

struct OptimalColBiasLapseModel <: ModelBase end
ϕnames(::OptimalColBiasLapseModel) =   [   :θ, :θslo,  :tnd,    :k,    :β,    :λ,   :σb,   :pl,   :pb]
initialϕ(::OptimalColBiasLapseModel) = [ 0.16,  0.00,  0.31,   152,  0.43,  0.91,  80.0,  0.10,   0.5]
function initlearner(::OptimalColBiasLapseModel, ϕ, evidence)
    θ, θslo, tnd, k, β, λ, σb, pl, pb = ϕ  
    σμm2 = 1 / var(k * evidence.^β * [-1, 1] / √(2))
    return ADF(zeros(1), fill(3abs2(σb), 1, 1), fill(λ, 1, 1), 
               [0.0], fill(abs2(σb), 1, 1)), σμm2
end
function performtrial!(::OptimalColBiasLapseModel, l, ϕ, evidence, corrresp)
    θ, θslo, tnd, k, β, λ, σb, pl, pb = ϕ  
    adfm, σμm2 = l
    wz0 = getμ(adfm)
    z0scaled = wz0[1] * σμm2  # can be NaN if wz0[1] = 0, σμm2 = Inf
    c, t, x = simdiffusionlin(k * evidence.^β, [-1, 1] / √(2),
                              isnan(z0scaled) ? 0.0 : z0scaled,
                              θ, θslo, simδt, simmaxt)
    σμm2 = min(σμm2, 1e60) # avoid NaN and Inf
    update!(adfm, [σμm2] / √(t + σμm2), 2corrresp - 1)
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd
end
function performtrial!(::OptimalColBiasLapseModel, l, ϕ, evidence, corrresp, σμ2)
    θ, θslo, tnd, k, β, λ, σb, pl, pb = ϕ  
    adfm, σμm2 = l
    wz0 = getμ(adfm)
    z0scaled = wz0[1] * σμm2  # can be NaN if wz0[3] = 0, σμm2 = Inf
    c, t, x = simdiffusionlin(k * evidence.^β, [-1, 1] / √(2),
                              isnan(z0scaled) ? 0.0 : z0scaled,
                              θ, θslo, simδt, simmaxt)
    σμm2 = min(σμm2, 1e60) # avoid NaN and Inf
    g = confidence(adfm, [σμm2], 2c - 1, t, σμm2)
    update!(adfm, [σμm2] / √(t + σμm2), 2corrresp - 1)
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd, g, 0.0
end


###############################################################################
# Model that learns the weights by assumed density filtering, but never adjusts
# the boundaries / bias. This variant includes lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - tnd, non-decision time
# 3 - k, proportionality factor
# 4 - β, power to evidence
# 5 - λ, assumed across-trials weight decay
# 6 - σd, assumed weight diffusion SD
# 7 - pl, lapse probability
# 8 - pb, bias probability

struct ADFLapseModel <: ModelBase end
ϕnames(::ADFLapseModel) =   [   :θ,  :tnd,    :k,    :β,    :λ,   :σd,   :pl,   :pb]
initialϕ(::ADFLapseModel) = [ 0.15,  0.28,  1250,   0.8,  0.95,   0.1,  0.05,   0.5]
function initlearner(::ADFLapseModel, ϕ, evidence)
    θ, tnd, k, β, λ, σd, pl, pb = ϕ    
    σμm2 = 1 / var(k * evidence.^β * [-1, 1] / √(2))  # 1 / σμm2
    σw2ini = 1 / σμm2 + abs2(σd)
    return ADF([-1, 1] / √(2), Matrix(σw2ini * I, 2, 2), Matrix(λ * I, 2, 2),
               [0.0, 0.0], Matrix(abs2(σd) * I, 2, 2)), σμm2
end
function performtrial!(::ADFLapseModel, l, ϕ, evidence, corrresp)
    θ, tnd, k, β, λ, σd, pl, pb = ϕ    
    adfm, σμm2 = l
    c, t, x = simdiffusionconst(k * evidence.^β, getμ(adfm), 0, θ, simmaxt)
    update!(adfm, x / √(t + σμm2), 2corrresp - 1)
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd
end
function performtrial!(::ADFLapseModel, l, ϕ, evidence, corrresp, σμ2)
    θ, tnd, k, β, λ, σd, pl, pb = ϕ    
    adfm, σμm2 = l
    wμpre = copy(getμ(adfm))
    c, t, x = simdiffusionconst(k * evidence.^β, wμpre, 0, θ, simmaxt)
    g = confidence(adfm, x, 2c - 1, t, σμm2)
    update!(adfm, x / √(t + σμm2), 2corrresp - 1)
    wμpost = copy(getμ(adfm))
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd, g, learningrate(wμpre, wμpost, x, t, g)
end


###############################################################################
# Model that learns the weights by assumed density filtering, but never adjusts
# the boundaries / bias. This variant includes a collapsing boundary, lapses
# and a choice bias.
#
# Parameters are
# 1 - θ, initial bound height
# 2 - θslo, slope of boundary, negative = collapsing
# 3 - tnd, non-decision time
# 4 - k, proportionality factor
# 5 - β, power to evidence
# 6 - λ, assumed across-trials weight decay
# 7 - σd, assumed weight diffusion SD
# 8 - pl, lapse probability
# 9 - pb, bias probability

struct ADFColLapseModel <: ModelBase end
ϕnames(::ADFColLapseModel) =   [   :θ, :θslo,  :tnd,    :k,    :β,    :λ,   :σd,   :pl,   :pb]
initialϕ(::ADFColLapseModel) = [ 0.19,    -0,  0.28,   355,  0.62,  0.95,   0.1,  0.09,   0.6]
function initlearner(::ADFColLapseModel, ϕ, evidence)
    θ, θslo, tnd, k, β, λ, σd, pl, pb = ϕ    
    σμm2 = 1 / var(k * evidence.^β * [-1, 1] / √(2))
    σw2ini = 1 / σμm2 + abs2(σd)
    return ADF([-1, 1] / √(2), Matrix(σw2ini * I, 2, 2), Matrix(λ * I, 2, 2),
               [0.0, 0.0], Matrix(abs2(σd) * I, 2, 2)), σμm2
end
function performtrial!(::ADFColLapseModel, l, ϕ, evidence, corrresp)
    θ, θslo, tnd, k, β, λ, σd, pl, pb = ϕ
    adfm, σμm2 = l
    c, t, x = simdiffusionlin(k * evidence.^β, getμ(adfm), 0, θ, θslo, simδt, simmaxt)
    update!(adfm, x / √(t + σμm2), 2corrresp - 1)
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd
end
function performtrial!(::ADFColLapseModel, l, ϕ, evidence, corrresp, σμ2)
    θ, θslo, tnd, k, β, λ, σd, pl, pb = ϕ
    adfm, σμm2 = l
    wμpre = copy(getμ(adfm))
    c, t, x = simdiffusionlin(k * evidence.^β, wμpre, 0, θ, θslo, simδt, simmaxt)
    g = confidence(adfm, x, 2c - 1, t, σμm2)
    update!(adfm, x / √(t + σμm2), 2corrresp - 1)
    wμpost = copy(getμ(adfm))
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd, g, learningrate(wμpre, wμpost, x, t, g)
end


###############################################################################
# Model that learns the both weights and biases by assumed density filtering.
# This variant includes lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - tnd, non-decision time
# 3 - k, proportionality factor
# 4 - β, power to evidence
# 5 - λ, assumed across-trials weight decay
# 6 - σd, assumed weight diffusion SD
# 7 - σb, assumed bias diffusion SD
# 8 - pl, lapse probability
# 9 - pb, bias probability

struct ADFBiasLapseModel <: ModelBase end
ϕnames(::ADFBiasLapseModel) =   [   :θ,  :tnd,    :k,    :β,    :λ,   :σd,   :σb,   :pl,   :pb]
initialϕ(::ADFBiasLapseModel) = [ 0.18,  0.28,   357,  0.62,  0.91,  0.16,  80.0,  0.05,   0.5]
function initlearner(::ADFBiasLapseModel, ϕ, evidence)
    θ, tnd, k, β, λ, σd, σb, pl, pb = ϕ  
    σμm2 = 1 / var(k * evidence.^β * [-1, 1] / √(2))
    σw2ini = 1 / σμm2 + abs2(σd)
    return ADF([-1, 1, 0] / √(2), Matrix(Diagonal([σw2ini, σw2ini, 3abs2(σb)])),
               Matrix(λ*I, 3, 3), [0.0, 0.0, 0.0],
               Matrix(Diagonal(abs2.([σd, σd, σb])))), σμm2
end
function performtrial!(::ADFBiasLapseModel, l, ϕ, evidence, corrresp)
    θ, tnd, k, β, λ, σd, σb, pl, pb = ϕ
    adfm, σμm2 = l
    wz0 = getμ(adfm)
    !any(isnan.(wz0[1:2])) || (println(adfm); println(ϕ))  # for debugging
    z0scaled = wz0[3] * σμm2  # can be NaN if wz0[3] = 0, σμm2 = Inf
    c, t, x = simdiffusionconst(k * evidence.^β, wz0[1:2],
                                isnan(z0scaled) ? 0.0 : z0scaled, θ, simmaxt)
    σμm2 = min(σμm2, 1e60) # avoid NaN and Inf
    update!(adfm, [x; σμm2] / √(t + σμm2), 2corrresp - 1)
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd
end
function performtrial!(::ADFBiasLapseModel, l, ϕ, evidence, corrresp, σμ2)
    θ, tnd, k, β, λ, σd, σb, pl, pb = ϕ
    adfm, σμm2 = l
    wz0 = copy(getμ(adfm))
    !any(isnan.(wz0[1:2])) || (println(adfm); println(ϕ))  # for debugging
    z0scaled = wz0[3] * σμm2  # can be NaN if wz0[3] = 0, σμm2 = Inf
    c, t, x = simdiffusionconst(k * evidence.^β, wz0[1:2],
                                isnan(z0scaled) ? 0.0 : z0scaled, θ, simmaxt)
    σμm2 = min(σμm2, 1e60) # avoid NaN and Inf
    g = confidence(adfm, [x; σμm2], 2c - 1, t, σμm2)
    update!(adfm, [x; σμm2] / √(t + σμm2), 2corrresp - 1)
    wμpost = copy(getμ(adfm))
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd, g, learningrate(wz0[1:2], wμpost[1:2], x, t, g)
end


###############################################################################
# Model that learns the both weights and biases by assumed density filtering.
# This variant includes a collapsing bound, lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - θslo, slope of boundary, negative = collapsing
# 3 - tnd, non-decision time
# 4 - k, proportionality factor
# 5 - β, power to evidence
# 6 - λ, assumed across-trials weight decay
# 7 - σd, assumed weight diffusion SD
# 8 - σb, assumed bias diffusion SD
# 9 - pl, lapse probability
# 10 - pb, bias probability

struct ADFColBiasLapseModel <: ModelBase end
ϕnames(::ADFColBiasLapseModel) =   [   :θ, :θslo,  :tnd,    :k,    :β,    :λ,   :σd,   :σb,   :pl,   :pb]
initialϕ(::ADFColBiasLapseModel) = [ 0.27,    -0,  0.27,   271,  0.57,  0.91,  0.16,  80.0,  0.05,   0.6]
function initlearner(::ADFColBiasLapseModel, ϕ, evidence)
    θ, θslo, tnd, k, β, λ, σd, σb, pl, pb = ϕ  
    σμm2 = 1 / var(k * evidence.^β * [-1, 1] / √(2))
    σw2ini = 1 / σμm2 + abs2(σd)
    return ADF([-1, 1, 0] / √(2), Matrix(Diagonal([σw2ini, σw2ini, 3abs2(σb)])),
               Matrix(λ*I, 3, 3), [0.0, 0.0, 0.0],
               Matrix(Diagonal(abs2.([σd, σd, σb])))), σμm2
end
function performtrial!(::ADFColBiasLapseModel, l, ϕ, evidence, corrresp)
    θ, θslo, tnd, k, β, λ, σd, σb, pl, pb = ϕ
    adfm, σμm2 = l
    wz0 = getμ(adfm)
    !any(isnan.(wz0[1:2])) || (println(adfm); println(ϕ))  # for debugging
    z0scaled = wz0[3] * σμm2  # can be NaN if wz0[3] = 0, σμm2 = Inf
    c, t, x = simdiffusionlin(k * evidence.^β, wz0[1:2],
                              isnan(z0scaled) ? 0.0 : z0scaled, θ, θslo, simδt, simmaxt)
    σμm2 = min(σμm2, 1e60) # avoid NaN and Inf
    update!(adfm, [x; σμm2] / √(t + σμm2), 2corrresp - 1)
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd
end
function performtrial!(::ADFColBiasLapseModel, l, ϕ, evidence, corrresp, σμ2)
    θ, θslo, tnd, k, β, λ, σd, σb, pl, pb = ϕ
    adfm, σμm2 = l
    wz0 = copy(getμ(adfm))
    !any(isnan.(wz0[1:2])) || (println(adfm); println(ϕ))  # for debugging
    z0scaled = wz0[3] * σμm2  # can be NaN if wz0[3] = 0, σμm2 = Inf
    c, t, x = simdiffusionlin(k * evidence.^β, wz0[1:2],
                              isnan(z0scaled) ? 0.0 : z0scaled, θ, θslo, simδt, simmaxt)
    σμm2 = min(σμm2, 1e60) # avoid NaN and Inf
    g = confidence(adfm, [x; σμm2], 2c - 1, t, σμm2)
    update!(adfm, [x; σμm2] / √(t + σμm2), 2corrresp - 1)
    wμpost = copy(getμ(adfm))
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd, g, learningrate(wz0[1:2], wμpost[1:2], x, t, g)
end
function performtrialwithstats!(::ADFColBiasLapseModel, l, ϕ, evidence, corrresp)
    θ, θslo, tnd, k, β, λ, σd, σb, pl, pb = ϕ
    adfm, σμm2 = l
    wz0 = copy(getμ(adfm))
    !any(isnan.(wz0[1:2])) || (println(adfm); println(ϕ))  # for debugging
    z0scaled = wz0[3] * σμm2  # can be NaN if wz0[3] = 0, σμm2 = Inf
    input = k * evidence.^β
    c, t, x = simdiffusionlin(input, wz0[1:2],
                              isnan(z0scaled) ? 0.0 : z0scaled, θ, θslo, simδt, simmaxt)
    σμm2 = min(σμm2, 1e60) # avoid NaN and Inf
    update!(adfm, [x; σμm2] / √(t + σμm2), 2corrresp - 1)
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd, t, wz0[1:2], z0scaled, x, input
end


###############################################################################
# Model that learns biases by assumed density filtering but uses random weights.
# This variant includes a collapsing bound, lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - θslo, slope of boundary, negative = collapsing
# 3 - tnd, non-decision time
# 4 - k, proportionality factor
# 5 - β, power to evidence
# 6 - σw, standard deviation for random weight draw
# 7 - λ, assumed across-trials weight decay
# 8 - σb, assumed bias diffusion SD
# 9 - pl, lapse probability
# 10 - pb, bias probability

struct RandomColBiasLapseModel <: ModelBase end
ϕnames(::RandomColBiasLapseModel) =   [   :θ, :θslo,  :tnd,    :k,    :β,   :σw,    :λ,   :σb,   :pl,   :pb]
initialϕ(::RandomColBiasLapseModel) = [ 0.27,    -0,  0.27,   271,  0.57,   0.5,  0.91,  80.0,  0.05,   0.6]
function initlearner(::RandomColBiasLapseModel, ϕ, evidence)
    θ, θslo, tnd, k, β, σw, λ, σb, pl, pb = ϕ  
    σμm2 = 1 / var(k * evidence.^β * [-1, 1] / √(2))
    return ADF([0.0], fill(3abs2(σb), 1, 1),
               fill(λ, 1, 1), [0.0], fill(abs2(σb), 1, 1)), σμm2
end
function performtrial!(::RandomColBiasLapseModel, l, ϕ, evidence, corrresp)
    θ, θslo, tnd, k, β, σw, λ, σb, pl, pb = ϕ
    adfm, σμm2 = l
    wz0 = [-1, 1] / √(2) + σw * randn(2)
    wz0 /= norm(wz0) # random direction with unit norm
    z0scaled = getμ(adfm)[1] * σμm2  # can be NaN if μ = 0, σμm2 = Inf
    c, t, x = simdiffusionlin(k * evidence.^β, wz0,
                              isnan(z0scaled) ? 0.0 : z0scaled, θ, θslo, simδt, simmaxt)
    σμm2 = min(σμm2, 1e60) # avoid NaN and Inf
    update!(adfm, [σμm2] / √(t + σμm2), 2corrresp - 1)
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd
end
function performtrial!(::RandomColBiasLapseModel, l, ϕ, evidence, corrresp, σμ2)
    θ, θslo, tnd, k, β, σw, λ, σb, pl, pb = ϕ
    adfm, σμm2 = l
    wz0 = [-1, 1] / √(2) + σw * randn(2)
    wz0 /= norm(wz0) # random direction with unit norm
    z0scaled = getμ(adfm)[1] * σμm2  # can be NaN if μ = 0, σμm2 = Inf
    c, t, x = simdiffusionlin(k * evidence.^β, wz0,
                              isnan(z0scaled) ? 0.0 : z0scaled, θ, θslo, simδt, simmaxt)
    σμm2 = min(σμm2, 1e60) # avoid NaN and Inf
    g = confidence(adfm, [σμm2], 2c - 1, t, σμm2)
    update!(adfm, [σμm2] / √(t + σμm2), 2corrresp - 1)
    wμpost = [-1, 1] / √(2) + σw * randn(2)
    wμpost /= norm(wμpost) # random direction with unit norm
    diffuse!(adfm)
    return addlapse(c, pl, pb), t + tnd, g, learningrate(wz0, wμpost, x, t, g)
end


###############################################################################
# Model that uses both random weights and biases.
# This variant includes a collapsing bound, lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - θslo, slope of boundary, negative = collapsing
# 3 - tnd, non-decision time
# 4 - k, proportionality factor
# 5 - β, power to evidence
# 6 - σw, standard deviation for random weight draw
# 6 - σwb, standard deviation for random bias draw
# 9 - pl, lapse probability
# 10 - pb, bias probability

struct RandomColLapseModel <: ModelBase end
ϕnames(::RandomColLapseModel) =   [   :θ, :θslo,  :tnd,    :k,    :β,   :σw,  :σwb,   :pl,   :pb]
initialϕ(::RandomColLapseModel) = [ 0.27,    -0,  0.27,   271,  0.57,   0.5,   0.5,  0.05,   0.6]
function initlearner(::RandomColLapseModel, ϕ, evidence)
    θ, θslo, tnd, k, β, σw, σwb, pl, pb = ϕ
    σμm2 = 1 / var(k * evidence.^β * [-1, 1] / √(2))
    return σμm2
end
function performtrial!(::RandomColLapseModel, σμm2, ϕ, evidence, corrresp)
    θ, θslo, tnd, k, β, σw, σwb, pl, pb = ϕ
    wz0 = [-1, 1] / √(2) + σw * randn(2)
    wz0 /= norm(wz0) # random direction with unit norm
    z0scaled = randn() * σwb * σμm2
    c, t, x = simdiffusionlin(k * evidence.^β, wz0, z0scaled, θ, θslo, simδt, simmaxt)
    return addlapse(c, pl, pb), t + tnd
end
function performtrial!(::RandomColLapseModel, σμm2, ϕ, evidence, corrresp, σμ2)
    θ, θslo, tnd, k, β, σw, σwb, pl, pb = ϕ
    wz0 = [-1, 1] / √(2) + σw * randn(2)
    wz0 /= norm(wz0) # random direction with unit norm
    z0scaled = randn() * σwb * σμm2
    c, t, x = simdiffusionlin(k * evidence.^β, wz0, z0scaled, θ, θslo, simδt, simmaxt)
    σμm2 = min(σμm2, 1e60) # avoid NaN and Inf
    g = 0.5  # arbitrary confidence
    wμpost = [-1, 1] / √(2) + σw * randn(2)
    wμpost /= norm(wμpost) # random direction with unit norm
    return addlapse(c, pl, pb), t + tnd, g, learningrate(wz0, wμpost, x, t, g)
end

###############################################################################
# Model that learns the weights by gamma rule without adjusting the boundaries /
# bias. This variant includes lapses and a choice bias.
# w(n+1) = w(n) + α (θcorr - θchosen) / θ * x and subseq. norm
#
# Parameters are
# 1 - θ, bound height
# 2 - tnd, non-decision time
# 3 - k, proportionality factor
# 4 - β, power to evidence
# 5 - α, learning rate for weight update
# 6 - pl, lapse probability
# 7 - pb, bias probability

struct GammaLapseModel <: ModelBase end
ϕnames(::GammaLapseModel) =   [   :θ,  :tnd,    :k,    :β,    :α,   :pl,   :pb]
initialϕ(::GammaLapseModel) = [ 0.33,  0.28,   386,  0.69,  0.07,  0.05,   0.5]
initlearner(::GammaLapseModel, ϕ, evidence) = [-1, 1] / √(2)
function performtrial!(::GammaLapseModel, l, ϕ, evidence, corrresp)
    c, t, x = simdiffusionconst(ϕ[3] * evidence.^ϕ[4], l, 0, ϕ[1], simmaxt)
    c = addlapse(c, ϕ[6], ϕ[7])
    # update weights by simplified delta-rule, then re-normalize
    l += ϕ[5] * 2 * (corrresp - c) * x  # focus on corr/incorr, skipping θ-normaliz.
    l /= norm(l)
    return c, t + ϕ[2]
end
function performtrial!(::GammaLapseModel, l, ϕ, evidence, corrresp, σμ2)
    c, t, x = simdiffusionconst(ϕ[3] * evidence.^ϕ[4], l, 0, ϕ[1], simmaxt)
    c = addlapse(c, ϕ[6], ϕ[7])
    g = Φ(ϕ[1] / √(t + 1 / σμ2))
    # update weights by delta-rule, then re-normalize
    lpre = copy(l)
    l += ϕ[5] * 2 * (corrresp - c) * x  # focus on corr/incorr, skipping θ-normaliz.
    lpost = copy(l)
    l /= norm(l)
    return c, t + ϕ[2], g, learningrate(lpre, lpost, x, t, g)
end


###############################################################################
# Model that learns the weights by gamma rule without adjusting the boundaries /
# bias. This variant includes collapsing boundaries, lapses and a choice bias.
# w(n+1) = w(n) + α (θcorr(t) - θchosen(t)) / θ(0) * x and subseq. norm
#
# Parameters are
# 1 - θ, bound height
# 2 - θslo, slope of boundary, negative = collapsing
# 3 - tnd, non-decision time
# 4 - k, proportionality factor
# 5 - β, power to evidence
# 6 - α, learning rate for weight update
# 7 - pl, lapse probability
# 8 - pb, bias probability

struct GammaColLapseModel <: ModelBase end
ϕnames(::GammaColLapseModel) =   [   :θ, :θslo,  :tnd,    :k,    :β,    :α,   :pl,   :pb]
initialϕ(::GammaColLapseModel) = [ 0.35,  -0.0,  0.28,   496,  0.72,  0.10,  0.06,   0.5]
initlearner(::GammaColLapseModel, ϕ, evidence) = [-1, 1] / √(2)
function performtrial!(::GammaColLapseModel, l, ϕ, evidence, corrresp)
    c, t, x = simdiffusionlin(ϕ[4] * evidence.^ϕ[5], l, 0, ϕ[1], ϕ[2], simδt, simmaxt)
    c = addlapse(c, ϕ[7], ϕ[8])
    # update weights by delta-rule, then re-normalize
    l += ϕ[6] * 2 * (corrresp - c) * (1 + t * ϕ[2] / ϕ[1]) * x
    l /= norm(l)
    return c, t + ϕ[3]
end
function performtrial!(::GammaColLapseModel, l, ϕ, evidence, corrresp, σμ2)
    c, t, x = simdiffusionlin(ϕ[4] * evidence.^ϕ[5], l, 0, ϕ[1], ϕ[2], simδt, simmaxt)
    c = addlapse(c, ϕ[7], ϕ[8])
    g = Φ((ϕ[1] + t * ϕ[2]) / √(t + 1 / σμ2))
    # update weights by delta-rule, then re-normalize
    lpre = copy(l)
    l += ϕ[6] * 2 * (corrresp - c) * (1 + t * ϕ[2] / ϕ[1]) * x
    lpost = copy(l)
    l /= norm(l)
    return c, t + ϕ[3], g, learningrate(lpre, lpost, x, t, g)
end


###############################################################################
# Model that learns the weights and boundaries / bias by gamma rule. 
# This variant includes lapses and a choice bias.
# w(n+1) = w(n) + α (θcorr - θchosen) / θ * x and subseq. norm
# z0(n+1) = z0(n) + α (θcorr - θchosen) / θ
#
# Parameters are
# 1 - θ, bound height
# 2 - tnd, non-decision time
# 3 - k, proportionality factor
# 4 - β, power to evidence
# 5 - α, learning rate for both updates
# 6 - pl, lapse probability
# 7 - pb, bias probability

struct GammaBiasLapseModel <: ModelBase end
struct GammaBiasState
    u::Vector{Float64}
    z0::Vector{Float64}  # single-element vector to make it mutable
end
ϕnames(::GammaBiasLapseModel) =   [   :θ,  :tnd,    :k,    :β,    :α,   :pl,   :pb]
initialϕ(::GammaBiasLapseModel) = [ 0.38,  0.27,   330,  0.70,  0.08,  0.05,   0.5]
initlearner(::GammaBiasLapseModel, ϕ, evidence) = GammaBiasState([-1, 1] / √(2), [0.0])
function performtrial!(::GammaBiasLapseModel, l, ϕ, evidence, corrresp)
    c, t, x = simdiffusionconst(ϕ[3] * evidence.^ϕ[4], l.u, l.z0[1], ϕ[1], simmaxt)
    c = addlapse(c, ϕ[6], ϕ[7])
    # update weights by delta-rule, then re-normalize
    αres = ϕ[5] * 2 * (corrresp - c)
    l.u[:] += αres * x
    l.u[:] /= norm(l.u)
    # updatde boundary bias by delta-rule without re-normalization
    l.z0[1] += αres
    return c, t + ϕ[2]
end
function performtrial!(::GammaBiasLapseModel, l, ϕ, evidence, corrresp, σμ2)
    c, t, x = simdiffusionconst(ϕ[3] * evidence.^ϕ[4], l.u, l.z0[1], ϕ[1], simmaxt)
    c = addlapse(c, ϕ[6], ϕ[7])
    g = Φ(ϕ[1] / √(t + 1 / σμ2))
    # update weights by delta-rule, then re-normalize
    upre = copy(l.u)
    αres = ϕ[5] * 2 * (corrresp - c)
    l.u[:] += αres * x
    upost = copy(l.u)
    l.u[:] /= norm(l.u)
    # updatde boundary bias by delta-rule without re-normalization
    l.z0[1] += αres
    return c, t + ϕ[2], g, learningrate(upre, upost, x, t, g)
end


###############################################################################
# Model that learns the weights and boundaries / bias by gamma rule. 
# This variant includes a collapsing bound, lapses and a choice bias.
# w(n+1) = w(n) + α (θcorr(t) - θchosen(t)) / θ(0) * x and subseq. norm
# z0(n+1) = z0(n) + α (θcorr(t) - θchosen(t)) / θ(0)
#
# Parameters are
# 1 - θ, bound height
# 2 - θslo, slope of boundary, negative = collapsing
# 3 - tnd, non-decision time
# 4 - k, proportionality factor
# 5 - β, power to evidence
# 6 - α, learning rate for both updates
# 7 - pl, lapse probability
# 8 - pb, bias probability

struct GammaColBiasLapseModel <: ModelBase end
ϕnames(::GammaColBiasLapseModel) =   [   :θ, :θslo,  :tnd,    :k,    :β,    :α,   :pl,   :pb]
initialϕ(::GammaColBiasLapseModel) = [ 0.38,   0.0,  0.27,   259,  0.64,  0.08,  0.05,   0.5]
initlearner(::GammaColBiasLapseModel, ϕ, evidence) = GammaBiasState([-1, 1] / √(2), [0.0])
function performtrial!(::GammaColBiasLapseModel, l, ϕ, evidence, corrresp)
    c, t, x = simdiffusionlin(ϕ[4] * evidence.^ϕ[5], l.u, l.z0[1], ϕ[1], ϕ[2], simδt, simmaxt)
    c = addlapse(c, ϕ[7], ϕ[8])
    # update weights by delta-rule, then re-normalize
    αres = ϕ[6] * 2 * (corrresp - c) * (1 + t * ϕ[2] / ϕ[1])
    l.u[:] += αres * x
    l.u[:] /= norm(l.u)
    # updatde boundary bias by delta-rule without re-normalization
    l.z0[1] += αres
    return c, t + ϕ[3]
end
function performtrial!(::GammaColBiasLapseModel, l, ϕ, evidence, corrresp, σμ2)
    c, t, x = simdiffusionlin(ϕ[4] * evidence.^ϕ[5], l.u, l.z0[1], ϕ[1], ϕ[2], simδt, simmaxt)
    c = addlapse(c, ϕ[7], ϕ[8])
    g = Φ((ϕ[1] + t * ϕ[2]) / √(t + 1 / σμ2))
    # update weights by delta-rule, then re-normalize
    upre = copy(l.u)
    αres = ϕ[6] * 2 * (corrresp - c) * (1 + t * ϕ[2] / ϕ[1])
    l.u[:] += αres * x
    upost = copy(l.u)
    l.u[:] /= norm(l.u)
    # updatde boundary bias by delta-rule without re-normalization
    l.z0[1] += αres
    return c, t + ϕ[3], g, learningrate(upre, upost, x, t, g)
end


###############################################################################
# Model that learns the weights by delta rule without adjusting the boundaries /
# bias. This variant includes lapses and a choice bias.
# w(n+1) = w(n) + αw (corrresp - resp) * x and subseq. norm, resp ∈ {0, 1}
#
# Parameters are
# 1 - θ, bound height
# 2 - tnd, non-decision time
# 3 - k, proportionality factor
# 4 - β, power to evidence
# 5 - αw, learning rate for weight update
# 6 - pl, lapse probability
# 7 - pb, bias probability

struct DeltaLapseModel <: ModelBase end
ϕnames(::DeltaLapseModel) =   [   :θ,  :tnd,    :k,    :β,   :αw,   :pl,   :pb]
initialϕ(::DeltaLapseModel) = [ 0.33,  0.28,   386,  0.69,  0.15,  0.05,   0.5]
initlearner(::DeltaLapseModel, ϕ, evidence) = [-1, 1] / √(2)
function performtrial!(::DeltaLapseModel, l, ϕ, evidence, corrresp)
    c, t, x = simdiffusionconst(ϕ[3] * evidence.^ϕ[4], l, 0, ϕ[1], simmaxt)
    c = addlapse(c, ϕ[6], ϕ[7])
    # update weights by delta-rule, then re-normalize
    l += ϕ[5] * (corrresp - c) * x
    l /= norm(l)
    return c, t + ϕ[2]
end
function performtrial!(::DeltaLapseModel, l, ϕ, evidence, corrresp, σμ2)
    c, t, x = simdiffusionconst(ϕ[3] * evidence.^ϕ[4], l, 0, ϕ[1], simmaxt)
    c = addlapse(c, ϕ[6], ϕ[7])
    g = Φ(ϕ[1] / √(t + 1 / σμ2))
    # update weights by delta-rule, then re-normalize
    lpre = copy(l)
    l += ϕ[5] * (corrresp - c) * x
    lpost = copy(l)
    l /= norm(l)
    return c, t + ϕ[2], g, learningrate(lpre, lpost, x, t, g)
end


###############################################################################
# Model that learns the weights by delta rule without adjusting the boundaries /
# bias. This variant includes collapsing boundaries, lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - θslo, slope of boundary, negative = collapsing
# 3 - tnd, non-decision time
# 4 - k, proportionality factor
# 5 - β, power to evidence
# 6 - αw, learning rate for weight update
# 7 - pl, lapse probability
# 8 - pb, bias probability

struct DeltaColLapseModel <: ModelBase end
ϕnames(::DeltaColLapseModel) =   [   :θ, :θslo,  :tnd,    :k,    :β,   :αw,   :pl,   :pb]
initialϕ(::DeltaColLapseModel) = [ 0.35,  -0.0,  0.28,   496,  0.72,  0.19,  0.06,   0.5]
initlearner(::DeltaColLapseModel, ϕ, evidence) = [-1, 1] / √(2)
function performtrial!(::DeltaColLapseModel, l, ϕ, evidence, corrresp)
    c, t, x = simdiffusionlin(ϕ[4] * evidence.^ϕ[5], l, 0, ϕ[1], ϕ[2], simδt, simmaxt)
    c = addlapse(c, ϕ[7], ϕ[8])
    # update weights by delta-rule, then re-normalize
    l += ϕ[6] * (corrresp - c) * x
    l /= norm(l)
    return c, t + ϕ[3]
end
function performtrial!(::DeltaColLapseModel, l, ϕ, evidence, corrresp, σμ2)
    c, t, x = simdiffusionlin(ϕ[4] * evidence.^ϕ[5], l, 0, ϕ[1], ϕ[2], simδt, simmaxt)
    c = addlapse(c, ϕ[7], ϕ[8])
    g = Φ((ϕ[1] + t * ϕ[2]) / √(t + 1 / σμ2))
    # update weights by delta-rule, then re-normalize
    lpre = copy(l)
    l += ϕ[6] * (corrresp - c) * x
    lpost = copy(l)
    l /= norm(l)
    return c, t + ϕ[3], g, learningrate(lpre, lpost, x, t, g)
end


###############################################################################
# Model that learns the weights and boundaries / bias by delta rule. 
# This variant includes lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - tnd, non-decision time
# 3 - k, proportionality factor
# 4 - β, power to evidence
# 5 - αw, learning rate for weight update
# 6 - αz, learning rate for boundary/bias
# 7 - pl, lapse probability
# 8 - pb, bias probability

struct DeltaBiasLapseModel <: ModelBase end
struct DeltaBiasState
    u::Vector{Float64}
    z0::Vector{Float64}  # single-element vector to make it mutable
end
ϕnames(::DeltaBiasLapseModel) =   [   :θ,  :tnd,    :k,    :β,   :αw,   :αz,   :pl,   :pb]
initialϕ(::DeltaBiasLapseModel) = [ 0.38,  0.27,   330,  0.70,  0.17,  0.23,  0.05,   0.5]
initlearner(::DeltaBiasLapseModel, ϕ, evidence) = DeltaBiasState([-1, 1] / √(2), [0.0])
function performtrial!(::DeltaBiasLapseModel, l, ϕ, evidence, corrresp)
    c, t, x = simdiffusionconst(ϕ[3] * evidence.^ϕ[4], l.u, l.z0[1], ϕ[1], simmaxt)
    c = addlapse(c, ϕ[7], ϕ[8])
    # update weights by delta-rule, then re-normalize
    l.u[:] += ϕ[5] * (corrresp - c) * x
    l.u[:] /= norm(l.u)
    # updatde boundary bias by delta-rule without re-normalization
    corrθ = ϕ[1] * (2corrresp - 1)
    l.z0[1] += ϕ[6] * (corrθ - l.z0[1])
    return c, t + ϕ[2]
end
function performtrial!(::DeltaBiasLapseModel, l, ϕ, evidence, corrresp, σμ2)
    c, t, x = simdiffusionconst(ϕ[3] * evidence.^ϕ[4], l.u, l.z0[1], ϕ[1], simmaxt)
    c = addlapse(c, ϕ[7], ϕ[8])
    g = Φ(ϕ[1] / √(t + 1 / σμ2))
    # update weights by delta-rule, then re-normalize
    upre = copy(l.u)
    l.u[:] += ϕ[5] * (corrresp - c) * x
    upost = copy(l.u)
    l.u[:] /= norm(l.u)
    # updatde boundary bias by delta-rule without re-normalization
    corrθ = ϕ[1] * (2corrresp - 1)
    l.z0[1] += ϕ[6] * (corrθ - l.z0[1])
    return c, t + ϕ[2], g, learningrate(upre, upost, x, t, g)
end


###############################################################################
# Model that learns the weights and boundaries / bias by delta rule. 
# This variant includes a collapsing bound, lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - θslo, slope of boundary, negative = collapsing
# 3 - tnd, non-decision time
# 4 - k, proportionality factor
# 5 - β, power to evidence
# 6 - αw, learning rate for weight update
# 7 - αz, learning rate for boundary/bias
# 8 - pl, lapse probability
# 9 - pb, bias probability

struct DeltaColBiasLapseModel <: ModelBase end
ϕnames(::DeltaColBiasLapseModel) =   [   :θ, :θslo,  :tnd,    :k,    :β,   :αw,   :αz,   :pl,   :pb]
initialϕ(::DeltaColBiasLapseModel) = [ 0.38,   0.0,  0.27,   259,  0.64,  0.17,  0.23,  0.05,   0.5]
initlearner(::DeltaColBiasLapseModel, ϕ, evidence) = DeltaBiasState([-1, 1] / √(2), [0.0])
function performtrial!(::DeltaColBiasLapseModel, l, ϕ, evidence, corrresp)
    c, t, x = simdiffusionlin(ϕ[4] * evidence.^ϕ[5], l.u, l.z0[1], ϕ[1], ϕ[2], simδt, simmaxt)
    c = addlapse(c, ϕ[8], ϕ[9])
    # update weights by delta-rule, then re-normalize
    l.u[:] += ϕ[6] * (corrresp - c) * x
    l.u[:] /= norm(l.u)
    # updatde boundary bias by delta-rule without re-normalization
    corrθ = (ϕ[1] + t * ϕ[2]) * (2corrresp - 1)
    l.z0[1] += ϕ[7] * (corrθ - l.z0[1])
    return c, t + ϕ[3]
end
function performtrial!(::DeltaColBiasLapseModel, l, ϕ, evidence, corrresp, σμ2)
    c, t, x = simdiffusionlin(ϕ[4] * evidence.^ϕ[5], l.u, l.z0[1], ϕ[1], ϕ[2], simδt, simmaxt)
    c = addlapse(c, ϕ[8], ϕ[9])
    g = Φ((ϕ[1] + t * ϕ[2]) / √(t + 1 / σμ2))
    # update weights by delta-rule, then re-normalize
    upre = copy(l.u)
    l.u[:] += ϕ[6] * (corrresp - c) * x
    upost = copy(l.u)
    l.u[:] /= norm(l.u)
    # updatde boundary bias by delta-rule without re-normalization
    corrθ = (ϕ[1] + t * ϕ[2]) * (2corrresp - 1)
    l.z0[1] += ϕ[7] * (corrθ - l.z0[1])
    return c, t + ϕ[3], g, learningrate(upre, upost, x, t, g)
end


###############################################################################
# Model that learns the weights and boundaries / bias by delta rule with
# exponentially collapsing bound, lapses and a choice bias. 
#
# Parameters are
# 1 - θ, bound height
# 2 - θdec, inv. time constant of bound decay
# 3 - tnd, non-decision time
# 4 - k, proportionality factor
# 5 - β, power to evidence
# 6 - αw, learning rate for weight update
# 7 - αz, learning rate for boundary/bias
# 8 - pl, lapse probability
# 9 - pb, bias probability

struct DeltaExpColBiasLapseModel <: ModelBase end
struct DeltaExpBiasState
    u::Vector{Float64}
    z0::Vector{Float64}  # single-element vector to make it mutable
    θbase::Vector{Float64}
end
ϕnames(::DeltaExpColBiasLapseModel) =   [   :θ, :θdec,  :tnd,    :k,    :β,   :αw,   :αz,   :pl,   :pb]
# values that I got from Andre: he is using my evidence * 10, which can be
# compensated for by using k = k(Andre) / 0.1^beta(Andre).
initialϕ(::DeltaExpColBiasLapseModel) = [0.4514, 2.1837, 0.275, 626, 0.705, 0.047, 0.133,  0.05,   0.5]
initlearner(::DeltaExpColBiasLapseModel, ϕ, evidence) = DeltaExpBiasState(
    [-1, 1] / √(2), [0.0], [ϕ[1] * exp(-t*ϕ[2]) for t in 0:simδt:simmaxt])
function performtrial!(::DeltaExpColBiasLapseModel, l, ϕ, evidence, corrresp)
    θ, θdec, tnd, k, β, αw, αz, pl, pb = ϕ
    c, t, x = simdiffusionvec(k * evidence .^ β, l.u, l.z0[1], l.θbase, simδt, simmaxt)
    # add lapses, or update parameters
    if rand() < pl
        c = (rand() < pb) ? 1 : 0
    else
        # no lapse - update parameters
        λ = 2corrresp - 1    # λ in {1, -1}
        z = (c == 1 ? θ : -θ) * exp(-θdec * t) - l.z0[1] # distance travelled
        l.z0[1] += αz * (λ - l.z0[1] / θ)
        l.u[:] += αw * (λ - z / θ) * x
        l.u[:] /= norm(l.u)
    end
    return c, t + tnd
end
function performtrial!(::DeltaExpColBiasLapseModel, l, ϕ, evidence, corrresp, σμ2)
    θ, θdec, tnd, k, β, αw, αz, pl, pb = ϕ
    c, t, x = simdiffusionvec(k * evidence .^ β, l.u, l.z0[1], l.θbase, simδt, simmaxt)
    θt = θ * exp(-θdec * t)
    g = Φ(θt / √(t + 1 / σμ2))
    # add lapses, or update parameters
    upre = upost = copy(l.u)
    if rand() < pl
        c = (rand() < pb) ? 1 : 0
    else
        # no lapse - update parameters
        λ = 2corrresp - 1    # λ in {1, -1}
        z = (c == 1 ? θt : -θt) - l.z0[1] # distance travelled
        l.z0[1] += αz * (λ - l.z0[1] / θ)
        l.u[:] += αw * (λ - z / θ) * x
        upost = copy(l.u)
        l.u[:] /= norm(l.u)
    end
    return c, t + tnd, g, learningrate(upre, upost, x, t, g)
end
function performtrialwithstats!(::DeltaExpColBiasLapseModel, l, ϕ, evidence, corrresp)
    θ, θdec, tnd, k, β, αw, αz, pl, pb = ϕ
    input = k * evidence .^ β
    upre = copy(l.u)
    z0pre = copy(l.z0[1])
    c, t, x = simdiffusionvec(input, upre, z0pre, l.θbase, simδt, simmaxt)
    # add lapses, or update parameters
    if rand() < pl
        c = (rand() < pb) ? 1 : 0
    else
        # no lapse - update parameters
        λ = 2corrresp - 1    # λ in {1, -1}
        z = (c == 1 ? θ : -θ) * exp(-θdec * t) - l.z0[1] # distance travelled
        l.z0[1] += αz * (λ - l.z0[1] / θ)
        l.u[:] += αw * (λ - z / θ) * x
        l.u[:] /= norm(l.u)
    end
    return c, t + tnd, t, upre, z0pre, x, input
end


###############################################################################
# Non-learning LCA model with time-invariant boundaries.
# This variant includes lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - tnd, non-decision time
# 3 - k, proportionality factor
# 4 - β, power to evidence
# 5 - τinv, inverse of leak time-constant
# 6 - winh, accumulator cross-inhibition
# 7 - pl, lapse probability
# 8 - pb, bias probability

struct LCALapseModel <: ModelBase end
ϕnames(::LCALapseModel) =   [   :θ,  :tnd,    :k,    :β,  :τinv, :winh,   :pl,   :pb]
initialϕ(::LCALapseModel) = [  0.5,   0.3,   200,   0.5,    0.1,   0.3,  0.05,   0.5]
initlearner(::LCALapseModel, ϕ, evidence) = Nothing
function performtrial!(::LCALapseModel, l, ϕ, evidence, corrresp)
    c, t, x = simlca(ϕ[3] * evidence .^ ϕ[4],
        fill(ϕ[5], 2), fill(ϕ[6], 2), ϕ[1], ϕ[1], simδt, simmaxt)
    return addlapse(c, ϕ[7], ϕ[8]), t + ϕ[2]
end
function performtrial!(::LCALapseModel, l, ϕ, evidence, corrresp, σμ2)
    c, t, x = simlca(ϕ[3] * evidence .^ ϕ[4],
        fill(ϕ[5], 2), fill(ϕ[6], 2), ϕ[1], ϕ[1], simδt, simmaxt)
    return addlapse(c, ϕ[7], ϕ[8]), t + ϕ[2], Φ(ϕ[1] / √(t + 1 / σμ2)), 0.0
end


###############################################################################
# Non-learning LCA model with linearly changing boundaries.
# This variant includes lapses and a choice bias.
#
# Parameters are
# 1 - θ, bound height
# 2 - θslo, slope of boundary, negative = collapsing
# 3 - tnd, non-decision time
# 4 - k, proportionality factor
# 5 - β, power to evidence
# 6 - τinv, inverse of leak time-constant
# 7 - winh, accumulator cross-inhibition
# 8 - pl, lapse probability
# 9 - pb, bias probability

struct LCAColLapseModel <: ModelBase end
ϕnames(::LCAColLapseModel) =   [   :θ, :θslo,  :tnd,    :k,    :β,  :τinv, :winh,   :pl,   :pb]
initialϕ(::LCAColLapseModel) = [  0.5,   0.0,   0.3,   200,   0.5,    0.1,   0.3,  0.05,   0.5]
initlearner(::LCAColLapseModel, ϕ, evidence) = Nothing
function performtrial!(::LCAColLapseModel, l, ϕ, evidence, corrresp)
    θn = collect(ϕ[1] .+ (simδt * ϕ[2]) .* (0:floor(Int, simmaxt/simδt)))
    c, t, x = simlca(ϕ[4] * evidence .^ ϕ[5],
        fill(ϕ[6], 2), fill(ϕ[7], 2), θn, θn, simδt, simmaxt)
    return addlapse(c, ϕ[8], ϕ[9]), t + ϕ[3]
end
function performtrial!(::LCAColLapseModel, l, ϕ, evidence, corrresp, σμ2)
    θn = collect(ϕ[1] .+ (simδt * ϕ[2]) .* (0:floor(Int, simmaxt/simδt)))
    c, t, x = simlca(ϕ[4] * evidence .^ ϕ[5],
        fill(ϕ[6], 2), fill(ϕ[7], 2), θn, θn, simδt, simmaxt)
    return addlapse(c, ϕ[8], ϕ[9]), t + ϕ[2],
        Φ(θn[min(length(θn), round(Int, t/simδt)+1)] / √(t + 1 / σμ2)), 0.0
end

