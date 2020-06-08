# common functions and constants

using DiffModels, Optim

# model base class
abstract type ModelBase end

# constants
const pmin = 1e-100
const sqrttwo = √(2)
const sqrttwoπ = √(2 * π)
const logtwoπ = log(2 * π)
const minwnorm = 1e-60  # smallest size of w (to avoid zero drift)

# returns if currently in interactive environment
inrepl() = isinteractive() || isdefined(Main, :IJulia) && Main.IJulia.inited

# robust mean, cutting top and bottom 10% (p) of values before performing mena
function robustmean(x::AbstractVector, p::Real=0.1)
    i = floor(Integer, p * length(x))
    mean(sort(x)[i:end-i])
end

# cumulative Gaussians utility functions
invΦ(p) = sqrttwo * erfinv(2p - 1)
Φ(x) = (1 + erf(x / sqrttwo)) / 2

# computes N(x|0,1) / Φ(x), more stable for asymptotes at small x
NoverΦ(x) = (x > -6 ? sqrt(2 / π) * exp(-0.5abs2(x)) / (1 + erf(x / sqrttwo)) : 
                      -x / (1 - 1/abs2(x) + 3/x^4))


# returns moving average of window size w
function movingavg(x::AbstractVector, w)
    n = length(x)
    if w > n
        return [mean(x)]
    end
    y = Array(typeof(mean(x[1])), n - w + 1)
    # this could be made faster by taking a running moving average, at the cost
    # of numerical stability
    for i = 1:length(y)
        y[i] = mean(slice(x, i:(i+w-1)))
    end
    y
end


# function to find probit fit to psychometric curve
# takes xn vector and associated pn=p(y=1|xn) vector and returns
# μ and σ that maximize likelihood of this data
function psychprobitfit(xn::AbstractVector, pn::AbstractVector)
    n = length(xn)
    @assert n == length(pn)
    # (neg) log-likelihood function, x = [μ, σ]
    function f(x::AbstractVector)
        llh = 0.0
        for i = 1:n
            xi = (xn[i] - x[1]) / x[2]
            llh += pn[i] * log(max(pmin,Φ(xi))) + (1 - pn[i]) * log(max(pmin,Φ(-xi)))
        end
        return -llh
    end
    # fit function using Optim.jl
    res = optimize(DifferentiableFunction(f),
                   [-Inf, 0.0], [Inf, Inf], [0.0, 3.0], Fminbox(LBFGS()))
    return Optim.minimizer(res)
end


# returns orthogonal matrix U with U[1,:] = u, and all other rows normalized
function orth_matrix(u::Vector{T}) where T <: Real
    n = length(u)
    unorm2 = dot(u, u)
    # first row
    U = Matrix{T}(undef, n, n)
    U[1,:] = u
    # following the modified Gram Schmidt procedure, assuming that
    # vk has single 1 at location i[k-1], where i[k] is the index of the
    # kth smallest item in u.
    i = sortperm(abs.(u))
    for k = 2:n
        # perform orthogonalisation with respect to unnormalized u1 separately
        ik = i[k-1]
        uk = -(u[ik] / unorm2) * u
        uk[ik] += 1
        for j = 2:(k-1)
            uj = view(U, j, :)      # normalized for all j >= 2
            uk -= dot(uk, uj) * uj
        end
        U[k,:] = uk / norm(uk)
    end
    U
end


# draws a sample from the multivariate Gaussian N(x | μ, Σ)
# while imposing constraint u^T x = z
function randConstMvn(μ, Σ, u, z)
    n = length(μ)
    # construct orthogonal matrix U such that U[1,:] = u
    U = orth_matrix(u)
    # use y = U x, such that y ~ N(U μ, U Σ U^T), and sample y[2:end] | y[1] = z
    μy = U * μ
    Σy = U * Σ * U'
    y = Vector{Float64}(undef, n)
    y[1] = z
    # find Λdd^-1, where Σy^-1 = [Λaa Λbb; Λcc Λdd], using matrix inv. lemma
    invΛdd = Σy[2:end, 2:end] .- (Σy[2:end, 1] * Σy[1, 2:end]') / Σy[1, 1]
    μd = μy[2:end] .+ Σy[2:end, 1] / Σy[1, 1] * (z .- μy[1])
    y[2:end] = rand(MvNormal(μd, invΛdd))
    # map back into x
    return U \ y
end


#NOTE TO SELF: z0 here needs to be re-scaled by σ2 / σμ2 already!

# simulates diffusion and returns (response, t, x)
# dx ~ N( c dt, I dt )     (c is a vector)
# and diffusion model bounds {-θ(t)-z0, θ(t)-z0} on u^T x, with both u and z0
# given as parameters. (θ(t)-z0)/||u|| is specified through b.
# The returned response is either 1 (upper) or 0 (lower bound)
function _simdiffusion(c, u, z0, b, δt, maxt)
    # sample (z, t) from 1D diffusion model, bounds at { θ-z0, -θ-z0 }
    μz = dot(u, c)
    σz = norm(u)
    t, resp = 0.0, true
    try
        t, resp = rand(sampler(ConstDrift(μz/σz, δt), b))
        t = min(t, maxt) # in case t > tmax
    catch y
        println("c = $(c), u = $(u)")
        println("μz = $(μz), σz = $(σz), θ = $(θ), z0 = $(z0)")
        println("μz/σz = $(μz/σz), (θ-z0)/σz = $((θ-z0)/σz)")
        flush(STDOUT)
        throw(y)
    end
    # get distance travelled z(t) = ±θ(t) - z0 at decision time t
    n = floor(Int, t / δt) + 1
    z = σz * (resp ?                       # as b[n] = (θ(t) - z0) / σz
              DiffModels.getubound(b, n) : # we get θ(t) - z0 by
              DiffModels.getlbound(b, n))  # b[n] * σz
    # find x(t) for given z(t)
    if length(c) == 1
        # single dimension, x is just scaled z = u x
        x = z ./ u
    else
        # draw x ~ N(μx(t), Σx(t)), subject to constraing u^T x(t) = z(t)
        μx = c * t
        Σx = Matrix(t * I, length(c), length(c))
        x = randConstMvn(μx, Σx, u, z)
    end
    # returns (response, t, x)
    return resp ? 1 : 0, t, x
end
# makes sure that ||u|| ≥ minwnorm * √(length(u)) and returns u, ||u||
function _lowerboundu(u)
    Dc = length(u)
    unorm2 = dot(u, u)
    if unorm2 < Dc * minwnorm^2
        return minwnorm * ones(Dc), minwnorm * √(Dc)
    else
        return u, √(unorm2)
    end
end
# simulate diffusion with bound vector θt in steps of δt, with θ(0) = θt[1]
function simdiffusionvec(c, u, z0::Real, θt::Vector, δt::Real, maxt::Real)
    z0 < θt[1]  || return 1, 0.0, zero(c)  # z0 above upper boundary
    z0 > -θt[1] || return 0, 0.0, zero(c)  # z0 below lower boundary
    u, σz = _lowerboundu(u)
    b = AsymBounds{VarBound, VarBound}(
        VarBound([(θi - z0)/σz for θi in θt], δt),
        VarBound([(θi + z0)/σz for θi in θt], δt))
    return _simdiffusion(c, u, z0, b, δt, maxt)
end
# simulate diffusion with constant bound at θ
function simdiffusionconst(c, u, z0::Real, θ::Real, maxt::Real)
    z0 < θ  || return 1, 0.0, zero(c)  # z0 above upper boundary
    z0 > -θ || return 0, 0.0, zero(c)  # z0 below lower boundary
    δt = 1e-5  # required for sampler, but not used for const. boundaries
    u, σz = _lowerboundu(u)
    b = ConstAsymBounds((θ-z0)/σz, -(θ+z0)/σz, δt)
    return _simdiffusion(c, u, z0, b, δt, maxt)
end
# simulate diffusion with linearly changing boundary θ + t * θslo
function simdiffusionlin(c, u, z0, θ::Real, θslo::Real, δt::Real, maxt::Real)
    z0 < θ  || return 1, 0.0, zero(c)  # z0 above upper boundary
    z0 > -θ || return 0, 0.0, zero(c)  # z0 below lower boundary
    u, σz = _lowerboundu(u)
    b = AsymBounds{LinearBound, LinearBound}(
        LinearBound((θ-z0)/σz, θslo/σz, δt, maxt),
        LinearBound((θ+z0)/σz, θslo/σz, δt, maxt))
    return _simdiffusion(c, u, z0, b, δt, maxt)
end

# simulates leaky competing accumulator model and returns (response, t, x)
# dxi = (ci - τinvi xi - winhi xj) dt + dWi, where i ∈ {1, 2}, j = 3-i,
# and bounds {θ1, θ2} on {x1, x2}. The returned response is either 0
# (x1 "wins") or 1 (x2 "wins").
function simlca(c, τinvi, winhi, θ1, θ2, δt, maxt)
    # pre-compute constants
    θ1n, θ2n, maxn = length(θ1), length(θ2), ceil(Int, maxt/δt)
    cδt = c .* δt
    τinviδt = 1 .- τinvi .* δt
    winhiδt = winhi .* δt
    sqrtδt = √(δt)

    # iterate in steps of size δt
    x = zeros(2)
    n = 0
    while n ≤ maxn
        x .= max.(0.0,
            τinviδt .* x .+ cδt .- winhiδt .* (@view x[end:-1:1]) .+
            sqrtδt .* randn(2))
        n += 1
        if x[1] ≥ θ1[min(θ1n, n)]
            # choose randomly if both races crossed the bound simultaneously
            return (x[2] ≥ θ2[min(θ2n, n)] && rand() < 0.5 ? 1 : 0), (n-1)*δt, x
        elseif x[2] ≥ θ2[min(θ2n, n)]
            return 1, (n-1)*δt, x
        end
    end
    return (x[2] > x[1] ? 1 : 0), (n-1)*δt, x
end
