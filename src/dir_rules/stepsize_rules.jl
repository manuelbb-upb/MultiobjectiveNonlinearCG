abstract type AbstractStepsizeRule end
abstract type AbstractStepsizeCache end

function init_stepsize_cache(::AbstractStepsizeRule, x, fx, DfxT, d, objf!, jacT!, meta)::AbstractStepsizeCache
    return nothing
end

function apply_stepsize!(dir, ::AbstractStepsizeCache, descent_cache, x, fx, DfxT, objf!, jacT!, meta)::Nothing
    return error("`apply_stepsize!` not implemented.")
end

Base.@kwdef struct FixedStepsizeRule{F<:Number} <: AbstractStepsizeRule
    stepsize :: F = MIN_PRECISION(1e-3)
end

struct FixedStepsizeCache{F}
    stepsize :: F
end

function init_stepsize_cache(sr::FixedStepsizeRule, x, fx, DfxT, d, objf!, jacT!, meta)
    FixedStepsizeCache(meta.precision(sr.stepsize))
end

function apply_stepsize!(dir, sc::FixedStepsizeCache, descent_cache, x, fx, DfxT, objf!, jacT!, meta)
    dir .*= sc.stepsize
    return nothing
end    

Base.@kwdef struct QuadApprox{E<:Real, M<:Real}
    eps :: E = MIN_PRECISION(1e-3)
    σ_fallback :: M = MIN_PRECISION(1)
end


Base.@kwdef struct QuadApproxCache{E<:Real, M<:Real, D}
    eps :: E = MIN_PRECISION(1e-3)
    σ_fallback :: M = MIN_PRECISION(1)
    σ_dat :: D
end

Base.@kwdef struct StandardArmijoRule{
    A<:Real, B<:Real, M<:Union{QuadApprox, Real}
} <: AbstractStepsizeRule
    a :: A = MIN_PRECISION(1e-4)
    b :: B = MIN_PRECISION(0.5)
    σ_init :: M = MIN_PRECISION(1)
end

function armijo_rhs(sc::AbstractStepsizeCache, dc::AbstractDirCache)
    error("`armijo_rhs` not defined for argument of type $(typeof(dc))")
end

struct StandardArmijoCache{A, B, M, X, Y} <: AbstractStepsizeCache
    a :: A
    b :: B
    σ_init :: M
    x :: X
    fx :: Y
end

init_init_cache(σ0::Real, T, x, DfxT) = T(σ0)
init_init_cache(σ0::QuadApprox, T, x, DfxT) = QuadApproxCache(σ0.eps, σ0.σ_fallback,(copy(x),copy(DfxT)))


function init_stepsize_cache(sr::StandardArmijoRule, x, fx, DfxT, d, objf!, jacT!, meta)
    T = meta.precision
    σ_init = init_init_cache(sr.σ_init, T, x, DfxT) 
    
    return StandardArmijoCache(T(sr.a), T(sr.b), σ_init, copy(x), copy(fx))
end

initial_stepsize(σ_init::Real, args...) = σ_init
function initial_stepsize(σ_init::QuadApproxCache, d, x, DfxT, jacT!)
    xε, DfxTε = σ_init.σ_dat
    ε = σ_init.eps
    xε .= x
    xε .+= ε .* d
    jacT!(DfxTε, xε)
    DfxTε .-= DfxT
    DfxTε ./= ε

    c1 = maximum(d'DfxT)
    c2 = maximum(d'DfxTε)

    if c2 > 0 && c1 <= 0
        return (-c1/c2)
    else
        return σ_init.σ_fallback
    end
end

function apply_stepsize!(dir, sc::StandardArmijoCache, descent_cache, x, fx, DfxT, objf!, jacT!, meta)
    x_ = sc.x
    fx_ = sc.fx

    Φx = maximum(fx)
    
    σ = initial_stepsize(sc.σ_init, dir, x, DfxT, jacT!)
    dir .*= σ
    
    x_ .= x .+ dir
    objf!(fx_, x_)
    Φx_ = maximum(fx_)

    a = sc.a
    b = sc.b
    ω = armijo_rhs(sc, descent_cache)
    ε = eps(σ * b^15)
    #while Φx - Φx_ < a * σ * ω && σ > ε
    while any(fx .- fx_ .< a * σ * ω) && σ > ε
        σ *= b
        dir .*= b
        x_ .= x .+ dir
        objf!(fx_, x_)
        Φx_ = maximum(fx_)
    end
    return nothing
end

Base.@kwdef struct ModifiedArmijoRule{A<:Real, B<:Real, M} <: AbstractStepsizeRule
    a :: A = MIN_PRECISION(1e-4)
    b :: B = MIN_PRECISION(0.5)
    σ_init :: M = MIN_PRECISION(1)
end

struct ModifiedArmijoCache{A, B, M, X, Y} <: AbstractStepsizeCache
    a :: A
    b :: B
    σ_init :: M
    x :: X
    fx :: Y
end

function init_stepsize_cache(sr::ModifiedArmijoRule, x, fx, DfxT, d, objf!, jacT!, meta)
    T = meta.precision
    σ_init = init_init_cache(sr.σ_init, T, x, DfxT) 
    return ModifiedArmijoCache(T(sr.a), T(sr.b), σ_init, copy(x), copy(fx))
end

function apply_stepsize!(dir, sc::ModifiedArmijoCache, descent_cache, x, fx, DfxT, objf!, jacT!, meta)
    x_ = sc.x
    fx_ = sc.fx

    Φx = maximum(fx)
    
    σ = initial_stepsize(sc.σ_init, dir, x, DfxT, jacT!)
    dir .*= σ
    
    x_ .= x .+ dir
    objf!(fx_, x_)
    Φx_ = maximum(fx_)

    a = sc.a
    b = sc.b
    ω = armijo_rhs(sc, descent_cache)
    ε = eps(σ * b^15)
    #while Φx - Φx_ < a * σ^2 * ω && σ > ε
    while any(fx .- fx_ .< a * σ^2 * ω) && σ > ε
        σ *= b
        dir .*= b
        x_ .= x .+ dir
        objf!(fx_, x_)
        Φx_ = maximum(fx_)
    end
    return nothing
end