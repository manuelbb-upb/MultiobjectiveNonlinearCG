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

Base.@kwdef struct StandardArmijoRule{A<:Real, B<:Real, M<:Real} <: AbstractStepsizeRule
    a :: A = MIN_PRECISION(1e-4)
    b :: B = MIN_PRECISION(0.5)
    σ_init :: M = MIN_PRECISION(1)
end

struct StandardArmijoCache{A, B, M, X, Y} <: AbstractStepsizeCache
    a :: A
    b :: B
    σ_init :: M
    x :: X
    fx :: Y
end

function init_stepsize_cache(sr::StandardArmijoRule, x, fx, DfxT, d, objf!, jacT!, meta)
    T = meta.precision
    return StandardArmijoCache(T(sr.a), T(sr.b), T(sr.σ_init), copy(x), copy(fx))
end

function apply_stepsize!(dir, sc::StandardArmijoCache, descent_cache, x, fx, DfxT, objf!, jacT!, meta)
    x_ = sc.x
    fx_ = sc.fx

    Φx = maximum(fx)
    
    σ = sc.σ_init
    dir .*= σ
    
    x_ .= x .+ dir
    objf!(fx_, x_)
    Φx_ = maximum(fx_)

    a = sc.a
    b = sc.b
    ω = criticality(descent_cache)
    ε = eps(σ * b^15)
    while Φx - Φx_ < a * σ * ω && σ > ε
        σ *= b
        dir .*= b
        x_ .= x .+ dir
        objf!(fx_, x_)
        Φx_ = maximum(fx_)
    end
    return nothing
end

Base.@kwdef struct ModifiedArmijoRule{A<:Real, B<:Real, M<:Real} <: AbstractStepsizeRule
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
    return ModifiedArmijoCache(T(sr.a), T(sr.b), T(sr.σ_init), copy(x), copy(fx))
end

function apply_stepsize!(dir, sc::ModifiedArmijoCache, descent_cache, x, fx, DfxT, objf!, jacT!, meta)
    x_ = sc.x
    fx_ = sc.fx

    Φx = maximum(fx)
    
    σ = sc.σ_init
    dir .*= σ
    
    x_ .= x .+ dir
    objf!(fx_, x_)
    Φx_ = maximum(fx_)

    a = sc.a
    b = sc.b
    ω = sum( dir.^2 )
    ε = eps(σ * b^15)
    while Φx - Φx_ < a * σ^2 * ω && σ > ε
        σ *= b
        dir .*= b
        x_ .= x .+ dir
        objf!(fx_, x_)
        Φx_ = maximum(fx_)
    end
    return nothing
end