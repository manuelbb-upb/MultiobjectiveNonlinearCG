
Base.@kwdef struct FixedStepsize{F<:Real} <: AbstractStepsizeRule
    sz :: F = 0.001
end
struct FixedStepsizeCache{F<:AbstractFloat} <: AbstractStepsizeCache
    sz :: F
end
function init_cache(sz_rule::FixedStepsize, mop::AbstractMOP)
    return FixedStepsizeCache(convert(float_type(mop), sz_rule.sz))
end

function stepsize!(d, xd, fxd, sz_cache::FixedStepsizeCache, mop::AbstractMOP, x, fx, Dfx, critval; kwargs...)
    @unpack sz = sz_cache
    d .*= sz
    @. xd = x + d
    return objectives!(fxd, mop, xd)
end

Base.@kwdef struct StandardArmijoBacktracking{F<:Real} <: AbstractStepsizeRule
    factor :: F = 0.5
    constant :: F = 1e-3
    sz0 :: F = 1.0
    mode :: Union{Val{:all}, Val{:any}, Val{:max}} = Val(:any)
end

struct StandardArmijoBacktrackingCache{F<:AbstractFloat} <: AbstractStepsizeCache
    factor :: F
    constant :: F
    sz0 :: F
    mode :: Union{Val{:all}, Val{:any}, Val{:max}}
    lhs_vec :: Vector{F}
end

function init_cache(sz_rule::StandardArmijoBacktracking, mop::AbstractMOP)
    F = float_type(mop)
    @unpack factor, constant, sz0, mode = sz_rule
    return StandardArmijoBacktrackingCache{F}(
        factor, constant, sz0, mode, zeros(F, dim_out(mop))
    )
end

function stepsize!(
    d, xd, fxd, sz_cache::StandardArmijoBacktrackingCache, mop::AbstractMOP, x, fx, Dfx, critval;
    kwargs...
)
    @unpack factor, constant, sz0, mode, lhs_vec = sz_cache
    return armijo_backtrack!(
        d, xd, fxd, mop, x, fx, critval, factor, constant, sz0, mode, lhs_vec; kwargs...
    )
end

function armijo_backtrack!(
    d, xd, fxd,
    mop,
    x, fx, critval,
    factor, constant, sz0, mode, lhs_vec;
    x_tol_abs = 0,
    x_tol_rel = 0,
    fx_tol_abs = 0,
    fx_tol_rel = 0,
    kwargs...
)
    x_tol_abs = max(x_tol_abs, mapreduce(eps, min, d))

    zero_step = false
    stop_code = nothing
    if critval <= 0
       zero_step = true
    end

    if !zero_step
        sz = sz0
        rhs = critval * factor * sz

        d .*= sz
        d_norm = LA.norm(d, Inf)

        @. xd = x + d
        stop_code = objectives!(fxd, mop, xd)
        if stop_code isa STOP_CODE
            zero_step = true
        end
    end

    x_norm = LA.norm(x, Inf)
    fx_norm = LA.norm(fx, Inf)

    _fx = similar(fx)
    objectives!(_fx, mop, x)
    @show fx
    @show fx .- _fx

    if !zero_step
        ## avoid re-computation of maximum for scalar test:
        φx = _armijo_φ(mode, fx)
        while true
            if sz <= 0
                zero_step = true
                break
            end

            if d_norm <= x_tol_abs
                break
            end

            if d_norm <= x_tol_rel * x_norm
                break
            end
            
            @. lhs_vec = fx - fxd
            
            # lhs_norm = LA.norm(lhs_vec, Inf)
            # if lhs_norm <= fx_tol_abs
                # break
            # end
            # if lhs_norm <= fx_tol_rel * fx_norm
                # break
            # end 
           
            if _armijo_test(mode, lhs_vec, φx, fxd, rhs)
                break
            end
            
            sz *= factor
            rhs *= factor   # ! important
            d *= factor
            d_norm *= factor
            @. xd = x + d
            stop_code = objectives!(fxd, mop, xd)
            if stop_code isa STOP_CODE
                zero_step = true
                break
            end
        end
    end
    @show sz
    if zero_step
        d .= 0
        xd .= x
        fxd .= fx
    end

    return stop_code 
end

_armijo_φ(::Val{:all}, fx)=fx
_armijo_φ(::Val{:any}, fx)=fx
_armijo_φ(::Val{:max}, fx)=maximum(fx)

function _armijo_test(::Val{:all}, lhs_vec, φx, fxd, rhs)
    return all( lhs_vec .>= rhs )
end
function _armijo_test(::Val{:any}, lhs_vec, φx, fxd, rhs)
    return maximum(lhs_vec) >= rhs
end
function _armijo_lhs(::Val{:any}, lhs_vec, φx, fxd, rhs)
    return φx - maximum(fxd) >= rhs
end

Base.@kwdef struct SteepestDescentDirection{
    sz_ruleType, 
} <: AbstractStepRule
    sz_rule :: sz_ruleType = FixedStepsize()
end

struct SteepestDescentDirectionCache{
    F<:AbstractFloat,
    sz_cacheType,
    fw_cacheType
} <: AbstractStepRuleCache
    criticality_ref :: Base.RefValue{F}
    sz_cache :: sz_cacheType
    fw_cache :: fw_cacheType
end

function criticality(carrays, step_cache::SteepestDescentDirectionCache)
    return step_cache.criticality_ref[]
end

include("multidir_frank_wolfe.jl")
function init_cache(step_rule::SteepestDescentDirection, mop::AbstractMOP)
    criticality_ref = Ref(convert(float_type(mop), Inf))
    sz_cache = init_cache(step_rule.sz_rule, mop)
    fw_cache = init_frank_wolfe_cache(float_type(mop), dim_out(mop))
    return SteepestDescentDirectionCache(criticality_ref, sz_cache, fw_cache)
end

function step!(
    it_index, carrays, step_cache::SteepestDescentDirectionCache, mop::AbstractMOP; 
    kwargs...
)
    ## compute KKT multipliers for steepest descent direction
    @unpack Dfx = carrays
    @show Dfx
    @unpack fw_cache = step_cache
    α = frank_wolfe_multidir_dual!(fw_cache, Dfx)
    
    ## use these to set steepest descent direction `d`
    @unpack d = carrays
    LA.mul!(d, Dfx', α)
    @show d
    @show Dfx*d

    ## before scaling `d`, set criticality
    @unpack criticality_ref = step_cache
    critval = criticality_ref[] = LA.norm(d, Inf)

    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack x, fx, xd, fxd = carrays
    @unpack sz_cache = step_cache

    return stepsize!(d, xd, fxd, sz_cache, mop, x, fx, Dfx, critval; kwargs...)
end
