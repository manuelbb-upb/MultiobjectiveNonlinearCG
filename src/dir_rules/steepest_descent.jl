
Base.@kwdef struct SteepestDescentRule{SR<:AbstractStepsizeRule} <: AbstractDirRule
    stepsize_rule :: SR
end

struct SteepestDescentCache{C, S, FW} <: AbstractDirCache 
    sd_norm_squared :: C
    stepsize_cache :: S
    fw_cache :: FW
end

function armijo_rhs(sc::StandardArmijoCache, dc::SteepestDescentCache)
    return dc.sd_norm_squared[]
end

function armijo_rhs(sc::ModifiedArmijoCache, dc::SteepestDescentCache)
    return dc.sd_norm_squared[]
end

function init_cache(descent_rule::SteepestDescentRule, x, fx, DfxT, d, objf!, jacT!, meta)
    T = meta.precision
    stepsize_cache = init_stepsize_cache(descent_rule.stepsize_rule, x, fx, DfxT, d, objf!, jacT!, meta)
    ## cache for convex optimizer
    fw_cache = init_frank_wolfe_cache(T, meta.dim_in, meta.dim_out)
    return SteepestDescentCache(Ref(zero(T)), stepsize_cache, fw_cache)
end

function set_steepest_descent!(fw_cache, DfxT)
    return frank_wolfe_multidir_dual!(fw_cache, eachcol(DfxT))
end

# helper
function set_d_and_norm!(descent_cache, d, DfxT)
    # compute and set steepest descent direction
    fw_cache = descent_cache.fw_cache
    set_steepest_descent!(fw_cache, DfxT)
    d .= fw_cache.sol

    # compute and set the criticality value
    ω = sum(d.^2)
    descent_cache.sd_norm_squared[] = ω
    return nothing
end

function first_step!(descent_cache::SteepestDescentCache, d, x, fx, DfxT, objf!, jacT!, meta)
    set_d_and_norm!(descent_cache, d, DfxT)
    
    apply_stepsize!(d, descent_cache.stepsize_cache, descent_cache, x, fx, DfxT, objf!, jacT!, meta)
    return nothing
end

function criticality(descent_cache::SteepestDescentCache) 
    return descent_cache.sd_norm_squared[]
end