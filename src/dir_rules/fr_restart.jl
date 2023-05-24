# Fletcher-Reeves

@with_kw struct FRRestart{SR} <: AbstractDirRule
    stepsize_rule :: SR = ModifiedArmijoRule()
    criticality_measure :: Symbol = :cg
    
    @assert criticality_measure == :cg || criticality_measure == :sd
end

struct FRRestartCache{N, P, D, S, FW} <: AbstractDirCache
    sd_norm_squared :: N            # ‖δₖ‖² = -φₖ(δₖ)
    # criticality :: C                # ‖δₖ‖² or -φₖ(dₖ)
    phi :: P                        # -φₖ(dₖ)
    d_prev :: D                     # dₖ₋₁
    criticality_measure :: Symbol
    stepsize_cache :: S
    fw_cache :: FW
end

function armijo_rhs(sc::StandardArmijoCache, dc::FRRestartCache)
    return dc.phi[]
end

function armijo_rhs(sc::ModifiedArmijoCache, dc::FRRestartCache)
    return sum(dc.d_prev.^2)
end

function init_cache(descent_rule::FRRestart, x, fx, DfxT, d, objf!, jacT!, meta)
    T = meta.precision
    stepsize_cache = init_stepsize_cache(descent_rule.stepsize_rule, x, fx, DfxT, d, objf!, jacT!, meta)
    ## cache for convex optimizer
    fw_cache = init_frank_wolfe_cache(T, meta.dim_in, meta.dim_out)
    return FRRestartCache(
        Ref(zero(T)),
        Ref(zero(T)),
        copy(d),
        descent_rule.criticality_measure,
        stepsize_cache,
        fw_cache,
    )
end

function first_step!(dc::FRRestartCache, d, x, fx, DfxT, objf!, jacT!, meta)
    # make `d` correspond to steepest descent direction and 
    # set `dc.sd_norm_squared` to conform to \|δ\|^2 = -φ(δ)
    set_d_and_norm!(dc, d, DfxT)
    # order of operations is important here!
    dc.d_prev .= d
    dc.phi[] = dc.sd_norm_squared[]
    apply_stepsize!(d, dc.stepsize_cache, dc, x, fx, DfxT, objf!, jacT!, meta)

    return nothing
end

function criticality(descent_cache::FRRestartCache)
    if descent_cache.criticality_measure == :sd
        return descent_cache.sd_norm_squared[]
    else
        return descent_cache.phi[]
    end
end

function step!(dc::FRRestartCache, d, x, fx, DfxT, objf!, jacT!, meta)
    ω = dc.sd_norm_squared[]
    iszero(ω) && return nothing # ω is a denominator, so better do nothing before deviding by 0...
    
    T = meta.precision

    # make `d` correspond to steepest descent direction δ and 
    # set `dc.sd_norm_squared` to conform to \|δ\|^2 = -φ(δ)
    set_d_and_norm!(dc, d, DfxT)
    
    # unpack temporary arrays
    d_ = dc.d_prev
    
    θ = maximum(d_'DfxT)
    θ += dc.phi[]
    θ /= ω

    # β = = φₖ(δₖ) / φₖ₋₁ (δₖ₋₁)
    β = dc.sd_norm_squared[] / ω

    if θ < 0
        θ = 1
        β = 0
    end

    d .*= θ
    d .+= β .* d_

    # store for next iteration
    dc.phi[] = abs(maximum(d'DfxT))
    d_ .= d

    # scale `d` with stepsize
    apply_stepsize!(d, dc.stepsize_cache, dc, x, fx, DfxT, objf!, jacT!, meta)

    return nothing
end