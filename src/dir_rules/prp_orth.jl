# Polak-Ribière-Polyak directions
@with_kw struct PRPGradProjection{SR} <: AbstractDirRule
    stepsize_rule :: SR = ModifiedArmijoRule()
    criticality_measure :: Symbol = :cg

    @assert criticality_measure == :cg || criticality_measure == :sd
end

struct PRPGPCache{N, P, D, O, S, FW} <: AbstractDirCache
    sd_norm_squared :: N            # ‖δₖ‖² = -φₖ(δₖ)
    phi :: P                        # -φₖ(dₖ)
    d_prev :: D                     # dₖ₋₁
    dj :: D                         # projection offset
    d_tmp :: D                      # tmp array
    y :: D                          # δₖ₋₁ - δₖ
    P :: O                          # orthogonal projector matrix
    criticality_measure :: Symbol
    stepsize_cache :: S
    fw_cache :: FW
end

function armijo_rhs(sc::StandardArmijoCache, dc::PRPGPCache)
    return dc.phi[]
end

function armijo_rhs(sc::ModifiedArmijoCache, dc::PRPGPCache)
    return sum(dc.d_prev.^2)
end

function init_cache(descent_rule::PRPGradProjection, x, fx, DfxT, d, objf!, jacT!, objf_and_jacT!, meta)
    T = meta.precision
    N = meta.dim_in
    stepsize_cache = init_stepsize_cache(descent_rule.stepsize_rule, x, fx, DfxT, d, objf!, jacT!, meta)
    ## cache for convex optimizer
    fw_cache = init_frank_wolfe_cache(T, meta.dim_in, meta.dim_out)
    return PRPGPCache(
        Ref(typemax(T)),
        Ref(typemax(T)),
        copy(d),
        copy(d),
        copy(d),
        copy(d),
        zeros(T, N, N),
        descent_rule.criticality_measure,
        stepsize_cache,
        fw_cache,
    )
end

function first_step!(dc::PRPGPCache, d, x, fx, DfxT, objf!, jacT!, meta)
    # make `d` correspond to steepest descent direction and 
    # set `dc.sd_norm_squared` to conform to \|δ\|^2 = -φ(δ)
    set_d_and_norm!(dc, d, DfxT)
    # store unscaled direction for computations in next iteration
    dc.phi[] = dc.sd_norm_squared[]
    dc.d_prev .= d
    dc.y .= d
    apply_stepsize!(d, dc.stepsize_cache, dc, x, fx, DfxT, objf!, jacT!, meta)

    return nothing
end

function criticality(descent_cache::PRPGPCache)
    if descent_cache.criticality_measure == :sd
        return descent_cache.sd_norm_squared[]
    else
        return descent_cache.phi[]
    end
end

function step!(dc::PRPGPCache, d, x, fx, DfxT, objf!, jacT!, meta)
    @show ω = dc.sd_norm_squared[]
    iszero(ω) && return nothing # ω is a denominator, so better do nothing before deviding by 0...
    
    # make `d` correspond to steepest descent direction δ and 
    # set `dc.sd_norm_squared` to conform to \|δ\|^2 = -φ(δ)
    set_d_and_norm!(dc, d, DfxT)
    
    # unpack temporary arrays
    d_ = dc.d_prev
    dj = dc.dj
    d_tmp = dc.d_tmp
    P = dc.P
    y = dc.y        # should hold δₖ₋₁ atm
    y .-= d         # now its δₖ₋₁ - δₖ
    
    β = maximum(y'DfxT) / ω # φₖ(yₖ) / -φₖ(δₖ)

    d_ .*= β
    φ = maximum(d_'DfxT)
    if φ <= 0
        # do nothing, _d is already a non-ascent direction
        y .= d
        d .+= d_
    else
        K = meta.dim_out
        T = meta.precision
        min_val_j = typemax(T)
        for j=1:K
            gj = DfxT[:, j]
            LA.mul!(P, gj, gj')
            P ./= sum(gj .^ 2)
            dj .= d_
            dj .-= P*dj
            max_val_ℓ = typemin(T)
            for ℓ=1:K
                gℓ = DfxT[:, ℓ]
                val_ℓ = gℓ'dj
                if val_ℓ >= max_val_ℓ
                    max_val_ℓ = val_ℓ
                end
            end
            if max_val_ℓ <= min_val_j
                min_val_j = max_val_ℓ
                d_tmp .= dj
            end
        end
    
        if min_val_j <= 0
            y .= d
            d .+= d_tmp
        end
    end

    # store for next iteration
    dc.phi[] = abs(maximum(d'DfxT))
    d_ .= d

    # scale `d` with stepsize
    apply_stepsize!(d, dc.stepsize_cache, dc, x, fx, DfxT, objf!, jacT!, meta)

    return nothing
end