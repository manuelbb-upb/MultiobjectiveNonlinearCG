# Polak-Ribière-Polyak directions

@with_kw struct PRP{SR} <: AbstractDirRule
    stepsize_rule :: SR
    criticality_measure :: Symbol = :cg

    @assert criticality_measure == :cg || criticality_measure == :sd
end

struct PRPCache{N, C, S, D} <: AbstractDirCache
    sd_norm_squared :: N
    criticality :: C
    stepsize_cache :: S
    d_prev :: D
    dj :: D
    dtmp :: D
    y :: D
    criticality_measure :: Symbol
end

function init_cache(descent_rule::PRP, x, fx, DfxT, d, objf!, jacT!, meta)
    T = meta.precision
    stepsize_cache = init_stepsize_cache(descent_rule.stepsize_rule, x, fx, DfxT, d, objf!, jacT!, meta)
    return PRPCache(
        Ref(zero(T)),
        Ref(zero(T)),
        stepsize_cache,
        copy(d),
        copy(d),
        copy(d),
        copy(d), 
        descent_rule.criticality_measure
    )
end

function first_step!(dc::PRPCache, d, x, fx, DfxT, objf!, jacT!, meta)
    # make `d` correspond to steepest descent direction and 
    # set `dc.criticality` to conform to \|δ\|^2 = -φ(δ)
    set_steepest_descent!(dc, d, x, fx, DfxT, objf!, jacT!, meta)
    # order of operations is very important here!
    dc.y .= d   # set `y` to store *unscaled* steepest descent direction for next iteration
    dc.criticality[] = dc.sd_norm_squared[]
    apply_stepsize!(d, dc.stepsize_cache, dc, x, fx, DfxT, objf!, jacT!, meta)
    dc.d_prev .= d  # store scaled direction for reference in next iteration

    return nothing
end

criticality(descent_cache::PRPCache) = descent_cache.criticality[]

function step!(dc::PRPCache, d, x, fx, DfxT, objf!, jacT!, meta)
    ω = dc.sd_norm_squared[]
    iszero(ω) && return nothing
    
    T = meta.precision

    set_steepest_descent!(dc, d, x, fx, DfxT, objf!, jacT!, meta)
    
    y = dc.y    # should store δₖ₋₁ atm
    y .-= d     # now its δₖ₋₁ - δₖ

    d_ = dc.d_prev
    dj = dc.dj
    dtmp = dc.dtmp

    K = meta.dim_out

    φmin = typemax(T)
    for j=1:K
        gj = DfxT[:, j]
        β = gj'y / ω
        θ = gj'd_ / ω
        @. dj = d + β * d_ - θ * y

        φmax = typemin(T)
        for gℓ=eachcol(DfxT)
            _φ = gℓ'dj
            if _φ >= φmax
                φmax = _φ
            end
        end

        if φmax <= φmin
            φmin = φmax
            dtmp .= dj
        end
    end

    y .= d      # set `y` to current steepest descent direction for next iteration
    d .= dtmp   # set `d` to conjugate gradient direction

    if dc.criticality_measure == :sd
        dc.criticality[] = dc.sd_norm_squared[]
    else
        dc.criticality[] = -φmin
    end

    # scale `d`:
    apply_stepsize!(d, dc.stepsize_cache, dc, x, fx, DfxT, objf!, jacT!, meta)

    d_ .= d     # store for next iteration

    return nothing
end