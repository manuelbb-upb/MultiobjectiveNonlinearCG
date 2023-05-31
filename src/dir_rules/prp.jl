# Polak-Ribière-Polyak directions

"""
    PRP(stepsize_rule, criticality_measure=:cg)

A `AbstractDirRule` defining modified Polak-Ribière-Polyak directions with guaranteed
descent.

* `stepsize_rule` is a compatible `AbstractDirRule` that dictates how stepsizes are chosen.
* `criticality_measure` is a symbol, either `:cg` or `:sd`, defining which measure is used
  for stopping criteria relying on criticality. `:sd` stands for "steepest descent" and uses
  the standard measure, `:cg` uses a measure inherent to the "conjugate-gradient" direction.
"""
@with_kw struct PRP{SR} <: AbstractDirRule
    stepsize_rule :: SR = ModifiedArmijoRule()
    criticality_measure :: Symbol = :cg

    @assert criticality_measure == :cg || criticality_measure == :sd
end

struct PRPCache{N, C, S, D, FW} <: AbstractDirCache
    sd_norm_squared :: N    # Ref to the squared norm of the steepest descent direction
    criticality :: C        # Ref to an appropriate criticality measure
    stepsize_cache :: S     # cache for the stepsize calculation
    d_prev :: D             # previous direction vector
    dj :: D                 # temporary direction vector
    dtmp :: D               # temporary direction vector
    y :: D                  # working array for δₖ₋₁ or δₖ₋₁ - δₖ
    criticality_measure :: Symbol
    fw_cache :: FW
    phi :: N
end

function armijo_rhs(sc::StandardArmijoCache, dc::PRPCache)
    return dc.phi[]
end

function armijo_rhs(sc::ModifiedArmijoCache, dc::PRPCache)
    return sum(dc.d_prev.^2)
end

function init_cache(descent_rule::PRP, x, fx, DfxT, d, objf!, jacT!, objf_and_jacT!, meta)
    T = meta.precision
    stepsize_cache = init_stepsize_cache(descent_rule.stepsize_rule, x, fx, DfxT, d, objf!, jacT!, meta)
    ## cache for convex optimizer
    fw_cache = init_frank_wolfe_cache(T, meta.dim_in, meta.dim_out)
    return PRPCache(
        Ref(typemax(T)),
        Ref(typemax(T)),
        stepsize_cache,
        copy(d),
        copy(d),
        copy(d),
        copy(d), 
        descent_rule.criticality_measure,
        fw_cache,
        Ref(zero(T))
    )
end

function first_step!(dc::PRPCache, d, x, fx, DfxT, objf!, jacT!, meta)
    # make `d` correspond to steepest descent direction and 
    # set `dc.sd_norm_squared` to conform to \|δ\|^2 = -φ(δ)
    set_d_and_norm!(dc, d, DfxT)
    # order of operations is very important here!
    # 1) set `y` to store *unscaled* steepest descent direction for next iteration
    dc.y .= d
    # 2) set criticality value -- in the first iteration the value of 
    #    `dc.criticality_measure` doesnt matter
    dc.criticality[] = dc.sd_norm_squared[]
    # 3) store unscaled direction for computations in next iteration
    dc.d_prev .= d
    apply_stepsize!(d, dc.stepsize_cache, dc, x, fx, DfxT, objf!, jacT!, meta)

    return nothing
end

criticality(descent_cache::PRPCache) = descent_cache.criticality[]

function step!(dc::PRPCache, d, x, fx, DfxT, objf!, jacT!, meta)
    ω = dc.sd_norm_squared[]
    iszero(ω) && return nothing # ω is a denominator, so better do nothing before deviding by 0...
    
    T = meta.precision

    # make `d` correspond to steepest descent direction δ and 
    # set `dc.sd_norm_squared` to conform to \|δ\|^2 = -φ(δ)
    set_d_and_norm!(dc, d, DfxT)
    
    y = dc.y    # should store δₖ₋₁ atm
    y .-= d     # now its δₖ₋₁ - δₖ

    # unpack temporary arrays
    d_ = dc.d_prev
    dj = dc.dj
    dtmp = dc.dtmp

    K = meta.dim_out

    #=
    ψ(ℓ,j) = let gℓ=DfxT[:,ℓ], gj=DfxT[:, j];
        β = gj'y / ω
        θ = gj'd_ / ω
        dd = @. d + β * d_ - θ * y
        gℓ'dd
    end
    Φ = zeros(K,K)
    for ℓ=1:K
        for j=1:K
            Φ[ℓ, j] = ψ(ℓ, j)
        end
    end
    display(Φ)
    @show maximum(minimum(Φ, dims=2))
    @show minimum(maximum(Φ, dims=1))
    =#
    φmax = typemin(T)
    for j=1:K
        gj = DfxT[:, j]
        β = gj'y / ω
        θ = gj'd_ / ω
        @. dj = d + β * d_ - θ * y
        # @. dj = β * d_ - θ * y

        φmin = typemax(T)
        for gℓ=eachcol(DfxT)
            _φ = gℓ'dj
            if _φ <= φmin
                φmin = _φ
            end
        end

        if φmax <= φmin
            φmax = φmin
            dtmp .= dj
        end
    end
    y .= d     # set `y` to current steepest descent direction for next iteration
    d .= dtmp  # set `d` to conjugate gradient direction
    # dc.phi[] = abs(maximum(d'DfxT))
    dc.phi[] = abs(φmax)

    if dc.criticality_measure == :sd
        dc.criticality[] = dc.sd_norm_squared[]
    else
        dc.criticality[] = dc.phi[]
    end    

    d_ .= d     # store for next iteration, **before scaling**

    # scale `d` with stepsize
    apply_stepsize!(d, dc.stepsize_cache, dc, x, fx, DfxT, objf!, jacT!, meta)

    return nothing
end