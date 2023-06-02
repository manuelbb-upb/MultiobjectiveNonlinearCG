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
    dl :: D                 # temporary direction vector
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
    dl = dc.dl

    K = meta.dim_out

    #= 
    As per our notes, we want to solve
    ```math
    \max_l \min_j ⟨ gₗ, δₖ + β(j) dₖ₋₁ + θ(j) yₖ ⟩,
    β(j) = ⟨ gⱼ, yₖ ⟩, θ(j) = ⟨ gⱼ, dₖ₋₁ ⟩ 
    ```
    Look at the function
    ```math
    \begin{algined}
        Ψ(l, j)     &= ⟨ gₗ, δₖ + β(j) dₖ₋₁ - θ(j) yₖ ⟩             \\
                    &= ⟨gₗ,δₖ⟩ + ⟨gⱼ,yₖ⟩⟨gₗ,dₖ₋₁⟩ - ⟨gⱼ,dₖ₋₁⟩⟨gₗ,yₖ⟩        \\
                    &= ⟨gₗ,δₖ⟩ + ⟨gⱼ,yₖ⟩⟨gₗ,dₖ₋₁⟩ - ⟨gⱼ,dₖ₋₁⟩⟨gₗ,yₖ⟩        \\
                    &= ⟨gₗ,δₖ⟩ + ⟨gⱼ,yₖ⟩θ(l) - ⟨gⱼ,dₖ₋₁⟩β(l)            \\
                    &= ⟨gₗ,δₖ⟩ - ( -⟨gⱼ,yₖ⟩θ(l) + ⟨gⱼ,dₖ₋₁⟩β(l) + ⟨gⱼ,δₖ⟩ ) + ⟨gⱼ,δₖ⟩   \\
                    &= ⟨gₗ,δₖ⟩ - Ψ(j, l) + ⟨gⱼ,δₖ⟩                   \\
                    &= -Ψ(j, l) + ⟨gₗ + gⱼ, δₖ⟩                     \\
                    &= -Ψ(j,l) + r(l,j).
    \end{aligned}
    ```
    ``r(l,j)`` is symmetric and ``r(l,j) = r(j,l) < 0`` for all ``j`` and ``l``.
    Hence,
    ```math
    \max_l \min_j Ψ(l,j)    = \max_l \min_j -Ψ(j,l) + r(j,l)
                            = \max_l -\max_j Ψ(j,l) - r(j,l)
                            = -\min_l \max_j Ψ(j,l) - r(j,l)
    ```
    Any of the expressions on the RHS can be used to determine optimal ``(l,j)``.
    Essentially, it boils down to a memory-computation tradeoff:
    In the LHS, for each gradient ``gₗ`` we would have to test potenntial directions ``dⱼ``.
    So it makes sense to precompute all ``dⱼ`` an store them in a matrix with ``nk`` entries.
    In the RHS, we fix the direction ``dₗ`` and test against all gradients, which are stored, 
    anyways.
    
    We can save even more computations by taking one of the earlier equalities:
    ```math
    \max_l \min_j Ψ(l,j) = \max_l \min_j ⟨gₗ,δₖ⟩ + ⟨gⱼ,yₖ⟩θ(l) - ⟨gⱼ,dₖ₋₁⟩β(l)
                         = -\min_l \max_j ⟨gⱼ,dₖ₋₁⟩β(l) - ⟨gⱼ,yₖ⟩θ(l) - ⟨gₗ,δₖ⟩
                         = -\min_l -⟨gₗ,δₖ⟩ + \max_j ⟨gⱼ, β(l)dₖ₋₁ - θ(l)yₖ⟩.
    ```
    And thats what we do below.
    We only have to keep in mind, that the optimal MiniMax solution of the RHS 
    ``δ + β(l)dₖ₋₁ - θ(l)yₖ`` is not the optimal MaxiMin solution of the LHS,
    so we also have store the optimal `j_opt` whenever the MiniMax changes 
    and (re-)compute β(j) and θ(j) afterwards.
    =#
    minl = typemax(T)
    j_opt = 0
    for l=1:K
        # We want to first fix ``l`` in and compute the maximum with respect to ``j``.
        # Fixing ``l`` requires calculating a preliminary direction `dl`:
        
        ## unpack gradient and compute coefficients
        gl = DfxT[:, l]
        β = gl'y
        θ = gl'd_
        ## set direction to correspond to ``β(l)dₖ₋₁ + θ(l)yₖ``
        @. dl = β * d_ - θ * y

        ## we compute the maximum in simple for loop
        maxj, j = findmax(gj'dl/ω for gj=eachcol(DfxT))
        ## adjust for offset ``-⟨gₗ,δₖ⟩`` here
        maxj -= gl'd

        if maxj <= minl
            minl = maxj
            ## minimax has changed, set coefficients for final direction
            j_opt = j
        end
    end
    
    # compute CG direction from optimal coefficients:
    gj = DfxT[:, j_opt]
    β = gj'y / ω
    θ = gj'd_ / ω
    ## use `dl` as a temporary array
    dl .= β*d_ - θ*y

    y .= d     # set `y` to current steepest descent direction for next iteration
    # now, finally set CG direction 
    d .+= dl

    #dc.phi[] = abs(maximum(d'DfxT))
    dc.phi[] = minl

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