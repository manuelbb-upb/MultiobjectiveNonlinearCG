include("multidir_frank_wolfe.jl")

Base.@kwdef struct FixedStepsize{F<:Real} <: AbstractStepsizeRule
    sz :: F = 0.001
end
struct FixedStepsizeCache{F<:AbstractFloat} <: AbstractStepsizeCache
    sz :: F
end
function init_cache(sz_rule::FixedStepsize, mop::AbstractMOP)
    return FixedStepsizeCache(convert(float_type(mop), sz_rule.sz))
end

function stepsize!(carrays, sz_cache::FixedStepsizeCache, mop::AbstractMOP, critval; kwargs...)
    @unpack sz = sz_cache
    @unpack d, x, xd, fxd = carrays
    d .*= sz
    @. xd = x + d
    return objectives!(fxd, mop, xd)
end

Base.@kwdef struct ArmijoBacktracking{F<:Real} <: AbstractStepsizeRule
    factor :: F = 0.5
    constant :: F = 1e-4
    sz0 :: F = 1.0
    mode :: Union{Val{:all}, Val{:any}, Val{:max}} = Val(:all)
    is_modified :: Union{Val{true}, Val{false}} = Val{false}()
end

struct ArmijoBacktrackingCache{F<:AbstractFloat} <: AbstractStepsizeCache
    factor :: F
    constant :: F
    sz0 :: F
    mode :: Union{Val{:all}, Val{:any}, Val{:max}}
    is_modified :: Union{Val{true}, Val{false}}
    lhs_vec :: Vector{F}
end

function init_cache(sz_rule::ArmijoBacktracking, mop::AbstractMOP)
    F = float_type(mop)
    @unpack factor, constant, sz0, mode, is_modified = sz_rule
    return ArmijoBacktrackingCache{F}(
        factor, constant, sz0, mode, is_modified, zeros(F, dim_out(mop))
    )
end

function stepsize!(
    carrays, sz_cache::ArmijoBacktrackingCache, mop::AbstractMOP, critval;
    kwargs...
)
    @unpack d, xd, fxd, x, fx = carrays
    @unpack factor, constant, sz0, mode, is_modified, lhs_vec = sz_cache
    return armijo_backtrack!(
        d, xd, fxd, mop, x, fx, critval, factor, constant, sz0, mode, lhs_vec, is_modified;
        kwargs...
    )
end

function armijo_backtrack!(
    d, xd, fxd,
    mop,
    x, fx, critval,
    factor, constant, sz0, mode, lhs_vec,
    is_modified :: Union{Val{true}, Val{false}} = Val{false}();
    x_tol_abs = 0,
    x_tol_rel = 0,
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
        rhs = critval * constant * sz

        d .*= sz
        d_norm = LA.norm(d, Inf)

        @. xd = x + d
        stop_code = objectives!(fxd, mop, xd)
        if stop_code isa STOP_CODE
            zero_step = true
        end
    end

    x_norm = LA.norm(x, Inf)

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
            
            if _armijo_test(mode, lhs_vec, φx, fxd, rhs)
                break
            end
            
            rhs = _armijo_apply_factor_to_rhs(is_modified, rhs, factor)
            sz *= factor
            d .*= factor
            d_norm *= factor
            @. xd = x + d
            stop_code = objectives!(fxd, mop, xd)
            if stop_code isa STOP_CODE
                zero_step = true
                break
            end
        end
    end
    if zero_step
        sz = 0
        d .= 0
        xd .= x
        fxd .= fx
    end

    if !(stop_code isa STOP_CODE)
        stop_code = sz
    end
    return stop_code 
end

function _armijo_apply_factor_to_rhs(is_modified::Val{false}, rhs, factor)
    return rhs * factor
end
function _armijo_apply_factor_to_rhs(is_modified::Val{true}, rhs, factor)
    ## rhs = constant * sz^2 * critval
    ## _sz = (sz * factor) ⇒ _sz^2 = sz^2 * factor^2
    ## ⇒ _rhs = constant * _sz^2 * critval = factor^2 * rhs
    return rhs * factor^2
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
function _armijo_test(::Val{:max}, lhs_vec, φx, fxd, rhs)
    return φx - maximum(fxd) >= rhs
end

Base.@kwdef struct SteepestDescentDirection{
    sz_ruleType, 
} <: AbstractStepRule
    sz_rule :: sz_ruleType = ArmijoBacktracking()
    set_metadata :: Bool = false
end

struct CommonStepCache{
    F<:AbstractFloat,
    sz_cacheType,
    fw_cacheType,
} <: AbstractStepRuleCache
    criticality_ref :: Base.RefValue{F}
    sz_cache :: sz_cacheType
    fw_cache :: fw_cacheType
end

function init_common_cache(sz_rule, mop)
    criticality_ref = Ref(convert(float_type(mop), Inf))
    sz_cache = init_cache(sz_rule, mop)
    fw_cache = init_frank_wolfe_cache(float_type(mop), dim_out(mop))
    return CommonStepCache(criticality_ref, sz_cache, fw_cache)
end

function init_step_meta(gather_meta_flag, dim_in, num_decomp, F)
    return init_step_meta(Val(gather_meta_flag), dim_in, num_decomp, F)
end
function init_step_meta(::Val{false}, dim_in, num_decomp, F) end
function init_step_meta(::Val{true}, dim_in, num_decomp, F)
    it_index = Ref(0)
    direction_matrix = Matrix{F}(undef, dim_in, num_decomp)
    direction_coefficients = Vector{F}(undef, num_decomp)
    sz = Ref(zero(F))
    return StepMeta(
        it_index, direction_matrix, direction_coefficients, sz
    ) 
end

struct SteepestDescentDirectionCache{
    ccacheType <: CommonStepCache,
    step_metaType
} <: AbstractStepRuleCache
    ccache :: ccacheType
    step_meta :: step_metaType
end

function criticality(carrays, step_cache::SteepestDescentDirectionCache)
    return step_cache.ccache.criticality_ref[]
end

metadata_type(step_cache::SteepestDescentDirectionCache)=typeof(step_cache.step_meta)
function _metadata(step_cache::SteepestDescentDirectionCache)
    return step_cache.step_meta
end

function init_cache(step_rule::SteepestDescentDirection, mop::AbstractMOP)
    ccache = init_common_cache(step_rule.sz_rule, mop)
    step_meta = init_step_meta(step_rule.set_metadata, dim_in(mop), 1, float_type(mop))
    return SteepestDescentDirectionCache(ccache, step_meta)
end
 
function step!(
    it_index, carrays, step_cache::SteepestDescentDirectionCache, mop::AbstractMOP; 
    kwargs...
)
    @unpack ccache, step_meta = step_cache
    
    ## set it_index in metadata
    if !isnothing(step_meta)
        step_meta.it_index[] = it_index
    end
    
    ## modify carrays.d to be steepest descent direction
    critval_sd = steepest_descent_direction!(carrays, ccache)

    ## set metadata
    if !isnothing(step_meta)
        step_meta.direction_matrix[:, 1] .= carrays.d
        step_meta.direction_coefficients[1] = 1
    end
    
    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack sz_cache = ccache
    sz = stepsize!(carrays, sz_cache, mop, critval_sd; kwargs...)
    if !(sz isa STOP_CODE)
        if !isnothing(step_meta)
            step_meta.sz[] = sz
        end
    end
    return sz
end

function steepest_descent_direction!(carrays, ccache)
    @unpack d, Dfx = carrays
    @unpack criticality_ref, fw_cache = ccache

    ## compute (negative) KKT multipliers for steepest descent direction
    α = frank_wolfe_multidir_dual!(fw_cache, Dfx)
    
    ## use these to set steepest descent direction `d`
    LA.mul!(d, Dfx', α)

    ## before scaling `d`, set criticality
    critval = criticality_ref[] = abs(maxdot(Dfx, d))
    return critval
end

function maxdot(Dfx, d)
    return mapreduce(Base.Fix1(LA.dot, d), max, eachrow(Dfx))
end

Base.@kwdef struct FletcherReevesRestart{
    F<:Real,
    sz_ruleType
} <: AbstractStepRule
    sz_rule :: sz_ruleType = ArmijoBacktracking(; is_modified=Val{true}())
    critval_mode :: Union{Val{:sd}, Val{:cg}} = Val{:sd}()
    wolfe_constant :: F = .1
end

abstract type AbstractCGCache <: AbstractStepRuleCache end

struct FletcherReevesRestartCache{
    F<:AbstractFloat,
    ccacheType
} <: AbstractCGCache
    ccache :: ccacheType
    critval_mode :: Union{Val{:sd}, Val{:cg}}
    criticality_ref :: Base.RefValue{F}
    wolfe_constant :: F
    d_prev :: Vector{F}
end
function criticality(carrays, step_cache::AbstractCGCache)
    return _cg_criticality(step_cache)
end
function _cg_criticality(step_cache)
    return _cg_criticality(step_cache.critval_mode, step_cache)
end
function _cg_criticality(::Val{:sd}, step_cache)
    return step_cache.ccache.criticality_ref[]
end
function _cg_criticality(::Val{:cg}, step_cache)
    return step_cache.criticality_ref[]
end

function init_cache(step_rule::FletcherReevesRestart, mop::AbstractMOP)
    ccache = init_common_cache(step_rule.sz_rule, mop)
    criticality_ref = deepcopy(ccache.criticality_ref)
    @unpack critval_mode = step_rule
    F = float_type(mop)
    wolfe_constant = convert(F, step_rule.wolfe_constant)
    d_prev = zeros(F, dim_in(mop))
    return FletcherReevesRestartCache(
        ccache, critval_mode, criticality_ref, wolfe_constant, d_prev)
end

function step!(
    it_index, carrays, step_cache::FletcherReevesRestartCache, mop::AbstractMOP; 
    kwargs...
)
    @unpack ccache = step_cache
    
    ## before updating steepest descent direction, store norm squared for
    ## CG coefficients 
    sd_prev_normsq = ccache.criticality_ref[]   # ‖δₖ₋₁‖^2
 
    ## modify carrays.d to be steepest descent direction
    sd_normsq = steepest_descent_direction!(carrays, ccache)    # store ‖δₖ‖^2 in ccache.criticality_ref[]

    @unpack d, Dfx = carrays
    @unpack d_prev, wolfe_constant = step_cache
    if it_index > 1
        ## check wolfe condition at unscaled direction
        fprev_dprev = step_cache.criticality_ref[]  # -max( ⟨∇fᵢ(xₖ₋₁), dₖ₋₁⟩ )
        upper_bound = wolfe_constant * fprev_dprev
        f_dprev = maxdot(Dfx, d_prev)   # max( ⟨∇fᵢ(xₖ), dₖ₋₁⟩ )
        lhs = max(
            abs(f_dprev),
            abs(LA.dot(d, d_prev))  # `d` is steepest descent direction atm
        )
        if lhs <= upper_bound
            β = sd_normsq / sd_prev_normsq
            θ = (f_dprev + fprev_dprev) / sd_prev_normsq
            d .*= θ
            d .+= β .* d_prev
        end
    end

    ## before scaling, store data for next iteration
    step_cache.criticality_ref[] = abs(maxdot(Dfx, d))
    step_cache.d_prev .= d

    d_normsq = sum( d.^2 )

    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack sz_cache = ccache
    return stepsize!(carrays, sz_cache, mop, d_normsq; kwargs...)
end

Base.@kwdef struct FletcherReevesFractionalLP{
    F<:Real,
    sz_ruleType
} <: AbstractStepRule
    sz_rule :: sz_ruleType = ArmijoBacktracking(; is_modified=Val{true}())
    critval_mode :: Union{Val{:sd}, Val{:cg}} = Val{:sd}()
    constant :: F = 1 + 1e-2
end

struct FletcherReevesFractionalLPCache{
    F<:AbstractFloat,
    ccacheType
} <: AbstractCGCache
    ccache :: ccacheType
    critval_mode :: Union{Val{:sd}, Val{:cg}}
    criticality_ref :: Base.RefValue{F}
    constant :: F

    d_prev :: Vector{F}

    Dprev_sdprev :: Vector{F}
    Dprev_dprev :: Vector{F}

    D_sd_tmp :: Vector{F}
end

function init_cache(step_rule::FletcherReevesFractionalLP, mop::AbstractMOP)
    ccache = init_common_cache(step_rule.sz_rule, mop)
    criticality_ref = deepcopy(ccache.criticality_ref)
    @unpack critval_mode = step_rule
    F = float_type(mop)
    constant = convert(F, step_rule.constant)
    d_prev = zeros(F, dim_in(mop))

    Dprev_sdprev = similar(d_prev)
    Dprev_dprev = similar(d_prev)
    D_sd_tmp = similar(d_prev)
    return FletcherReevesFractionalLPCache(
        ccache, critval_mode, criticality_ref, constant, d_prev,
        Dprev_sdprev, Dprev_dprev, D_sd_tmp
    )
end

function step!(
    it_index, carrays, step_cache::FletcherReevesFractionalLPCache, mop::AbstractMOP; 
    kwargs...
)
    @unpack ccache = step_cache

    ## modify carrays.d to be steepest descent direction
    steepest_descent_direction!(carrays, ccache)    # store ‖δₖ‖^2 in ccache.criticality_ref[]

    @unpack d, Dfx = carrays
    @unpack d_prev, Dprev_sdprev, Dprev_dprev, D_sd_tmp, constant = step_cache
    
    ## prepare storing product ∇f(xₖ)⋅δₖ for next iteration
    LA.mul!(D_sd_tmp, Dfx, d)  
    
    if it_index > 1
        ## Find minimizing index `w` for 
        ## ∇f_w(xₖ)ᵀ dₖ₋₁ ÷ ∇f_w(xₖ)ᵀ δₖ₋₁ 
        opt_val = Inf
        w_g_dprev = NaN     # ∇f_w(xₖ)ᵀ dₖ₋₁ 
        w_g_sd = NaN        # ∇f_w(xₖ)ᵀ δₖ₋₁ 
        w = 0
        for (_w, g) in enumerate(eachrow(Dfx))
            g_dprev = LA.dot(g, d_prev)
            g_sdprev = LA.dot(g, d)
            _opt_val = g_dprev / g_sdprev
            if _opt_val < opt_val
                opt_val = _opt_val
                w_g_dprev = g_dprev
                w_g_sd = g_sdprev
                w = _w
            end
        end
        ## use index to define `θ` and `β`
        ### retrieve terms
        w_gprev_dprev = Dprev_dprev[w]      # ∇f_w(xₖ₋₁)ᵀ dₖ₋₁ 
        w_gprev_sdprev = Dprev_sdprev[w]    # ∇f_w(xₖ₋₁)ᵀ δₖ₋₁ 
        
        denom = -constant * w_gprev_sdprev
        θ = (w_g_dprev - w_gprev_dprev - (constant-1) * w_gprev_sdprev) / denom
        β = -w_g_sd / denom

        ## set CG direction dₖ = θ δₖ + β dₖ₋₁
        d .*= θ
        d .+= β * d_prev
    end

    ## before scaling, store data for next iteration
    step_cache.criticality_ref[] = abs(maxdot(Dfx, d))
    d_prev .= d
    Dprev_sdprev .= D_sd_tmp
    LA.mul!(Dprev_dprev, Dfx, d)

    d_normsq = sum( d.^2 )

    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack sz_cache = ccache
    return stepsize!(carrays, sz_cache, mop, d_normsq; kwargs...)
end

Base.@kwdef struct PRP3{
    sz_ruleType
} <: AbstractStepRule
    sz_rule :: sz_ruleType = ArmijoBacktracking(; is_modified=Val{true}())
    critval_mode :: Union{Val{:sd}, Val{:cg}} = Val{:sd}()
end

struct PRP3Cache{
    F<:AbstractFloat,
    ccacheType
} <: AbstractCGCache
    ccache :: ccacheType
    critval_mode :: Union{Val{:sd}, Val{:cg}}
    criticality_ref :: Base.RefValue{F}

    d_prev :: Vector{F}
    sd_prev :: Vector{F}
    y :: Vector{F}
end

function init_cache(step_rule::PRP3, mop::AbstractMOP)
    ccache = init_common_cache(step_rule.sz_rule, mop)
    criticality_ref = deepcopy(ccache.criticality_ref)
    @unpack critval_mode = step_rule
    F = float_type(mop)
    d_prev = zeros(F, dim_in(mop))
    sd_prev = similar(d_prev)
    y = similar(d_prev)

    return PRP3Cache(
        ccache, critval_mode, criticality_ref, 
        d_prev, sd_prev, y,
    )
end

function step!(
    it_index, carrays, step_cache::PRP3Cache, mop::AbstractMOP; 
    kwargs...
)
    @unpack ccache = step_cache
    ## before updating steepest descent direction, store norm squared for
    ## denominator in CG coefficients 
    sd_prev_normsq = ccache.criticality_ref[]   # ‖δₖ₋₁‖^2
    
    ## modify carrays.d to be steepest descent direction
    steepest_descent_direction!(carrays, ccache)    # store ‖δₖ‖^2 in ccache.criticality_ref[]

    @unpack d, Dfx = carrays
    @unpack sd_prev, d_prev, y = step_cache
   
    if it_index == 1
        β = θ = 0
    else
        ## set difference vector
        ### at this point, `sd_prev` holds δₖ₋₁
        y .= sd_prev - d     # yₖ = δₖ₋₁ - δₖ

        ## determine (wβ, vβ) by solving discrete minimax problem
        ## min_w max_v ∇f_w(xₖ)ᵀyₖ ⋅ ∇f_v(xₖ)ᵀ dₖ₋₁
        wβ = vβ = 0
        wβ_g_y = NaN    # ∇f_wβ(xₖ)ᵀyₖ

        ## determine (wθ, vθ) by solving discrete maximin problem
        ## max_w min_v ∇f_v(xₖ)ᵀyₖ ⋅ ∇f_w(xₖ)ᵀ dₖ₋₁
        wθ = vθ = 0
        wθ_g_dprev = NaN    # ∇f_wθ(xₖ)ᵀ dₖ₋₁

        minimax_outer = Inf
        maximin_outer = -Inf
        for (w, gw) in enumerate(eachrow(Dfx))
            minimax_inner = -Inf
            maximin_inner = Inf
            vmax = 0
            vmin = 0
            w_g_y = LA.dot(gw, y)
            w_g_dprev = LA.dot(gw, y)
            for (v, gv) in enumerate(eachrow(Dfx))
                v_g_dprev = LA.dot(gv, d_prev)
                _minimax_inner = w_g_y * v_g_dprev
                if _minimax_inner > minimax_inner
                    minimax_inner = _minimax_inner
                    vmax = v
                end
                v_g_y = LA.dot(gv, y)
                _maximin_inner = w_g_dprev * v_g_y
                if _maximin_inner < maximin_inner
                    maximin_inner = _maximin_inner
                    vmin = v
                end
            end
            if minimax_inner < minimax_outer
                minimax_outer = minimax_inner
                wβ_g_y = w_g_y
                wβ = w
                vβ = vmax
            end
            if maximin_inner > maximin_outer
                maximin_outer = maximin_inner
                wθ_g_dprev = w_g_dprev
                wθ = w
                vθ = vmin
            end
        end
        
        ψβ = minimax_outer
        ψθ = maximin_outer
        ## Note: Minimax Theorem holds for discrete problems: ψβ ≥ ψθ

        ## determine balancing coefficients
        αβ, αθ = if ψθ <= 0 && ψβ >= 0
            ### sign swith / restart
            (0, 0)
        elseif ψθ > 0
            ### both positive, 0 < ψθ ≤ ψβ
            ## shrink larger factor
            (ψθ/ψβ, 1)
        else
            ### both negative, ψθ ≤ ψβ < 0
            ### grow smaller factor
            (1, ψβ/ψθ)
        end

        ## finally, set coefficients
        β = αβ * wβ_g_y / sd_prev_normsq
        θ = αθ * wθ_g_dprev / sd_prev_normsq
    end

    ## before modifying `d`, store steepest descent direction for next iteration
    sd_prev .= d

    ## build CG direction
    d .+= β * d_prev
    d .-= θ * y
    
    ## before scaling, store data for next iteration
    step_cache.criticality_ref[] = abs(maxdot(Dfx, d))
    d_prev .= d

    d_normsq = sum( d.^2 )

    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack sz_cache = ccache
    return stepsize!(carrays, sz_cache, mop, d_normsq; kwargs...)
end

Base.@kwdef struct PRPConeProjection{
    sz_ruleType
} <: AbstractStepRule
    sz_rule :: sz_ruleType = ArmijoBacktracking(; is_modified=Val{true}())
    critval_mode :: Union{Val{:sd}, Val{:cg}} = Val{:sd}()
end

struct PRPConeProjectionCache{
    F<:AbstractFloat,
    ccacheType
} <: AbstractCGCache
    ccache :: ccacheType
    critval_mode :: Union{Val{:sd}, Val{:cg}}
    criticality_ref :: Base.RefValue{F}

    d_prev :: Vector{F}
    d_orth :: Vector{F}
    d_opt :: Vector{F}
    Dfx_prev :: Matrix{F}
end

function init_cache(step_rule::PRPConeProjection, mop::AbstractMOP)
    ccache = init_common_cache(step_rule.sz_rule, mop)
    criticality_ref = deepcopy(ccache.criticality_ref)
    @unpack critval_mode = step_rule

    F = float_type(mop)
    d_prev = zeros(F, dim_in(mop))
    d_orth = similar(d_prev)
    d_opt = similar(d_prev)

    Dfx_prev = zeros(F, dim_out(mop), dim_in(mop))

    return PRPConeProjectionCache(
        ccache, critval_mode, criticality_ref, 
        d_prev, d_orth, d_opt, Dfx_prev
    )
end

function step!(
    it_index, carrays, step_cache::PRPConeProjectionCache, mop::AbstractMOP; 
    kwargs...
)
    @unpack ccache = step_cache
    ## before updating steepest descent direction, store norm squared for
    ## denominator in CG coefficients 
    sd_prev_normsq = ccache.criticality_ref[]   # -𝔣(δₖ₋₁, xₖ₋₁) = ‖δₖ₋₁‖^2
    
    ## modify carrays.d to be steepest descent direction
    steepest_descent_direction!(carrays, ccache)    # store ‖δₖ‖^2 in ccache.criticality_ref[]

    @unpack d, Dfx = carrays
    @unpack d_prev, d_opt, d_orth, Dfx_prev = step_cache
   
    if it_index > 1
        sd_normsq = ccache.criticality_ref[]    # -𝔣(δₖ, xₖ)
        g_prev_sd = maxdot(Dfx_prev, d)         # 𝔣(δₖ, xₖ₋₁)
        β = (g_prev_sd + sd_normsq) / sd_prev_normsq

        d_prev .*= β
        minimax_outer = Inf
        for (w, gw) in enumerate(eachrow(Dfx))
            project_on_ker!(d_orth, d_prev, gw)
            minimax_inner = maxdot(Dfx, d_orth)
            if minimax_inner < minimax_outer
                minimax_outer = minimax_inner
                d_opt .= d_orth
            end
        end
        if minimax_outer <= 0
            ## build CG direction
            d .+= d_opt
        end
    end
    
    ## before scaling, store data for next iteration
    step_cache.criticality_ref[] = abs(maxdot(Dfx, d))
    d_prev .= d

    d_normsq = sum( d.^2 )

    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack sz_cache = ccache
    return stepsize!(carrays, sz_cache, mop, d_normsq; kwargs...)
end

function project_on_ker!(
    d⊥::AbstractVector, d::AbstractVector, g::AbstractVector
)
    @assert length(d⊥) == length(d) == length(g)
    g_normsq = sum( g.^2 )
    d⊥ .= d
    for (i, gi) = enumerate(g)
        d⊥[i] -= gi * LA.dot(g, d) / g_normsq
    end
    return d⊥
end

