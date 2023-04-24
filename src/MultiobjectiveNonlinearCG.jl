module MultiobjectiveNonlinearCG

const MIN_PRECISION = Float32;

abstract type AbstractStepsizeRule end
abstract type AbstractStepsizeCache end

abstract type AbstractDirRule end
abstract type AbstractDirCache end

stepsize_rule(::AbstractDirRule)::AbstractStepsizeRule=nothing

function init_cache(::AbstractDirRule, x, fx, DfxT, d, objf!, jacT!, meta)::AbstractDirCache
    return nothing
end

function init_cache(::AbstractStepsizeRule, x, fx, DfxT, d, objf!, jacT!, meta)::AbstractDirCache
    return nothing
end

function stepsize(::AbstractStepsizeCache, x, fx, DfxT, d, objf!, jacT!, meta)::Number
    return nothing
end

Base.@kwdef struct FixedStepsize{F} <: AbstractStepsizeRule
    stepsize :: F = MIN_PRECISION(1e-3)
end    

Base.@kwdef struct SteepestDescent{SR<:AbstractStepsizeRule} <: AbstractDirRule 
    stepsize_rule :: SR = FixedStepsize()
end

struct SteepestDescentCache{SD} <: AbstractDirCache
end

include("multidir_frank_wolfe.jl")

struct MetaDataDev1
    dim_in :: Int
    dim_out :: Int
end

struct MetaDataDev2{P}
    dim_in :: Int
    dim_out :: Int
    precision :: P
end

MetaData = MetaDataDev2

function optimize(
    x0 :: AbstractVector{X}, fx0::AbstractVector{Y}, objf!, jacT!;
    max_iter=100,
    descent_rule=SteepestDescent(),
) where {X<:Number, Y<:Number}
    @assert !isempty(objectives) "There are no objective functions."

    # initialize/prealloc iterates:
    T = Base.promote_type(MIN_PRECISION, X, Y)
    x = T.(x0)
    fx = T.(fx0)

    # prealloc transposed jacobian
    dim_in = length(x)
    dim_out = length(fx)
    precision = T
    DfxT = zeros(precision, dim_in, dim_out)
    # also set metadata
    meta = MetaData(dim_in, dim_out, precision)

    # prealloc array for step `d`
    d = similar(x)

    if max_iter > 0
        descent_cache, stepsize_cache = first_iteration!(x, fx, DfxT, d, objf!, jacT!, descent_rule, meta)
    end
end

function first_iteration!(
    # mutated
    x, fx, DfxT, d,
    # not-mutated
    objf!, jacT!, descent_rule, meta
)
    jacT!(DfxT, x)

    descent_cache = init_cache(descent_rule, x, fx, DfxT, d, objf!, jacT!, meta)
    first_direction!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
    stepsz_rule = stepsize_rule(descent_rule)
    stepsize_cache = init_cache(stepsz_rule, x, fx, DfxT, d, objf!, jacT!, meta)
    σ = stepsize(stepsize_cache, descent_cache, x, fx, DfxT, d, objf!, jacT!, meta)
    x .+= σ .* d
    return descent_cache, stepsize_cache
end

function iterate!(
    # mutated:

    # not mutated:
)
end

end