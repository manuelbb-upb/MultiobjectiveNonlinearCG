module MultiobjectiveNonlinearCG

const MIN_PRECISION = Float32;

abstract type AbstractStoppingCriterion end

function stop_before_iteration(::AbstractStoppingCriterion, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Bool
    return false
end

function stop_after_iteration(::AbstractStoppingCriterion, it_index, x, fx, DfxT, d, objf!, jacT!, meta)::Bool
    return false
end

function check_criteria(stopping_criteria, crit_eval_func, args...)
    do_iteration = true
    for crit in stopping_criteria
        if crit_eval_func(crit, args...)
            do_iteration = false
            break
        end
    end
    return do_iteration  
end

function check_criteria_before(stopping_criteria, args...)
    return check_criteria(stopping_criteria, stop_before_iteration, args...)
end
function check_criteria_after(stopping_criteria, args...)
    return check_criteria(stopping_criteria, stop_after_iteration, args...)
end

abstract type AbstractDirRule end
abstract type AbstractDirCache end

# mandatory
function init_cache(::AbstractDirRule, x, fx, DfxT, d, objf!, jacT!, meta)::AbstractDirCache
    return nothing
end

# mandatory
function first_step!(descent_cache::AbstractDirCache, d, x, fx, DfxT, objf!, jacT!, meta)::Nothing
    @error "No implementation of `first_step!` for `$(descent_cache)`."
end

# optional, but likely needed
function step!(descent_cache::AbstractDirCache, d, x, fx, DfxT, objf!, jacT!, meta)::Nothing
    return first_step!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
end

# optional
function stop_after(descent_cache::AbstractDirCache, it_index, d, x, fx, DfxT, objf!, jacT!, meta)::Bool
    return false
end

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

include("dir_rules/all_rules.jl")

function optimize(
    x0 :: AbstractVector{X}, fx0::AbstractVector{Y}, objf!, jacT!;
    max_iter=100,
    descent_rule=SteepestDescentRule(FixedStepsizeRule()),
    stopping_criteria = Any[]
) where {X<:Number, Y<:Number}

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

    # if there are iterations, perform first iteration and allocate `descent_cache`
    if max_iter > 0
        if check_criteria_before(stopping_criteria, 1, x, fx, DfxT, d, objf!, jacT!, meta)
            jacT!(DfxT, x)
            descent_cache = init_cache(descent_rule, x, fx, DfxT, d, objf!, jacT!, meta)
            first_step!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
            x .+= d
            objf!(fx, x)
        end
    end
    if (
        check_criteria_after(stopping_criteria, 1, x, fx, DfxT, d, objf!, jacT!, meta) &&
        !stop_after(descent_cache, 1, d, x, fx, DfxT, objf!, jacT!, meta)
    )
        for it_ind=2:max_iter
            if check_criteria_before(stopping_criteria, it_ind, x, fx, DfxT, d, objf!, jacT!, meta)
                jacT!(DfxT, x)
                step!(descent_cache, d, x, fx, DfxT, objf!, jacT!, meta)
                x .+= d
                objf!(fx, x)

                if (
                    !check_criteria_after(stopping_criteria, it_ind, x, fx, DfxT, d, objf!, jacT!, meta) ||
                    stop_after(descent_cache, it_ind, d, x, fx, DfxT, objf!, jacT!, meta)
                )
                    break
                end
            end
        end        
    end

    return x, fx
end

end