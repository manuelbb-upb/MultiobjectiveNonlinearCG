# For stopping, we might wish to count the number of function calls -- or even
# avoid evaluation when a limit is hit.
# We have a wrapper for that:
struct CountedMOP{mopType, T<:Real} <: AbstractMOP 
    mop :: mopType
    num_calls :: Vector{Int}
    max_calls :: Vector{T}
end

function CountedMOP(
    mop::AbstractMOP;
    max_func_calls=Inf,
    max_grad_calls=Inf,
)
    return CountedMOP(mop, Int[0, 0], [max_func_calls, max_grad_calls])
end

# The number of function calls is stored in the integer vector:
counted_mop_calls(cmop, ::OutputValues) = first(cmop.num_calls)
counted_mop_max_calls(cmop, ::OutputValues) = first(cmop.max_calls)
counted_mop_calls(cmop, ::OutputGradients) = last(cmop.num_calls)
counted_mop_max_calls(cmop, ::OutputGradients) = last(cmop.max_calls)
inc_num_calls!(cmop, ::OutputValues) = (cmop.num_calls[1] += 1)
inc_num_calls!(cmop, ::OutputGradients) = (cmop.num_calls[2] += 1)

# Forward the interface:
# The methods are fairly simple:
dim_in(cmop::CountedMOP)=dim_in(cmop.mop)
dim_out(cmop::CountedMOP)=dim_out(cmop.mop)
float_type(cmop::CountedMOP)=float_type(cmop.mop)

# We modify the evaluation functions to return a stop code if necessary:
# These are first returned by a pre-check function …
_budget_code(::OutputValues)=STOP_BUDGET_FUNCS
_budget_code(::OutputGradients)=STOP_BUDGET_GRADS
function check_counted_function(cmop, diff_order::AbstractDifferentiationOrder)
    if counted_mop_calls(cmop, diff_order) >= counted_mop_max_calls(cmop, diff_order)
        return _budget_code(diff_order)
    end
    return nothing
end
# … and then passed down for further inspection in the algorithm:
function counted_func!(cmop, diff_order, x, mutable_args...)
    ret_code = check_counted_function(cmop, diff_order)
    !(isnothing(ret_code)) && return ret_code
    ret_code = func!(mutable_args..., cmop.mop, diff_order, x)
    inc_num_calls!(cmop, diff_order)
    return ret_code
end
func!(y, cmop::CountedMOP, diff_order::OutputValues, x)=counted_func!(cmop, diff_order, x, y)
func!(Dy, cmop::CountedMOP, diff_order::OutputGradients, x)=counted_func!(cmop, diff_order, x, Dy)
func!(Dy, cmop::CountedMOP, diff_order::OutputValuesAndGradients, x)=counted_func!(cmop, diff_order, x, y, Dy)