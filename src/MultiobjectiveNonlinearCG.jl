module MultiobjectiveNonlinearCG    #src
# # Useful Tools
# I really like the `@unpack` macro for NamedTuples and custom types:
import UnPack: @unpack
# `LinearAlgebra` also comes in handy...
import LinearAlgebra as LA
# # Stop Codes
# We define stop codes right at the beginning, because they should be available 
# globally for all functions to return:
@enum STOP_CODE :: Int8 begin
    STOP_MAX_ITER
    STOP_BUDGET_FUNCS
    STOP_BUDGET_GRADS
    STOP_CRIT_TOL_ABS
    STOP_X_TOL_REL
    STOP_X_TOL_ABS
    STOP_FX_TOL_REL
    STOP_FX_TOL_ABS
end

# # The Multi-Objective Optimization Problem
#=
We want to minimize multiple nonlinear, smooth objectives.
The unconstrained problem is 
```math
    \min_{x ∈ ℝ^N} 
        \begin{bmatrix}
			f_1(x)
			\\
            ⋮
            \\
            f_K(x)
        \end{bmatrix}
    \tag{MOP}
```
To perform optimization, we need a way to query metadata for (MOP), 
and evaluate and differentiate the objective functions.
=#

# ## Interface
# We accept problems subtyping `AbstractMOP`:
abstract type AbstractMOP end
# Such problem objects should implement these metadata functions:
"Return the input dimension (the number of variables) of problem `mop`."
dim_in(mop::AbstractMOP)=0
"Return the output dimension (the number of objectives) of problem `mop`."
dim_out(mop::AbstractMOP)=0
# The `float_type` function is optional. It forces the element type for arrays.
float_type(mop::AbstractMOP)=Float64

# The primal value vector is queried with the mutating function `func!`,
# where the first argument is the value vector.
# Thereafter follow immutable arguments.
# First, the problem `mop` to define evaluation.
# Then an object of type `AbstractDifferentiationOrder`:
abstract type AbstractDifferentiationOrder end
# For our first order method, there are only 2 options:
struct OutputValues <: AbstractDifferentiationOrder end
struct OutputGradients <: AbstractDifferentiationOrder end
# Finally, `func!` takes the input vector.
func!(y, mop::AbstractMOP, ::OutputValues, x)=nothing
# For the Jacobian, the first argument should be a matrix, and the index is different:
func!(Dy, mop::AbstractMOP, ::OutputGradients, x)=nothing
# Optionally, we can optimize for a combined pass:
struct OutputValuesAndGradients <: AbstractDifferentiationOrder end
function func!(y, Dy, mop::AbstractMOP, ::OutputValuesAndGradients, x)
    stop_code = func!(y, mop, OutputValues(), x)
    if stop_code isa STOP_CODE
        return stop_code
    end
    return func!(Dy, mop, OutputGradients(), x)
end
    
# The indexed interface is meant to be implemented, but we also derive functions
# with a more memorable name and simpler signature:
objectives!(y, mop::AbstractMOP, x)=func!(y, mop, OutputValues(), x)
jac!(Dy, mop::AbstractMOP, x)=func!(Dy, mop, OutputGradients(), x)
objectives_and_jac!(y, Dy, mop::AbstractMOP, x)=func!(y, Dy, mop, OutputValuesAndGradients(), x)

# ## Example Implementation
# We provide a very basic implementation for testing purposes: `BasicMOP`.
include("basic_mop.jl")

# Besides the “reference implementation” `BasicMOP`, we also have a wrapper for 
# counting and restricting function calls.
# The type `CountedMOP` is defined in a separate file:
include("counted_mop.jl")

# # Steps and Descent Directions
# Our algorithm is designed to be very flexible with regard to the actual descent steps used.
# We take some configuration of type `AbstractStepRule`.
# Then, a cache of type `AbstractStepRuleCache` is constructed.
# We use the cache for storing temporary data and for dispatch, e.g., with `step!`
abstract type AbstractStepRule end
abstract type AbstractStepRuleCache end

# Initialization function for cache corresponding to `step_rule`:
function init_cache(step_rule::AbstractStepRule, mop::AbstractMOP)
    return nothing
end

# Step computation function.
# The object `carrays` is a `NamedTuple` with **common arrays**.
# The methods for `step!` must set `carrays.d` to contain the step vector in variable 
# space, `carrays.xd` to the next iteration vector, and `carrays.fxd` to the next value 
# vector.
function step!(it_index, carrays, ::AbstractStepRuleCache, mop::AbstractMOP; kwargs...)
    nothing
end

# A function to return a criticality value, for stopping:
criticality(carrays, ::AbstractStepRuleCache)=Inf

# ## Stepsize
# Within the `step!` implementation, we might wish to employ different stepsize methods.
# We provide an interface similar to the `AbstractStepRule` interface.
# We don't explicity use such objects in the algorithm, but an `AbstractStepRuleCache`
# can reference them, for example.
abstract type AbstractStepsizeRule end
abstract type AbstractStepsizeCache end

# Initalization function for `AbstractStepsizeCache`:
function init_cache(sz_rule::AbstractStepsizeRule, ::AbstractMOP)
    nothing
end

# Mutating stepsize function.
# A `stepsize!` method must correctly set `d`, `xd` and `fxd`.
function stepsize!(carrays, ::AbstractStepsizeCache, ::AbstractMOP, critval; kwargs...)
    nothing
end

# ### Actual Directions
# The implementations are stored in a separate file:
include("descent.jl")

# # Main Algorithm

# Before giving the complete loop, we define a default (no-op) callback.
# It is called at the beginning of each iteration.
# A user can provide their own callback function instead.
function default_callback(it_index, carrays, mop, step_cache) end

# The callback can be used for stopping.
# If it returns a `STOP_CODE`, then we interrupt the algorithm and return.

# Because we check stopping criteria quite often, there is the helper macro 
# `@ignorebreak`.
## helper for `@ignorebreak`
function _parse_ignoraise_expr(ex)
	has_lhs = false
	if Meta.isexpr(ex, :(=), 2)
		lhs, rhs = esc.(ex.args)
		has_lhs = true
	else
		lhs = nothing	# not really necessary
		rhs = esc(ex)
	end
	return has_lhs, lhs, rhs
end

"""
    @ignorebreak do_something(args...)
    @ignorebreak lhs = do_something(args...)

If the expression `do_something` returns a `STOP_CODE`, then `break`.
Otherwise assign the result to `lhs`, if there is a left-hand side.
"""
macro ignorebreak(ex, indent_ex=0)
	has_lhs, lhs, rhs = _parse_ignoraise_expr(ex)
	return quote
		ret_val = $(rhs)
		do_break = ret_val isa STOP_CODE
		$(if has_lhs
			:($(lhs) = ret_val)
		else
			:(ret_val = nothing)
		end)
		do_break && break		
	end
end

# One more helper to save some lines:
function nothing_or_stop_code( stop_code_condition, stop_code)
    if stop_code_condition
        return stop_code
    end
    return nothing
end

# The common arrays `carrays` (that we have already seen above) come from this function:
function initialize_common_arrays(mop)
    F = float_type(mop)
    n_vars = dim_in(mop)
    n_objfs = dim_out(mop)

    ## current variable vector and values
    x = zeros(F, n_vars)
    fx = zeros(F, n_objfs)
    ## current Jacobian
    Dfx = zeros(F, n_objfs, n_vars)

    ## (scaled) descent direction
    d = similar(x)
    ## next variable vector and next values
    xd = similar(x)
    fxd = similar(fx)

    ## return NamedTuple for easy unpacking
    return (; x, fx, Dfx, d, xd, fxd)
end

# Prepare for logging with a nice helper function:
import Printf: @sprintf
import Logging: @logmsg, Info
function pretty_row_vec(
	x::AbstractVector;
	cutoff=80
)
	repr_str = "["
	lenx = length(x)
	for (i, xi) in enumerate(x)
		xi_str = @sprintf("%.2e", xi)
		if length(repr_str) + length(xi_str) >= cutoff
			repr_str *= "..."
			break
		end
		repr_str *= xi_str
		if i < lenx
			repr_str *= ", "
		end
	end
	repr_str *= "]"
	return repr_str
end
pretty_row_vec(x)=string(x)
# The optimization algorithm:
function optimize(
    x0, mop::AbstractMOP,
    @nospecialize(callback=default_callback)
    ;
    step_rule = SteepestDescentDirection(),
    max_iter = 500,
    max_func_calls = Inf,
    max_grad_calls = Inf,
    crit_tol_abs = 1e-10,    # Optim.jl has 1e-8
    x_tol_rel = 0,
    x_tol_abs = 0,
    fx_tol_rel = 0,
    fx_tol_abs = 0,
    log_level = Info
)
    ## some sanity checks
    @assert length(x0) == dim_in(mop) "Variable vector has wrong length."
    @assert dim_in(mop) > 0 "Variable dimension must be positive."
    @assert dim_out(mop) > 0 "Output dimension must be positive."

    ## safeguard `max_iter`
    if isnothing(max_iter) || max_iter < 0
        max_iter = Inf
    end

    ## enable counting and restrict number of function calls
    mop = CountedMOP(mop; max_func_calls, max_grad_calls)
   
    ## initialize common arrays and set first iteration site
    carrays = initialize_common_arrays(mop)
    @unpack x, fx, Dfx = carrays
    copyto!(x, x0)

    ## prepare cache for steps
    step_cache = init_cache(step_rule, mop)

    @unpack d, xd, fxd = carrays
    it_index = 1
    stop_code = nothing
    callback_called = false
    while true
        callback_called = false
        @ignorebreak stop_code = nothing_or_stop_code(it_index > max_iter, STOP_MAX_ITER)
        
        @logmsg log_level """#========================================#
        Iteration $(it_index)
        x  = $(pretty_row_vec(x))
        fx = $(pretty_row_vec(fx))"""

        @ignorebreak stop_code = objectives_and_jac!(fx, Dfx, mop, x)

        @ignorebreak stop_code = step!(
            it_index, carrays, step_cache, mop;
            crit_tol_abs, x_tol_rel, x_tol_abs, fx_tol_rel, fx_tol_abs
        )
        
        critval = criticality(carrays, step_cache)
        @ignorebreak stop_code = nothing_or_stop_code( 
            critval <= crit_tol_abs, STOP_CRIT_TOL_ABS)

        ## compute change values **before** updating `x`, `fx` etc.
        x_change = LA.norm(d, Inf)
        fx_norm = LA.norm(fx, Inf)
        fx .-= fxd
        fx_change = LA.norm(fx, Inf)
 
        @logmsg log_level """\n
        critval   = $(critval)
        x_change  = $(x_change)
        fx_change = $(fx_change)"""
        
        @ignorebreak stop_code = nothing_or_stop_code(
            x_change <= x_tol_abs, STOP_X_TOL_ABS)
        @ignorebreak stop_code = nothing_or_stop_code(
            x_change <= x_tol_rel * LA.norm(x, Inf), STOP_X_TOL_REL)

        @ignorebreak stop_code = nothing_or_stop_code(
            fx_change <= fx_tol_abs, STOP_FX_TOL_ABS)
        @ignorebreak stop_code = nothing_or_stop_code(
            fx_change <= x_tol_rel * fx_norm, STOP_FX_TOL_REL)
      
        @ignorebreak stop_code = begin
            callback_called = true
            callback(it_index, carrays, mop, step_cache)
        end
        
        x .= xd
        fx .= fxd
        @ignorebreak stop_code = jac!(Dfx, mop, xd)
       
        it_index += 1
    end
    if !callback_called
        callback(it_index, carrays, mop, step_cache)
    end

    return (; 
        x, fx, carrays, step_cache, stop_code, 
        num_func_calls = counted_mop_calls(mop, OutputValues()),
        num_grad_calls = counted_mop_calls(mop, OutputGradients())
    )
end

# # User Utilities
# This file has some tools that might be cool for experiments:
include("utils.jl")
end#src