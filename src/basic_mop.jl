# `BasicMOP` implements the `AbstractMOP` interface.
# The type simply stores all information in its fields:
Base.@kwdef struct BasicMOP{F<:AbstractFloat, objectivesType, jacType} <: AbstractMOP
    float_type :: Type{F} = Float64
    dim_in :: Int = 0
    dim_out :: Int = 0
    objectives! :: objectivesType = nothing
    jac! :: jacType = nothing
end
# The methods are fairly simple:
dim_in(mop::BasicMOP)=mop.dim_in
dim_out(mop::BasicMOP)=mop.dim_out
float_type(mop::BasicMOP{F}) where F = F
# Evaluation:
func!(y, mop::BasicMOP, ::OutputValues, x) = mop.objectives!(y, x)
func!(Dy, mop::BasicMOP, ::OutputGradients, x) = mop.jac!(Dy, x)

# ### Setup Functions
# The `BasicMOP` type is immutable to have strongly typed fields `objectives!` and `jac!`.
# Changes can be done with macros from `Accessors`.

# Additionally, we allow for providing out-of-place evaluation functions by wrapping them:
struct MakeInPlaceWrapper{wrappedType}
    wrapped :: wrappedType
end
function (wrapper::MakeInPlaceWrapper)(y, x)
    _y = wrapper.wrapped(x)
    Base.copyto!(y, _y)
    return nothing
end
# With that, our utility functions can be defined.
# First, a helper to set any function field of `mop` and wrap the function handle:
import Accessors: set, PropertyLens
function set_function_field(mop, ::Val{func_field}, func!; is_inplace=true) where {func_field}
    if is_inplace
        ## nothing to do, just set the field and return new object
        return set(mop, PropertyLens{func_field}(), func!)
    end
    _func! = MakeInPlaceWrapper(func!)
    return set_objectives(mop, _func!; is_inplace=true)
end
# From this, the specialized utility functions are derived:
"""
    set_objectives(mop, objectives_func!; is_inplace=true)

Return a new `BasicMOP` that has its `objectives!` field set to a function derived from
`objectives_func!`. If `is_inplace==true`, then `objectives_func!` must have signature
`objectives_func!(y, x)`, i.e., it is an “in-place” function.
"""
function set_objectives(mop::BasicMOP, objectives_func!; is_inplace=true)
    return set_function_field(mop, Val{:objectives!}(), objectives_func!; is_inplace)
end

"""
    set_jac(mop, jac_func!; is_inplace=true)

Return a new `BasicMOP` that has its `jac!` field set to a function derived from
`jac_func!`.
"""
function set_jac(mop::BasicMOP, jac_func!; is_inplace=true)
    return set_function_field(mop, Val{:jac!}(), jac_func!; is_inplace)
end