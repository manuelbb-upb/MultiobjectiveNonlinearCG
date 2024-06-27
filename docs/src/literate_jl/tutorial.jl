# This file is meant to be parsed by Literate.jl            #src
# Activate `docs` environment:                              #src
if !(joinpath(@__DIR__, "..", "..") in LOAD_PATH)           #src
    pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))   #src
end                                                         #src
#%%#src
# # MultiobjectiveNonlinearCG Tutorials
import MultiobjectiveNonlinearCG as MCG
using CairoMakie
#%%#src
mop = MCG.BasicMOP(;
    dim_in = 2,
    dim_out = 2,
    objectives! = function (y, x)
        y[1] = sum( (x .- 1).^2 )
        y[2] = sum( (x .+ 1).^2 )
        nothing
    end,
    jac! = function(Dy, x)
        Dy[1, :] .= 2 .* (x .- 1)
        Dy[2, :] .= 2 .* (x .+ 1)
        nothing
    end
)
#%%#src
struct GatheringCallback{F}
    x :: Vector{Vector{F}}
    fx :: Vector{Vector{F}}
end
function GatheringCallback(mop)
    F = MCG.float_type(mop)
    x = Vector{F}[]
    fx = Vector{F}[]
    return GatheringCallback(x, fx)
end
function (callback::GatheringCallback)(it_index, carrays, mop, step_cache)
    push!(callback.x, copy(carrays.x))
    push!(callback.fx, copy(carrays.fx))
end
_x_mat(callback)=reduce(hcat, callback.x)
_fx_mat(callback)=reduce(hcat, callback.fx)
#%%#src
x0 = [3.1416, -2.7182]
step_rule = MCG.SteepestDescentDirection(MCG.StandardArmijoBacktracking(; factor=.8))
callback = GatheringCallback(mop)
opt_res = MCG.optimize(x0, mop, callback; step_rule, max_iter=100)

fig, ax, _ = lines([(-1, -1), (1,1)])
ax.aspect = AxisAspect(1)
scatterlines!(ax, _x_mat(callback); markersize = range(15, 5, length(callback.x)))
fig