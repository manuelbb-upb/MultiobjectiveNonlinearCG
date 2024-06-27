# This file is meant to be parsed by Literate.jl            #src
# Activate `docs` environment:                              #src
using Pkg                                                   #src
Pkg.activate(joinpath(@__DIR__, "..", ".."))                #src
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
import MultiobjectiveNonlinearCG: GatheringCallback, _x_mat, _fx_mat
#%%#src
x0 = [3.1416, -2.7182]
step_rule = MCG.SteepestDescentDirection(MCG.ArmijoBacktracking(; factor=.8, is_modified=Val(true)))
step_rule = MCG.FletcherReevesRestart()
callback = GatheringCallback(mop)
opt_res = MCG.optimize(x0, mop, callback; step_rule, max_iter=10)

fig, ax, _ = lines([(-1, -1), (1,1)])
ax.aspect = AxisAspect(1)
scatterlines!(ax, _x_mat(callback); markersize = range(15, 5, length(callback.x)))
fig