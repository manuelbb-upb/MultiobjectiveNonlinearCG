# This file is meant to be parsed by Literate.jl            #src
# Activate `docs` environment:                              #src
using Pkg                                                   #src
Pkg.activate(joinpath(@__DIR__, "..", ".."))                #src
#%%#src
# # MultiobjectiveNonlinearCG Tutorials
import MultiobjectiveNonlinearCG as MCG
using CairoMakie
#%%#src

## Define the function:
f(x1, x2, a, b) = b * (x2 - x1^2)^2 + (a - x1)^2

# The gradient of the Rosenbrock function is known analitically (see above):
df(x1, x2, a, b) = [
    -4*b*(x2-x1^2)*x1 - 2*(a-x1),
    2*b*(x2-x1^2)
]

objectives! = function (y, x)
    y[1] = f(x[1], x[2], 1, 100)
    y[2] = f(x[1], x[2], 2, 100)
end

jac! = function (Dy, x)
    Dy[1, :] .= df(x[1], x[2], 1, 100)
    Dy[2, :] .= df(x[1], x[2], 2, 100)
end

mop = MCG.BasicMOP(;
    dim_in = 2,
    dim_out = 2,
    objectives!,
    jac!
)

x0 = [4, 4.5]
#%%
step_rule = MCG.ThreeTermPRP()
callback = MCG.GatheringCallback(Float64)
opt_res = MCG.optimize(x0, mop, callback; step_rule, max_iter=50)