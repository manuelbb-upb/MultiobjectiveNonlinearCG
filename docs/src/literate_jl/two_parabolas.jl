# This file is meant to be parsed by Literate.jl
using Pkg #src
Pkg.activate(joinpath(@__DIR__, "..", "..")) #src

# # Two-Parabolas Example

#=
The two parabolas problem in 2D reads as
```math
\min_{\symbf{x} ∈ ℝ^2}
 \begin{bmatrix}
    f_1(\symbf{x})
    \\
    f_2(\symbf{x})
 \end{bmatrix}
 =
 \begin{bmatrix}
 (x₁ - 1)^2 + (x₂ - 1)^2
 \\
 (x₁ + 1)^2 + (x₂ + 1)^2
 \end{bmatrix}
 .
```
The Pareto-Set is the line connecting the individual 
minima, i.e.,
```math
\mathcal P_S
  =
    \left\{
      \symbf{x} ∈ ℝ^2: x₁ = x₂, x₁ ∈ [-1, 1]
    \right\}.
```

It's easy to setup for `MultiobjectiveNonlinearCG`.
At the lowest level, its `optimize` method requires a starting point, its image vector, 
a modifying objective functional and a functional setting the transposed jacobian.
Let's define all that:
=#

"Evaluate the objectives at `x` and store result in `y`."
function objf!(y, x)
    y[1] = sum( (x .- 1).^2 )
    y[2] = sum( (x .+ 1).^2 )
    return nothing
end

"Evaluate objective derivatives at `x` and store transposed jacobian in `DfxT`."
function jacT!(DfxT, x)
    DfxT[:, 1] = x .- 1
    DfxT[:, 2] = x .+ 1
    DfxT .*= 2
    return nothing
end

## initialize starting values
x0 = [π, -2*ℯ]
fx0 = zeros(2)
objf!(fx0, x0)

# ## Optimize using Steepest Descent

import MultiobjectiveNonlinearCG as M

#%% #src
#=
To use multi-objective steepest descent, we'll first have to decide on a stepsize procedure.
We can use a fixed stepsize, standard Armijo or modified Armijo backtracking.
For steepest descent, it's most sensible to use standard Armijo:
=#
descent_rule = M.SteepestDescentRule(M.StandardArmijoRule())

# We are nearly ready to go.
# A maximum number of iterations can be provided by the `max_iter` keyword argument.
max_iter = 100
# There is also a set of default stopping criteria, which can be inspected by looking 
# at `M.DEFAULT_CALLBACKS`.
# To have a fair comparison, we reset them:
callbacks = [
  M.CriticalityStop(; eps_crit=1e-10),
]

x_fin, fx_fin, stop_code, meta = M.optimize(
  x0, fx0, objf!, jacT!; 
  max_iter, callbacks, descent_rule
)
meta.num_iter[]
#%% #src
# ## Optimize with Modified PRP Direction
# Test some non-linear conjugate gradient direction:
descent_rule = M.PRP(M.ModifiedArmijoRule(), :cg)
#descent_rule = M.PRP(; stepsize_rule=M.StandardArmijoRule(), criticality_measure=:cg)
x_fin, fx_fin, stop_code, meta = M.optimize(
  x0, fx0, objf!, jacT!; 
  max_iter, callbacks, descent_rule
)
meta.num_iter[]