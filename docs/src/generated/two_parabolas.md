```@meta
EditURL = "<unknown>/docs/src/literate_jl/two_parabolas.jl"
```

````@example two_parabolas
include(joinpath(joinpath("..", "literate_jl"), "makie_theme.jl")) #hide
nothing #hide
````

# Two-Parabolas Example

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

````@example two_parabolas
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

# initialize starting values
import Random
x0 = 10 .* [-π, 2*ℯ]
fx0 = zeros(2)
objf!(fx0, x0)
````

## Optimize using Steepest Descent

````@example two_parabolas
import MultiobjectiveNonlinearCG as M
````

To use multi-objective steepest descent, we'll first have to decide on a stepsize procedure.
We can use a fixed stepsize, standard Armijo or modified Armijo backtracking.
For steepest descent, it's most sensible to use standard Armijo:

````@example two_parabolas
descent_rule = M.SteepestDescentRule(M.StandardArmijoRule())
````

We are nearly ready to go.
A maximum number of iterations can be provided by the `max_iter` keyword argument.

````@example two_parabolas
max_iter = 100
````

There is also a set of default stopping criteria, which can be inspected by looking
at `M.DEFAULT_CALLBACKS`.
To have a fair comparison, we reset them.
Additionally, we use a special gathering callback to later plot the iterates:

````@example two_parabolas
cache1 = M.GatheringCallbackCache(Float64)
callbacks = [
  M.CriticalityStop(; eps_crit=1e-6),
  M.GatheringCallback(cache1),
]

x_fin, fx_fin, stop_code, meta1 = M.optimize(
  x0, objf!, jacT!;
  objf_is_mutating=true,
  jacT_is_mutating=true,
  fx0, max_iter, callbacks, descent_rule,
)
meta1.num_iter[]
````

## Optimize with Modified PRP Direction
First initialize the gathering callback for this trial:

````@example two_parabolas
cache2 = M.GatheringCallbackCache(Float64)
callbacks = [
  M.CriticalityStop(; eps_crit=1e-6),
  M.GatheringCallback(cache2),
]
````

Now, test some non-linear conjugate gradient direction:

````@example two_parabolas
descent_rule = M.PRP(M.ModifiedArmijoRule(), :sd)
#descent_rule = M.FRRestart(M.ModifiedArmijoRule(), :sd)
x_fin, fx_fin, stop_code, meta2 = M.optimize(
  x0, objf!, jacT!;
  objf_is_mutating=true,
  jacT_is_mutating=true,
  fx0, max_iter, callbacks, descent_rule,
)
meta2.num_iter[]
````

## Plotting the results

````@example two_parabolas
# We use `CairoMakie` for plotting.
using CairoMakie
using Printf
set_theme!(DOC_THEME) #hide

# the `let` block is optional and used just to avoid polluting the global scope
let
  fig = Figure()
  ax = Axis(fig[1,1]; aspect=1)

  colors = Makie.wong_colors()
  lines!(ax, [(-1,-1), (1,1)]; linewidth=10f0, label="PS", color=colors[1])
  scatterlines!(ax, Tuple.(cache1.x_arr), label="sd ($(meta1.num_iter))", color=colors[2])
  scatterlines!(ax, Tuple.(cache2.x_arr), label="prp ($(meta2.num_iter))", color=colors[3])

  axislegend(ax)

  fig
end
````

Both runs finish after two iterations :)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

