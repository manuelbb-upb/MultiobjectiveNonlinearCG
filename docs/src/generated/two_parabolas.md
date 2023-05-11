```@meta
EditURL = "<unknown>/docs/src/literate_jl/two_parabolas.jl"
```

This file is meant to be parsed by Literate.jl

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
x0 = [π, -2*ℯ]
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
max_iter = 1000
````

There is also a set of default stopping criteria, which can be inspected by looking
at `M.DEFAULT_CALLBACKS`.

````@example two_parabolas
x_fin, fx_fin = M.optimize(x0, fx0, objf!, jacT!; max_iter)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

