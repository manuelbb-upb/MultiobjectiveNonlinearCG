# This file is meant to be parsed by Literate.jl            #src
# Activate `docs` environment:                              #src
using Pkg                                                   #src
Pkg.activate(joinpath(@__DIR__, "..", ".."))                #src
#%%#src
# Import dependencies. Optimizer …
import MultiobjectiveNonlinearCG as MCG
# … and Plotting library
using CairoMakie
includet("PlotHelpers.jl") #src
#md include("PlotHelpers.jl")
PLOT_DIR = @__DIR__
#%%#src
# # Problem Setup
# Define 2D2D problem.

#=
The two paraboloids problem in 2D reads as
```math
\min_{\symbf{x} ∈ ℝ^2}
 \begin{bmatrix}
    f_1(\symbf{x})
    \\
    f_2(\symbf{x})
 \end{bmatrix}
 =
 \begin{bmatrix}
 (x₁ - a)^2 + (x₂ - a)^2
 \\
 (x₁ + a)^2 + (x₂ + a)^2
 \end{bmatrix}
 .
```
The Pareto-Set is the line connecting the individual 
minima, i.e.,
```math
\mathcal P_S
  =
    \left\{
      \symbf{x} ∈ ℝ^2: x₁ = x₂, x₁ ∈ [-a, a]
    \right\}.
```

It's easy to setup for `MultiobjectiveNonlinearCG`.
At the lowest level, its `optimize` method requires a starting point, its image vector, 
a modifying objective functional and a functional setting the transposed jacobian.
Let's define all that:
=#
# These functions are set up to work for plotting:
f(x1, x2, a) =  (x1 + a)^2 + (x2 + a)^2
df(x1, x2, a) = 2 .* [x1 + a, x2 + a]
a1 = -1
a2 = +1
f1(x1, x2) = f(x1, x2, a1)
f2(x1, x2) = f(x1, x2, a2)
df1(x1, x2) = df(x1, x2, a1)
df2(x1, x2) = df(x1, x2, a2)

# We make them compatible with the optimizer:
objectives! = function(y, x)
    x1, x2 = x[1:2]
    y[1] = f1(x1, x2)
    y[2] = f2(x1, x2)
end

jac! = function(Dy, x)
    x1, x2 = x[1:2]
    Dy[1, :] .= df1(x1, x2)
    Dy[2, :] .= df2(x1, x2)
end
# And finally set up the optimization problem:
dim_in = 2
dim_out = 2
mop = MCG.BasicMOP(; dim_in, dim_out, objectives!, jac!)
#%%#src
# Starting point `x0` …
x0 = [-3.14, 0]
# … and `step_rule` for descent directions.
# We request metadata gathering to plot the descent dirctions:
step_rule = MCG.SteepestDescentDirection(;
    set_metadata = true,
    sz_rule = MCG.ArmijoBacktracking(; factor=.75)
)
# Now, the callback should automatically gather `StepMeta`.
callback = MCG.GatheringCallback(Float64)
# We pass it to `optimize`. Note, that we extract `cback`, because the callback
# is changed by initialization:
opt_res = MCG.optimize(x0, mop, callback; step_rule)
cback = opt_res.callback;
import .PlotHelpers: get_limits
# # Plots
# Helper function to extract iteration and direction matrices from `cback`:
function extract_matrices(cback, N=typemax(Int), i0=1)
    xmat = MCG._x_mat(cback)
    N = min(N, size(xmat, 2))
    xmat = xmat[:, i0:N]
    dirs = reduce(hcat, (sm.direction_matrix[:, 1] for sm in cback.step_meta[i0:N]))
    xdmat = xmat .+ dirs
    szs = reduce(hcat, (sm.sz[] for sm in cback.step_meta[i0:N]))
    return xmat, dirs, xdmat, szs
end
# A function to plot iteration steps as arrows into `ax`:
function plot_iteration!(
    ax, xmat, dirs, szs, i;
    it_color=:black, it_alpha=1,
    it_color_ligth=:black, it_alpha_light=it_alpha/2,
    make_dot = false
)
    x_it = xmat[:, i]
    _sd = dirs[:, i]
    sz = szs[i]
    sd = sz * _sd
    
    if make_dot
        scatter!(
            ax, [Point(x_it[1:2]...),]; 
            color=(it_color, it_alpha),
            markersize=10f0
        )
    end
    arrows!(
        ax, [Point(x_it[1:2]...),], [Point(_sd[1:2]...)]; 
        color=(it_color_ligth, it_alpha_light),
        linestyle=(:dash, :dense),
        arrowsize=10f0,
        linewidth=2f0
    )
    
    arrows!(
        ax, [Point(x_it[1:2]...),], [Point(sd[1:2]...)]; 
        color=(it_color, it_alpha),
        linewidth=2.5f0,
        arrowsize=15f0,
    )
    
end
# Function to plot the whole figure:
function iteration_figure(
    x1_lims, x2_lims, 
    xmat, dirs, xdmat, szs,
    I=Int[]; 
    it_colors=:black,
    it_alphas=1.0,
    kwargs...
)
    fig = Figure(; 
        size=(500, 500),
        fontsize=20f0
    )
    ax = Axis(fig[1,1]; aspect=AxisAspect(1))    

    X1 = range(x1_lims[1], x1_lims[2], 100)
    X2 = range(x2_lims[1], x2_lims[2], 100)

    lines!(ax, [(a1, a1), (a2, a2)]; 
        color=PlotHelpers.upbLimeGreen,
        linewidth=4f0
    )
    contour!(ax, X1, X2, f1; 
        color=(PlotHelpers.upbUltraBlue, .75),
        linewidth=2f0)
    contour!(ax, X1, X2, f2; 
        color=(PlotHelpers.upbFuchsiaRed, .75),
        linewidth=2f0)

    if !isa(it_colors, AbstractVector)
        it_colors = fill(it_colors, length(I))
    end
    if !isa(it_alphas, AbstractVector)
        it_alphas = fill(it_alphas, length(I))
    end
    for (_i, i) in enumerate(I)
        plot_iteration!(ax, xmat, dirs, szs, i;
            kwargs...,
            it_color=it_colors[_i],
            it_alpha=it_alphas[_i]
        )
    end
    fig
end
# Call plotting functions and create figure:
xmat, dirs, xdmat, szs = extract_matrices(cback)
## global canvas limits with margin
x1_lims, x2_lims = get_limits(
    xmat, xdmat,
    fill(-2, dim_in, 1), fill(2, dim_in, 1)
)

N = min(size(xmat, 2), 10) # number of iterations to plot
fig = iteration_figure(
    x1_lims, x1_lims, xmat, dirs, xdmat, szs, 1:N;
    make_dot=true, it_alphas=range(1, .2, N)
)

if !isnothing(PLOT_DIR)
    save(joinpath(PLOT_DIR, "backtracking_algo.png"), fig)
end
fig