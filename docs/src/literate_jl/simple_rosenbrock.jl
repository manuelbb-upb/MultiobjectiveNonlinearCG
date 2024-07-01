# This file is meant to be parsed by Literate.jl            #src
# Activate `docs` environment:                              #src
using Pkg                                                   #src
Pkg.activate(joinpath(@__DIR__, "..", ".."))                #src
#%%#src
# # Dependencies
import MultiobjectiveNonlinearCG as MCG
using CairoMakie
using LaTeXStrings
includet("PlotHelpers.jl")  #src
#md includet("PlotHelpers.jl")
import .PlotHelpers: get_limits
import UnPack: @unpack
import HaltonSequences: HaltonPoint
import Random: randperm
import Logging: Debug
import Printf: @sprintf
# # Plotting Theme
# Let's define the theming right away:
custom_theme = Theme(
    size = (1920, 1080),
    fontsize = 45f0,
    linewidth = 5f0,
    markersize = 40f0,
    Scatter = (
        strokewidth = 1f0,
        strokecolor = :white,
    ),
    colormap=:acton,
    Label = (
        fontsize = 50f0,
    ),
    palette = (color = PlotHelpers.upb_palette, ),
    Axis = (
        xlabelsize = 50f0,
        ylabelsize = 50f0,
    ),
)
## set to `nothing` to disable saving of plots
PLOT_DIR = @__DIR__

#%%#src
# # Two Rosenbrock Functions

# In single-objective optimization, the 
# [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) is a 
# prominent example of an objective function that is hard for vanilla steepest descent.
# It has a flat valley around its global optimum, so the gradients become small.
#
#=
In this example, we are going to look at a bi-objective problem constructed from 
parameterized Rosenbrock functions.
The Rosenbrock function 
```math
f_{a,b}(x_1, x_2) = b( x_2 - x_1^2 )^2 + (a - x_1)^2
```
has its global minimum at ``(a, a^2)`` with ``f(a, a^2) = 0``.
Its gradient is
```math
\nabla f_a(\symbf{x}) = 
    \begin{bmatrix}
        -4b(x_2 - x_1^2)x_1 - 2(a-x_1) \\
        2b(x_2 - x_1^2)
    \end{bmatrix}
= 
    \begin{bmatrix}
        4bx_1^3 -4b x_1 x_2 + 2x_1 - 2a
        \\
        -2b x_1^2 + 2b x_2
    \end{bmatrix}.
```
=#
## define the function
f_rb(x1, x2, a, b) = b * (x2 - x1^2)^2 + (a - x1)^2

## and its gradient
df_rb(x1, x2, a, b) = [
    -4*b*(x2-x1^2)*x1 - 2*(a-x1),
    2*b*(x2-x1^2)
]
#%%#src
#=
## Bi-Objective Problem
Now let ``a_1, a_2 \in ℝ`` and ``b_1 > 0, b_2 > 0``,
define ``f_1 = f_{a_1, b_1}`` 
and ``f_2 = f_{a_2, b_2}``, and consider
```math
\min_{\symbf{x}\in ℝ^2} 
    \begin{bmatrix}
        f_1(\symbf x)
        \\
        f_2(\symbf x)
    \end{bmatrix}.
```
The KKT conditions read
```math
α 
    \begin{bmatrix}
        4b_1x_1^3 - 4b_1 x_1 x_2 + 2x_1 - 2a_1
        \\
        -2b_1 x_1^2 + 2b_1 x_2
    \end{bmatrix}
+ (1-α)
    \begin{bmatrix}
        4b_2x_1^3 - 4b_2 x_1 x_2 + 2x_1 - 2a_2
        \\
        -2b_2 x_1^2 + 2b_2 x_2
    \end{bmatrix}
=0, \qquad α\in [0,1].
```
The second equation gives
```math
\begin{aligned}
-2(αb_1 + (1-α)b_2)x_1^2 + 2(αb_1 + (1-α)b_2)x_2 &= 0 \\
⇔ \quad x_2 &= x_1^2.
\end{aligned}
```
With this, the terms ``4b_1x_1^3 - 4b_1 x_1 x_2`` and 
``4b_2x_1^3 - 4b_2 x_1 x_2`` cancel out and the first equation becomes:
```math
\begin{aligned}
0 &= 2x_1 - 2αa_1 -2(1-α)a_2\\
⇔ \; x_1 &= ( αa_1 + (1-α) a_2)
\end{aligned}
```
Thus, the Pareto-critical set is a parabolic segment 
with ``x_1`` ranging from ``a_1`` to ``a_2`` and ``x_2 = x_1^2``.
=#

# ## Preparation 
# Let's fix some parameters and define our objectives:
#%% #src
a1 = 1.0
a2 = 2.0
b1 = b2 = 100
f1_rb(x1, x2) = f_rb(x1, x2, a1, b1)
f2_rb(x1, x2) = f_rb(x1, x2, a2, b2)

## define gradients:
df1_rb(x1, x2) = df_rb(x1, x2, a1, b1)
df2_rb(x1, x2) = df_rb(x1, x2, a2, b2)

# For the parameters from above, the pareto set is easy to compute:
pset_x1 = range(a1, a2, 100)
pset_x2 = pset_x1.^2
pset = vcat(pset_x1', pset_x2')
# In general, criticality is inferred from th norm of the steepest descent direction:
function crit_map(x1, x2)
    Dy = [df1_rb(x1, x2)'; df2_rb(x1, x2)']
    α = MCG.frank_wolfe_multidir_dual(Dy)
    δ = Dy'*α
    return sqrt(sum(δ.^2))
end
# Let us plot the problem surface for illustration:
let
    with_theme(custom_theme; markersize=25f0) do
        fig = Figure(; size=(1000, 1800))
        
        Label(fig[1:2, 0], L"a_1=%$(a1),\;a_2=%$(a2),\;b_1=%$(b1),\;b_2=%$(b2)"; rotation=π/2)

        ax1 = Axis(fig[1,1]; 
            aspect=1, xlabel=L"x_1", ylabel=L"x_2",
            xaxisposition=:top,
        )
        tickformatter = vals -> [L"10^{ %$(round(Int, log10(v))) }" for v in vals]
        ax2 = Axis(fig[2,1]; 
            aspect=1, xlabel=L"f_1", ylabel=L"f_2", 
            xscale=log10, yscale=log10,
            xticks=[10^i for i=0:5],
            yticks=[10^i for i=0:5],
            xtickformat=tickformatter,
            ytickformat=tickformatter,
            xminorticks=IntervalsBetween(10),
            yminorticks=IntervalsBetween(10),
            xminorticksvisible=true,
            yminorticksvisible=true,
        )

        x1_lims = (-1.0, 5.0)
        x2_lims = (-1.0, 5.0)

        X1 = range(x1_lims..., 200)
        X2 = range(x2_lims..., 200)

        h = heatmap!(ax1, X1, X2, crit_map; colorscale=log10)
        lines!(ax1, pset; color=PlotHelpers.pset_color())
        Colorbar(fig[1,2], h)

        x1_grid = range(x1_lims..., 20)
        x2_grid = range(x2_lims..., 20)
        x = reduce(hcat, [x1, x2] for x1=x1_grid, x2=x2_grid)
        fmap = x -> [f1_rb(x[1], x[2]), f2_rb(x[1], x[2])]
        y = mapreduce(fmap, hcat, eachcol(x))
        yopt = mapreduce(fmap, hcat, eachcol(pset))
        scatter!(ax1, x; colormap=:lisbon, color=axes(x, 2))
        lines!(ax2, yopt; color=PlotHelpers.pset_color())
        scatter!(ax2, y; colormap=:lisbon, color=axes(x, 2))
        rowsize!(fig.layout, 1, Aspect(1, 1))

        if !isnothing(PLOT_DIR)
            save(joinpath(PLOT_DIR, "rosenbrock_values.png"), fig)
        end
        fig
    end
end
#%%#src 
# In-place functions for optimization:
objectives!_rb = function (y, x)
    y[1] = f1_rb(x[1], x[2])
    y[2] = f2_rb(x[1], x[2])
end

jac!_rb = function (Dy, x)
    Dy[1, :] .= df_rb(x[1], x[2], 1, 100)
    Dy[2, :] .= df_rb(x[1], x[2], 2, 100)
end
# Build the MOP:
mop = MCG.BasicMOP(;
    dim_in = 2,
    dim_out = 2,
    objectives! = objectives!_rb,
    jac! = jac!_rb
)
# ## Comparison of CG with Steepest Descent
# Prepare initial sites.
# Halton Points are well spread.
lb = [-4.0; -4.0;;]
ub = [6.0; 6.0;;]
n_points = 10
x0 = mapreduce(x -> lb .+ (ub .- lb) .* x, hcat, HaltonPoint(2; length=n_points))
x0 = x0[:, randperm(n_points)] # nicer color cycling while plotting
step_rule_dict = Dict{Symbol, Any}()
# Baseline for comparison:
step_rule_dict[:sd] = MCG.SteepestDescentDirection(;
    sz_rule=MCG.ArmijoBacktracking(;is_modified=Val(false))
)
# Settings for CG Directions:
sz_rule_cg = MCG.ArmijoBacktracking(;is_modified=Val(true))
step_rule_dict_cg = Dict(
    :frr => MCG.FletcherReevesRestart(; wolfe_constant = .99, sz_rule = sz_rule_cg),
    :frf => MCG.FletcherReevesFractionalLP(; sz_rule = sz_rule_cg),
    :prp3 => MCG.PRP3(; sz_rule = sz_rule_cg),
    :prpc => MCG.PRPConeProjection(; sz_rule = sz_rule_cg)
)
merge!(step_rule_dict, step_rule_dict_cg)
results = Dict{Symbol, Any}()
# Function to perform the optimization runs and store results:
function run_optimization!(
    results, x0_mat, step_rule_key, step_rule; max_iter=30, log_level=Debug
)
    global mop
    it_matrices = Vector{Matrix{Float64}}()
    for x0 in eachcol(x0_mat)
        callback = MCG.GatheringCallback(Float64)
        opt_res = MCG.optimize(x0, mop, callback; step_rule, max_iter, log_level)
        push!(it_matrices, MCG._x_mat(opt_res.callback))
    end
    results[step_rule_key] = it_matrices
    return results
end
# Actually do the experiments:
max_iter_comparisons = 30
for (srk,sr) in pairs(step_rule_dict)
    run_optimization!(results, x0, srk, sr; max_iter = max_iter_comparisons)
end
#%%#src
# ### Plotting
# We want to compare every method against steepest descent and can do some 
# work beforehand:
x_it_sd = results[:sd]
_x1_lims, _x2_lims = PlotHelpers.get_limits(pset, lb, ub, x_it_sd...)
_lims = vcat( collect(_x1_lims)', collect(_x2_lims)' )
# Main plotting function:
function comparison_fig(
    results, step_rule_key, base_lims; 
    I = nothing, alpha=.4f0
)
    global x_it_sd, step_rule_dict
    
    x_it_cg = results[step_rule_key]
    step_rule_sd = step_rule_dict[:sd]
    step_rule_cg = step_rule_dict[step_rule_key]

    fig, ax1, ax2 = comparison_base_fig(x_it_cg, base_lims, step_rule_sd, step_rule_cg)
    comparison_trajectories!(ax1, ax2, x_it_sd, x_it_cg, I; alpha)
    return fig
end
# Helper function to make base figure:
function comparison_base_fig(x_it, base_lims, step_rule_sd, step_rule_cg)
    global step_rule_dict, max_iter_comparisons, custom_theme
    
    x1_lims, x2_lims = PlotHelpers.get_limits(base_lims, x_it...)
    limits = (x1_lims, x2_lims)
    
    X1 = range(x1_lims..., 200)
    X2 = range(x2_lims..., 200)
    C = [crit_map(x1, x2) for x1=X1, x2=X2]

    return with_theme(custom_theme) do
        fig = Figure()
        Label(fig[0, 1:2], 
        L"\text{Bi-Rosenbrock Problem},\; a_1 = %$(a1),\;a_2=%$(a2),\;\text{max iter}=%$(max_iter_comparisons).")
        
        ax1 = Axis(
            fig[1,1]; 
            aspect=1, limits, xlabel=L"x_1", ylabel=L"x_2", 
            title=PlotHelpers.step_rule_label(step_rule_sd)
        )
        ax2 = Axis(
            fig[1,2]; 
            aspect=1, limits, xlabel=L"x_1", ylabel=L"x_2",
            title=PlotHelpers.step_rule_label(step_rule_cg),
            ylabelvisible=false
        )

        h = heatmap!(ax1, X1, X2, C; colorscale=log10)
        heatmap!(ax2, X1, X2, C; colorscale=log10)

        Colorbar(fig[1, 3], h; width=40f0, label=L"\log10~\Vert \delta \Vert^2")
        
        lines!(ax1, pset; color=PlotHelpers.pset_color(), linewidth=10f0)
        lines!(ax2, pset; color=PlotHelpers.pset_color(), linewidth=10f0)
        
        rowsize!(fig.layout, 1, Aspect(2, 1))

        fig, ax1, ax2
    end
end
# Plotting function for trajectories into both axes:
function comparison_trajectories!(
        ax1, ax2, x_it_sd, x_it_cg, I=nothing;
        alpha=.4,
    )
    if isnothing(I)
        I = eachindex(x_it_sd)
    end
    with_theme(custom_theme;) do
        for i in I
            color_sd = color_cg = Cycled(i)
            traj_sd = x_it_sd[i]
            traj_cg = x_it_cg[i]
            ms_fin = Makie.theme(:markersize)[] * 1.4f0
            sw_fin = Makie.theme(:Scatter)[:strokewidth][] + .5f0
            lines!(ax1, traj_sd; color=color_sd, alpha)
            lines!(ax2, traj_cg; color=color_cg, alpha)
            scatter!(ax1, traj_sd[:, 1:1]; color = color_sd, marker=:circle)
            scatter!(ax1, traj_sd[:, end:end]; color = color_sd, marker=:star6, 
                markersize=ms_fin, strokewidth=sw_fin)
            scatter!(ax2, traj_cg[:, 1:1]; color = color_sd, marker=:circle)
            scatter!(ax2, traj_cg[:, end:end]; color = color_sd, marker=:star6,
                markersize=ms_fin, strokewidth=sw_fin)
        end
    end
end
#%%#src
# Make the actual figures:
if !isnothing(PLOT_DIR)
    plot_subdir = joinpath(PLOT_DIR, "sd_comparisons")
    if !isdir(plot_subdir)
        mkdir(plot_subdir)
    end
end
for step_rule_key in (:prp3, :prpc, :frr, :frf)
    fig = comparison_fig(results, step_rule_key, _lims;)

    if !isnothing(plot_subdir)
        save(joinpath(plot_subdir, "$(step_rule_key)_sd.png"), fig)
    end
    for i=1:n_points
        fig = comparison_fig(results, step_rule_key, _lims; I=[i,], alpha=1f0)
        if !isnothing(plot_subdir)
            save(joinpath(plot_subdir, "$(step_rule_key)_sd$(@sprintf("%03d", i)).png"), fig)
        end
    end
end