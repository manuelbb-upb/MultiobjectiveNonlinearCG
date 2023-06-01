# This file is meant to be parsed by Literate.jl #src
if !(joinpath(@__DIR__, "..", "..") in LOAD_PATH) #src
    pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "..")) #src
end #src
using Pkg #src
Pkg.activate(@__DIR__) #src

SAVEFIGS=true #src
SAVEPATH=joinpath(@__DIR__, "..", "slides", "assets") #src

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
Let's plot the Rosenbrock function's contours to see the valley.
We use `CairoMakie` for plotting.
=#
using CairoMakie
using Printf
# Moreover, there are some custom definitions in an external file:
include(joinpath(@__DIR__, "makie_theme.jl"))
nothing #hide

## Define the function:
f(x1, x2, a, b) = b * (x2 - x1^2)^2 + (a - x1)^2

# I also want to show the drawbacks of using Standard Gradient Descent,
# so let's do an optimization run.
# The gradient of the Rosenbrock function is known analitically (see above):
df(x1, x2, a, b) = [
    -4*b*(x2-x1^2)*x1 - 2*(a-x1),
    2*b*(x2-x1^2)
]

# For (multi-objective) optimization, we use our package:
import MultiobjectiveNonlinearCG as M

# Setup objective for single objective optimization:
objf_so = x -> [f(x[1], x[2], 1.0, 100),]
jacT_so = x -> reshape(df(x[1], x[2], 1.0, 100), :, 1)

## cache for gathering iteration data:
cache_so = M.GatheringCallbackCache(Float64)
callbacks_so = [M.GatheringCallback(cache_so),]

descent_rule_so = M.SteepestDescentRule(M.StandardArmijoRule())
_ = M.optimize([-1.8, 0.0], objf_so, jacT_so; 
    descent_rule=descent_rule_so, max_iter=100, callbacks=callbacks_so)

# Let's proceed to plotting:
## I am using a `let` block here to not pollute the global scope ...
let
    set_theme!(DOC_THEME2) #hide
    ## evaluation range
    X1 = LinRange(-2, 2, 100)
    X2 = X1
    ## define function for ``a=1, b=100``
    F = (x1, x2) -> f(x1, x2, 1, 100)
    
    ## initialize figure
    fig = Figure(;)
    
    ## set global title
    Label(fig[1, 1:4], "Rosenbrock Function."; fontsize=60f0)
    
    ## plot filled contours in left axis
    ax1 = Axis(fig[2,2]; aspect=1, title=L"f(x_1, x_2)", ylabelvisible=false)
    c = contourf!(ax1, X1, X2, F)
    scatter!(ax1, (1, 1); color=DOC_COLORS[:min])
    ## and also give it a colorbar
    Colorbar(fig[2,1], c; 
        flipaxis=false, ticks=[0, 1e3, 2e3, 3e3], 
        tickformat=nums->[@sprintf("%dK",n/1000) for n in nums]
    )

    ## plot log contours in right axis
    ax2 = Axis(fig[2,3], aspect=1, title=L"\log(f(x_1, x_2)))")
    c = contourf!(ax2, X1, X2, log10 ∘ F)
    scatter!(ax2, (1, 1); color=DOC_COLORS[:min])
    Colorbar(fig[2,4], c;
        ticks=[-2, -1, 0, 1, 2, 3], 
    )

    ## plot iterates
    x_it = Tuple.(cache_so.x_arr)
    xlims!(ax1, (-2,2))
    ylims!(ax1, (-2,2))
    xlims!(ax2, (-2,2))
    ylims!(ax2, (-2,2))
    scatterlines!(ax1, x_it; markersize=15f0,color=DOC_COLORS[:sd])
    scatterlines!(ax2, x_it; markersize=15f0,color=DOC_COLORS[:sd])

    linkaxes!(ax1, ax2)

    ## make colorbars have nice size
    rowsize!(fig.layout, 2, Aspect(2,1))

    ## display plot
    SAVEFIGS && save(joinpath(SAVEPATH, "so_rosenbrock.png"), fig) #src
    fig
end
#%% #src
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

# ### Plotting the Pareto-Set
# Let's fix some parameters and define our objectives:
#%% #src
a1 = 1.0
a2 = 2.0
b1 = b2 = 100
f1(x1, x2) = f(x1, x2, a1, b1)
f2(x1, x2) = f(x1, x2, a2, b2)

# Again, I do the plotting in a `let` block, because I might want 
# to use other ranges later on:
let
    set_theme!(DOC_THEME2) #hide

    ## evaluation range
    X1 = LinRange(-4.1, 4.1, 100)
    X2 = X1

    ## initialize figure
    fig = Figure(;)
    ax1 = Axis(fig[2,2]; aspect=1, title=L"f_1(x_1, x_2)", ylabelvisible=false)
    ax2 = Axis(fig[2,3], aspect=1, title=L"f_2(x_1, x_2)")

    linkaxes!(ax1, ax2)

    ## set global title
    Label(fig[1, 1:4], "2 Rosenbrock Functions."; fontsize=60f0)

    ## plot filled contours in left axis
    c = contourf!(ax1, X1, X2, f1)
    ## and also give it a colorbar
    Colorbar(fig[2,1], c; 
        flipaxis=false, 
        tickformat=nums->[@sprintf("%.1fK",n/1000) for n in nums]
    )

    ## plot log contours in right axis
    c = contourf!(ax2, X1, X2, f2)
    Colorbar(fig[2,4], c;
            tickformat=nums->[@sprintf("%.1fK",n/1000) for n in nums]
    )

    ## plot Pareto set into both axes
    A = LinRange(a1, a2, 100)
    lines!(ax1, A, A.^2; color=DOC_COLORS[:PS], label="PS")
    lines!(ax2, A, A.^2; color=DOC_COLORS[:PS], label="PS")

    ## plot minima
    scatter!(ax1, (a1, a1^2); color=DOC_COLORS[:min])
    scatter!(ax2, (a2, a2^2); color=DOC_COLORS[:min])

    ## make colorbars have nice size
    rowsize!(fig.layout, 2, Aspect(2,1))

    SAVEFIGS && save(joinpath(SAVEPATH, "bi_rosenbrock.png"), fig) #src
    ## display plot
    fig
end

#%% #src

# ## Optimization

## define gradients:
df1(x1, x2) = df(x1, x2, a1, b1)
df2(x1, x2) = df(x1, x2, a2, b2)

# We need a vector-vector objective and a vector-matrix transposed
# jacobian:
objf(x) = [f1(x[1], x[2]), f2(x[1], x[2])]
jacT(x) = hcat(df1(x[1], x[2]), df2(x[1], x[2]))

# Reproducibly initialize starting site:
import Random
Random.seed!(2178)
x0 = -4 .+ 8 .+ rand(2)

# Setup an optimization run to gather information in `cache`
function do_experiment(x0; descent_rule, max_iter=100)
    cache = M.GatheringCallbackCache(Float64)
    callbacks = vcat(M.DEFAULT_CALLBACKS, M.GatheringCallback(cache))
    _ = M.optimize(x0, objf, jacT; descent_rule, max_iter, callbacks)
    return cache
end

# Compare Steepest Descent and some CG directions.
# To make repeated tasks easier, I set up a list of experiments to do:
experiment_settings = [
    ("SD", M.SteepestDescentRule(M.StandardArmijoRule()), :sd,),
    ("SDm ", M.SteepestDescentRule(M.ModifiedArmijoRule()), :sdM,),
    ("PRP3", M.PRP(M.ModifiedArmijoRule(), :sd), :prp3,),
    ("PRP2", M.PRPGradProjection(M.ModifiedArmijoRule(), :sd), :prpOrth,),
    ("FR", M.FRRestart(M.ModifiedArmijoRule(), :sd), :frRestart,),
];

# In single-objective optimization, some people try to improve the convergence 
# speed by minimizing a quadratic model along the CG direction.
push!(experiment_settings,  ("SDsz ", M.SteepestDescentRule(M.StandardArmijoRule(;σ_init=M.QuadApprox())), :sdSZ,))
push!(experiment_settings,  ("PRP3sz ", M.PRP(M.ModifiedArmijoRule(;σ_init=M.QuadApprox()), :sd), :prp3SZ,))

experiment_results = []
for (_, descent_rule, _) in experiment_settings
    cache = do_experiment(x0; descent_rule)
    ## Stacking iteration sites into matrices makes calculation
    ## of axis bounds easier:
    X = reduce(hcat, cache.x_arr)
    push!(experiment_results, X)
end

# After the experiments have run, I need a helper function 
# to determine the plotting limits:
function get_lims(Xs...; margin=0.1)
    xlims = [Inf, -Inf]
    ylims = [Inf, -Inf]
    
    for X in Xs
        _xlims, _ylims = extrema(X, dims=2)
        xlims[1] = min(xlims[1], _xlims[1])
        xlims[2] = max(xlims[2], _xlims[2])
        ylims[1] = min(ylims[1], _ylims[1])
        ylims[2] = max(ylims[2], _ylims[2])
    end
    
    if margin > 0
        xw = xlims[2] - xlims[1]
        xlims[1] -= margin * xw
        xlims[2] += margin * xw
        
        yw = ylims[2] - ylims[1]
        ylims[1] -= margin * yw
        ylims[2] += margin * yw
    end

    return Tuple(xlims), Tuple(ylims)
end

# Finally, the plotting is done much the same as before.
# First, plot all trajectories into a decision space plot:
(x1_min, x1_max), (x2_min, x2_max) = get_lims(experiment_results...)
let
    set_theme!(DOC_THEME2) #hide
    ## evaluation range 
    X1 = LinRange(x1_min, x1_max, 100)
    X2 = LinRange(x2_min, x2_max, 100)
    ## initialize figure
    fig = Figure(;)
    ax1 = Axis(fig[2,2]; aspect=1, title=L"f_1(x_1, x_2)", ylabelvisible=false)
    ax2 = Axis(fig[2,3], aspect=1, title=L"f_2(x_1, x_2)")

    linkaxes!(ax1, ax2)

    ## set global title
    Label(fig[1, 1:4], L"2 Rosenbrock Functions ($a_1=1, a_2=2$)."; fontsize=60f0)

    ## plot filled contours in left axis
    c = contourf!(ax1, X1, X2, f1)
    ## and also give it a colorbar
    Colorbar(fig[2,1], c; 
        flipaxis=false, 
        tickformat=nums->[@sprintf("%.1fK",n/1000) for n in nums]
    )

    ## plot log contours in right axis
    c = contourf!(ax2, X1, X2, f2)
    Colorbar(fig[2,4], c;
            tickformat=nums->[@sprintf("%.1fK",n/1000) for n in nums]
    )

    ## plot Pareto set into both axes
    A = LinRange(a1, a2, 100)
    lines!(ax1, A, A.^2; color=DOC_COLORS[:PS], label="PS")
    lines!(ax2, A, A.^2; color=DOC_COLORS[:PS], label="PS")

    ## make colorbars have nice size
    rowsize!(fig.layout, 2, Aspect(2,1))

    ## trajectories of optimization
    for (ci, settings) = enumerate(experiment_settings[1:5])
        label, _, prop_key = settings
        X = experiment_results[ci]
        label *= "($(size(X,2)))"
        scatterlines!(ax1, X; 
            markersize=10f0, label, color=DOC_COLORS[prop_key], linestyle=DOC_LSTYLES[prop_key])
        scatterlines!(ax2, X; 
            markersize=10f0, label, color=DOC_COLORS[prop_key], linestyle=DOC_LSTYLES[prop_key])
    end
    ## activate legend
    axislegend(ax1; position=:lb)

    ## display plot
    SAVEFIGS && save(joinpath(SAVEPATH, "rosenbrock_trajectories.png"), fig) #src
    fig
end
# The plot looks a bit cluttered, so do individual plots, too
function experiment_single_plot(ind)
    global x1_min, x1_max, x2_min, x2_max
    global experiment_settings, experiment_results
    
    set_theme!(DOC_THEME2) #hide
    ## evaluation range 
    X1 = LinRange(x1_min, x1_max, 100)
    X2 = LinRange(x2_min, x2_max, 100)
    ## initialize figure
    fig = Figure(;)
    ax1 = Axis(fig[2,2]; aspect=1, title=L"f_1(x_1, x_2)", ylabelvisible=false)
    ax2 = Axis(fig[2,3], aspect=1, title=L"f_2(x_1, x_2)")

    linkaxes!(ax1, ax2)

    ## set global title
    Label(fig[1, 1:4], L"2 Rosenbrock Functions ($a_1=1, a_2=2$)."; fontsize=60f0)

    ## plot filled contours in left axis
    c = contourf!(ax1, X1, X2, f1)
    ## and also give it a colorbar
    Colorbar(fig[2,1], c; 
        flipaxis=false, 
        tickformat=nums->[@sprintf("%.1fK",n/1000) for n in nums]
    )

    ## plot log contours in right axis
    c = contourf!(ax2, X1, X2, f2)
    Colorbar(fig[2,4], c;
            tickformat=nums->[@sprintf("%.1fK",n/1000) for n in nums]
    )

    ## plot Pareto set into both axes
    A = LinRange(a1, a2, 100)
    lines!(ax1, A, A.^2; color=DOC_COLORS[:PS], label="PS")
    lines!(ax2, A, A.^2; color=DOC_COLORS[:PS], label="PS")

    ## make colorbars have nice size
    rowsize!(fig.layout, 2, Aspect(2,1))

    ## trajectories of optimization
    for ci in ind
        label, _, prop_key = experiment_settings[ci]
        X = experiment_results[ci]
        label *= "($(size(X,2)))"
        scatterlines!(ax1, X; 
            markersize=10f0, label, color=DOC_COLORS[prop_key], linestyle=DOC_LSTYLES[prop_key])
        scatterlines!(ax2, X; 
            markersize=10f0, label, color=DOC_COLORS[prop_key], linestyle=DOC_LSTYLES[prop_key])
    end
    ## activate legend
    axislegend(ax1; position=:lb)

    ## display plot
    SAVEFIGS && save(joinpath(SAVEPATH, "rosenbrock_trajectory_$(join( [first(experiment_settings[i]) for i in ind], "_")).png"), fig) #src
    fig
end

experiment_single_plot([1,])
experiment_single_plot([2,])
experiment_single_plot([3,])
experiment_single_plot([4,])
experiment_single_plot([5,])
experiment_single_plot([6,])
experiment_single_plot([7,])

#%% #src
# ### Criticality Plot

# The behavior of steepest descent becomes more apparent,
# when looking at the criticality as defined by
# the norm of the steepest descent direction.

function crit_map(x1, x2)
    δ = M.frank_wolfe_multidir_dual((df1(x1,x2), df2(x1,x2)))
    return sqrt(sum(δ.^2))
end

# The plot reveals a near critical parabola:
function experiment_crit_plot(ind)
    global x1_min, x1_max, x2_min, x2_max
    global experiment_settings, experiment_results
    
    set_theme!(DOC_THEME) #hide
    ## evaluation range (more fine-grained here)
    X1 = LinRange(x1_min, x1_max, 200)
    X2 = LinRange(x2_min, x2_max, 200)
    
    fig = Figure()
    ax = Axis(fig[1,1]; title=L"\log(\Vert\mathbf{\delta}\Vert^2)")
    
    c = heatmap!(ax, X1, X2, log10 ∘ crit_map)
    Colorbar(fig[1,2], c)
    
    ## plot Pareto set into both axes
    A = LinRange(a1, a2, 100)
    lines!(ax, A, A.^2; color=DOC_COLORS[:PS], label="PS", linewidth=5f0) 
    
    ## trajectories of optimization
    for ci in ind
        label, _, prop_key = experiment_settings[ci]
        X = experiment_results[ci]
        label *= "($(size(X,2)))"
        ## shadow for distinguishable lines
        lines!(ax, X; color=:black, linewidth=8f0)
        scatterlines!(ax, X; 
            markersize=18f0, label, color=DOC_COLORS[prop_key], linestyle=DOC_LSTYLES[prop_key],
            strokecolor=:black, strokewidth=1f0
        )
    end

    axislegend(ax)
    SAVEFIGS && save(joinpath(SAVEPATH, "rosenbrock_crit_trajectory_$(join( [first(experiment_settings[i]) for i in ind], "_")).png"), fig) #src
    fig
end
experiment_crit_plot([1,])
experiment_crit_plot([2,])
experiment_crit_plot([3,])
experiment_crit_plot([4,])
experiment_crit_plot([5,])

# ## Statistics

using Statistics
# The next experiment fixes a set of `num_runs` starting points and tests
# various methods against each other.
# We limit the number of iterations and stop only if we are completely critical.
num_runs = 100
max_iter = 20
rand_x0 = () -> -50 .+ 100 .* rand(2)
X0 = [ rand_x0() for i=1:num_runs ]

# For now, we are only interested in the iteration sites.
# We extract them from `cache` and store them in `X_it`:
function do_runs(descent_rule; max_iter=20)
    X_it = Vector{Vector{Vector{Float64}}}()
    for x0 in X0
        cache = M.GatheringCallbackCache(Float64)
        callbacks = [M.CriticalityStop(;eps_crit=-Inf), M.GatheringCallback(cache)]
        _ = M.optimize(x0, objf, jacT; descent_rule, max_iter, callbacks)
        
        x_arr = copy(cache.x_arr)
        last_x = last(x_arr)
        for _=length(x_arr)+1:max_iter
            push!(x_arr, last_x)
        end
        push!(X_it, x_arr)
    end
    return X_it
end

# Perform all the experiments with previous settings:
experiment_results2 = [
    do_runs(descent_rule) for (_, descent_rule, _) in experiment_settings 
];

# For every run, we want to compute the criticality ``\|\symbf{δ}\|`` at each point:
crit_map(x) = crit_map(x[1], x[2])
function crit_index(X_it)
    crit_vals = crit_map.(X_it)
    crit_vals ./= first(crit_vals) # “normalize” with respect to first value
    return crit_vals
end

# Finally, we do some statistics.
function consume_results(X_it)
    C = mapreduce( crit_index, hcat, X_it)
    μ = vec(mean(C; dims=2))
    σ = vec(std(C; dims=2))
    quart1 = quantile.(eachrow(C), 0.25)
    quart2 = vec(median(C; dims=2))
    quart3 = quantile.(eachrow(C), 0.75)
    return μ, σ, quart1, quart2, quart3
end

# Run and plot:
function compare_crit_plot(ind)
    global experiment_settings, experiment_results2
    
    fig = Figure()
    ax = Axis(fig[1,1]; 
        yscale=log10, xlabel=L"k", 
        ylabel=L"\Vert\delta(x^{(k)})\Vert/\Vert\delta_0\Vert", 
        title="2 Rosenbrocks, $(num_runs) runs in [-50, 50]².",
        yminorticksvisible=true, yticks=LogTicks([0, -1, -2, -3, -4, -5, -6, -7]),
        yminorticks=IntervalsBetween(10),
        limits = (nothing, (5e-8, 1.2))
    )
    for ci in ind
        X_it = experiment_results2[ci]
        label, _, prop_key = experiment_settings[ci]

        μ, σ, q1, q2, q3 = consume_results(X_it)
        it = 0:length(μ)-1
        lines!(ax, it, μ; label = "μ " * label, color=DOC_COLORS[prop_key])
        lines!(ax, it, q2; label = "median", color=DOC_COLORS[prop_key], linestyle=:dash)
        band!(ax, it, q1, q3; label = "50 %", color=(DOC_COLORS[prop_key], 0.2))
    end
    axislegend(ax; patchsize=(40f0, 20f0))

    SAVEFIGS && save(joinpath(SAVEPATH, "rosenbrock_statistics_$(join( [first(experiment_settings[i]) for i in ind], "_")).png"), fig) #src
    fig
end

compare_crit_plot([1,3])
compare_crit_plot([1,4])
compare_crit_plot([1,5])