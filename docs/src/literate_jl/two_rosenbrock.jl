# This file is meant to be parsed by Literate.jl #src
using Pkg #src
Pkg.activate(joinpath(@__DIR__, "..", "..")) #src

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

    linkaxes!(ax1, ax2)

    ## make colorbars have nice size
    rowsize!(fig.layout, 2, Aspect(2,1))

    ## display plot
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

    ## display plot
    fig
end

#%% #src

# ## Optimization

# For multi-objective optimization, we use our package:
import MultiobjectiveNonlinearCG as M

# The gradients are known analitically:
df(x1, x2, a, b) = [
    -4*b*(x2-x1^2)*x1 - 2*(a-x1),
    2*b*(x2-x1^2)
]

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

# Compare Steepest Descent and a PRP variant:
c_sd = do_experiment(x0; descent_rule=M.SteepestDescentRule(M.StandardArmijoRule()))
c_prp = do_experiment(x0; descent_rule=M.PRP(M.ModifiedArmijoRule(), :sd))

# Stacking iteration sites into matrices makes calculation
# of axis bounds easier:
X_sd = reduce(hcat, c_sd.x_arr)
X_prp = reduce(hcat, c_prp.x_arr)

# I use a helper for this
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

# Finally, the plotting is done much the same as before:
(x1_min, x1_max), (x2_min, x2_max) = get_lims(X_sd, X_prp)
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

    ## make colorbars have nice size
    rowsize!(fig.layout, 2, Aspect(2,1))

    scatterlines!(ax1, X_prp; 
        markersize=10f0, label="prp", color=DOC_COLORS[:prpMinMax], linestyle=DOC_LSTYLES[:prpMinMax])
    scatterlines!(ax1, X_sd; 
        markersize=10f0, label="sd", color=DOC_COLORS[:sd], linestyle=DOC_LSTYLES[:sd])
    scatterlines!(ax2, X_prp; 
        markersize=10f0, label="prp", color=DOC_COLORS[:prpMinMax], linestyle=DOC_LSTYLES[:prpMinMax])
    scatterlines!(ax2, X_sd; 
        markersize=10f0, label="sd", color=DOC_COLORS[:sd], linestyle=DOC_LSTYLES[:sd])

    ## activate legend
    axislegend(ax1; position=:lb)

    ## display plot
    fig
end
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
let
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
    
    scatterlines!(ax, X_prp;
        markersize=10f0, label="prp", color=DOC_COLORS[:prpMinMax], linestyle=DOC_LSTYLES[:prpMinMax])
    scatterlines!(ax, X_sd; 
        markersize=10f0, label="sd", color=DOC_COLORS[:sd], linestyle=DOC_LSTYLES[:sd])
    
    axislegend(ax)
    fig
end

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
        callbacks = [M.CriticalityStop(;eps_crit=0.0), M.GatheringCallback(cache)]
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

# Define the experiments by name and rule:
names_rules = (
    sd = M.SteepestDescentRule(M.StandardArmijoRule()),
    prpMinMax = M.PRP(M.ModifiedArmijoRule(), :sd)
)

# Run and plot:
let
    fig = Figure()
    ax = Axis(fig[1,1]; yscale=log10, xlabel="it", ylabel="crit")
    for (name, rule) = pairs(names_rules)
        X_it = do_runs(rule; max_iter)
        μ, σ, q1, q2, q3 = consume_results(X_it)
        it = 0:length(μ)-1
        lines!(ax, it, μ; label=string(name), color=DOC_COLORS[name])
        lines!(ax, it, q2; color=DOC_COLORS[name], linestyle=:dash)
        band!(ax, it, q1, q3; color=(DOC_COLORS[name], 0.2))
    end
    axislegend(ax)
    fig
end