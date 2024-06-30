# This file is meant to be parsed by Literate.jl            #src
# Activate `docs` environment:                              #src
using Pkg                                                   #src
Pkg.activate(joinpath(@__DIR__, "..", ".."))                #src
#%%#src
# # MultiobjectiveNonlinearCG Tutorials
import MultiobjectiveNonlinearCG as MCG
using CairoMakie
includet("PlotHelpers.jl")  #src
#md includet("PlotHelpers.jl")
import .PlotHelpers: get_limits
import UnPack: @unpack

#%%#src

## Define the function:
f(x1, x2, a, b) = b * (x2 - x1^2)^2 + (a - x1)^2

# The gradient of the Rosenbrock function is known analitically (see above):
df(x1, x2, a, b) = [
    -4*b*(x2-x1^2)*x1 - 2*(a-x1),
    2*b*(x2-x1^2)
]

# ### Plotting the Pareto-Set
# Let's fix some parameters and define our objectives:
a1 = 1.0
a2 = 2.0
b1 = b2 = 100
f1(x1, x2) = f(x1, x2, a1, b1)
f2(x1, x2) = f(x1, x2, a2, b2)

## define gradients:
df1(x1, x2) = df(x1, x2, a1, b1)
df2(x1, x2) = df(x1, x2, a2, b2)

# In-place functions for optimization:
objectives! = function (y, x)
    y[1] = f1(x[1], x[2])
    y[2] = f2(x[1], x[2])
end

jac! = function (Dy, x)
    Dy[1, :] .= df(x[1], x[2], 1, 100)
    Dy[2, :] .= df(x[1], x[2], 2, 100)
end
# Build the MOP:
mop = MCG.BasicMOP(;
    dim_in = 2,
    dim_out = 2,
    objectives!,
    jac!
)

x0 = [4, 4.5]
#%%
result_df = DataFrame(
    step_rule = MCG.AbstractStepRule[],
    x = Matrix{Float64}[],
    callback = MCG.GatheringCallback{Float64}[]
)
for step_rule in (
    MCG.FletcherReevesRestart(),
    MCG.FletcherReevesFractionalLP(),
    MCG.PRP3(),
    MCG.PRPConeProjection()
)
    callback = MCG.GatheringCallback(Float64)
    opt_res = MCG.optimize(x0, mop, callback; step_rule, max_iter=50)
    push!(result_df, (step_rule, MCG._x_mat(callback), callback))
end
#%%
function rosenbrock_base_plot(x1_lims, x2_lims)
    fig = Figure(; )
    ax1 = Axis(fig[1,1])
    ax2 = Axis(fig[1,2])
    X1 = range(x1_lims..., 100)
    X2 = range(x2_lims..., 100)
    xlims!(ax1, x1_lims)
    xlims!(ax2, x1_lims)
    ylims!(ax1, x2_lims)
    ylims!(ax2, x2_lims)
    levels = collect(-3:1:9)
    c1 = heatmap!(ax1, X1, X2, f1; colormap=:acton, colorscale=log)
    c2 = contourf!(ax2, X1, X2, log ∘ f2; levels, extendhigh=:auto, colormap=:acton) 
    Colorbar(fig[1, 0], c1)
    Colorbar(fig[1, 3], c2; ticks=(levels, string.(levels)))
    return fig, ax1, ax2
end
let
    x1_lims, x2_lims = get_limits(result_df.x)
    fig, ax1, ax2 = rosenbrock_base_plot(x1_lims, x2_lims)
    for row in eachrow(result_df)
        @unpack step_rule, x, callback = row
        markersize = PlotHelpers.critval_markersizes(callback.critvals)
        color = 1:size(row.x, 2)
        base_color = PlotHelpers.step_rule_color(step_rule)
        colormap = PlotHelpers.cmap_gradient(base_color)
        label = PlotHelpers.step_rule_label(step_rule)
        scatterlines!(ax1, Point2[]; color = base_color, label)
        scatterlines!(
            ax1, row.x;
            markersize,
            color, colormap,
            strokecolor=(:white, .1f0),
            strokewidth=1f0,
        )
    end
    axislegend(ax1)
    fig
end