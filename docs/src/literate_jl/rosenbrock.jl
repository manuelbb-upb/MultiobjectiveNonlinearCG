# This file is meant to be parsed by Literate.jl
using Pkg #src
Pkg.activate(joinpath(@__DIR__, "..", "..")) #src

import MultiobjectiveNonlinearCG as M
import ForwardDiff as FD
using CairoMakie
CairoMakie.activate!()

#%% #src
struct ExperimentData
    x :: Matrix{Float64}
    fx :: Matrix{Float64}
    name :: Symbol
end

## Settings
LB = fill(-2.0, 2)
UB = fill(2.0, 2)
a_1 = 1.0
a_2 = 1.2

f1 = x -> sum( 100 * ( x[2:end] .- x[1:end-1].^2 ).^2 + (a_1 .- x[1:end-1]).^2 )
f2 = x -> sum( 100 * ( x[2:end] .- x[1:end-1].^2 ).^2 + (a_2 .- x[1:end-1]).^2 )

f = x -> [f1(x); f2(x)]
Df = x -> transpose(FD.jacobian(f, x))

rand_x0 = () -> LB .+ (UB .- LB) .* rand(2)

function do_experiments( exps; max_iter = 100)
    x0 = rand_x0()
    x0 = [-3, 2.5]
    results = ExperimentData[]
    for (ename, descent_rule) = exps
        cache = M.GatheringCallbackCache(Float64)
        callbacks = [
            M.CriticalityStop(;eps_crit=1e-5),
            M.GatheringCallback(cache),
        ]
    

        _ = M.optimize( x0, f, Df; max_iter, objf_iip=false, jacT_iip=false, 
            descent_rule, callbacks)
    
        res = ExperimentData(
            reduce(hcat, cache.x_arr),
            reduce(hcat, cache.fx_arr),
            ename
        )
        push!(results, res)
    end

    return results
end

function get_lims(results; margin=0.05)
    global LB, UB
    # xlims, ylims = collect.(to_limits(LB, UB))
    xlims = [Inf, -Inf]
    ylims = [Inf, -Inf]
    for res in results
        _xlims, _ylims = extrema(res.x, dims=2)
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

function crit_map(x1, x2)
    global Df
    x = [x1, x2]
    D_T = Df(x)
    δ = M.frank_wolfe_multidir_dual(eachcol(D_T))
    return sum(δ.^2)    
end
#%% #src

# !!! note
#     Until [this issue in Makie](https://github.com/MakieOrg/Makie.jl/pull/2900)
#     is fixed, having a nice log scaled colorbar is ugly.
#     Inspired by a [discourse post](https://discourse.julialang.org/t/heatmap-with-log-scale-colorbar-cscale/63018)

struct LogMinorTicks end
	
function MakieLayout.get_minor_tickvalues(
        ::LogMinorTicks, scale, tickvalues, vmin, vmax
)
    vals = Float64[]
    for (lo, hi) in zip(
            @view(tickvalues[1:end-1]),
            @view(tickvalues[2:end])
        )
        steps = log10.(LinRange(10^lo, 10^hi, 11))
        append!(vals, steps[2:end-1])
    end
    vals
end

custom_formatter(values) = map(
    v -> "10" * Makie.UnicodeFun.to_superscript(round(Int64, v)),
    values
)

using Colors, ColorSchemes

darken(col; fac=0.5) = RGB(col.r * 0.75f0, col.g * 0.75f0, col.b * 0.75f0)
tint(col; fac=0.8) = RGB( col.r + (1-col.r) * 0.25f0, col.g + (1-col.g) * 0.25f0, col.b + (1-col.b) * 0.25f0)

function plot_x_trajectories(results)
    
    global a_1, a_2

    xlims, ylims = get_lims(results)

    fig = Figure()
    ax = Axis(fig[1,1]; limits = (xlims, ylims))

    X1 = LinRange(xlims..., 125)
    X2 = LinRange(ylims..., 125)
    cont = contourf!(ax, X1, X2, log10 ∘ crit_map;
        levels = 20,
        colormap = :bilbao
    )

    ps1 = LinRange(a_1, a_2, 100)
    ps2 = ps1.^2
    lines!(ax, ps1, ps2; 
        color = :red, linewidth = 1.2
    )

    cols = Makie.wong_colors()
    for (res, col) in zip(results, cols)
        c = range(darken(col), tint(col), size(res.x, 2) )
        scatterlines!(ax, res.x; 
            label=string(res.name),
            strokewidth=0.5,
            markercolor=c,
        )
    end

    cb = Colorbar(fig[1,2], cont;
        tickformat = custom_formatter,
        minorticksvisible=true,
		minorticks=LogMinorTicks()
    )
    #cb.axis.attributes[:scale][] = log10
    #@show cb.axis.attributes[:limits][]

    axislegend(ax; position=:lb)
    fig
end

exps = (
#    (:sd, M.SteepestDescentRule(M.StandardArmijoRule(; σ_init=M.QuadApprox()))),
    (:sd, M.SteepestDescentRule(M.StandardArmijoRule())),
#    (:sd2, M.SteepestDescentRule(M.ModifiedArmijoRule(; b=0.8))),
    (:prp, M.PRP(M.ModifiedArmijoRule(), :sd)),
    (:fr, M.FRRestart(M.ModifiedArmijoRule(), :sd)),
)
results = do_experiments(exps; max_iter = 50)

plot_x_trajectories(results)