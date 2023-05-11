using Pkg #src
Pkg.activate(joinpath(@__DIR__, "..", "..")) #src

#using Plots
#plotlyjs()
using PlotlyJS
import MultiobjectiveNonlinearCG as M
import ForwardDiff as FD

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

exps = (
#    (:sd, M.SteepestDescentRule(M.StandardArmijoRule(; σ_init=M.QuadApprox()))),
    (:sd, M.SteepestDescentRule(M.StandardArmijoRule())),
#    (:sd2, M.SteepestDescentRule(M.ModifiedArmijoRule(; b=0.8))),
    (:prp, M.PRP(M.ModifiedArmijoRule(), :sd)),
    (:fr, M.FRRestart(M.ModifiedArmijoRule(), :sd)),
)
results = do_experiments(exps; max_iter = 50)
#%%
import ColorSchemes
#let
    xlims, ylims = get_lims(results)
    X1 = collect(LinRange(xlims..., 150))
    X2 = collect(LinRange(ylims..., 150))
    Z = [log10(crit_map(x1, x2)) for x1=X1, x2=X2]
    levels = contour(;
        x=X1, y=X2, z=Z, transpose=false, line_width=0.1,
        colorscale=ColorSchemes.acton10.colors, 
        #ncontours=30, 
        smoothing=1.3,
        contours_start=floor(Int, minimum(Z)),
        contours_end=8,
        contours_size=0.5
    )
    #=for res in results
        plot!(fig, res.x[1,:], res.x[2,:]; markershape=:circle, label=string(res.name))
    end=#
    plot(levels)
#end