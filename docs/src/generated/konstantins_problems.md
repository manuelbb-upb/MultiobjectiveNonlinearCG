```@meta
EditURL = "<unknown>/docs/src/literate_jl/konstantins_problems.jl"
```

# Scalable Convex Problem
For reference, see
> K. Sonntag and S. Peitz,
> “Fast Multiobjective Gradient Methods with Nesterov Acceleration via
> Inertial Gradient-like Systems.”
> arXiv, Jul. 26, 2022. Accessed: Feb. 08, 2023. [Online].
> Available: http://arxiv.org/abs/2207.12707

The objectives involve random coefficients.
Make everything reproducible:

````@example konstantins_problems
import Random
Random.seed!(3142)

#= The objectives are
```math
    f_ℓ(\symbf{x}) = \ln \sum_j^p \exp( A^{(ℓ)}_{j,:} \cdot x ).
```
Here, ``A^{(ℓ)}\in ℝ^{p\times n}`` is random.
We implement it accordingly:
=#
import ForwardDiff as FD
function setup_problem(n_vars, n_objf, n_coeff)
    funcs = Any[]
    grads = Any[]
    for l=1:n_objf
        A = -1 .+ 2 .* rand(n_coeff, n_vars)
        objf_l = x -> log(sum(exp.(A*x)))
        grad_l = x -> FD.gradient(objf_l, x)
        push!(funcs, objf_l)
        push!(grads, grad_l)
    end
    F = x -> reduce(vcat, f(x) for f in funcs)
    JT = x -> reduce(hcat, g(x) for g in grads)

    return F, JT
end
````

We propagate `NUM0` inital points:

````@example konstantins_problems
NUM0=50
NUM_VARS=20
NUM_OBJF=3
NUM_COEFF=50
MAX_ITER=75

LB = -15
UB = 15
rand_x0 = (n_vars) -> -15 .+ (UB-LB) .* rand(n_vars)
X0 = [rand_x0(NUM_VARS) for i=1:NUM0]

objf, jacT = setup_problem(NUM_VARS, NUM_OBJF, NUM_COEFF)
````

Compare Steepest Descent and some CG directions.
To make repeated tasks easier, I set up a list of experiments to do:

````@example konstantins_problems
import MultiobjectiveNonlinearCG as M

experiment_settings = [
    ("SD", M.SteepestDescentRule(M.StandardArmijoRule()), :sd,),
    ("PRP3", M.PRP(M.ModifiedArmijoRule(), :sd), :prp3,),
];
nothing #hide
````

Setup an optimization run to gather information in `cache`:

````@example konstantins_problems
function do_experiment(x0, objf, jacT; descent_rule, max_iter=100)
    cache = M.GatheringCallbackCache(Float64)
    callbacks = vcat(M.DEFAULT_CALLBACKS, M.GatheringCallback(cache))
    _ = M.optimize(x0, objf, jacT; descent_rule, max_iter, callbacks)
    return cache
end
````

Perform all the experiments and store results in a Dict.
Might take some time:

````@example konstantins_problems
experiment_results = let res=Dict();
    for cfg in experiment_settings
        label, descent_rule, _ = cfg
        res[label]=Any[]
        for x0 in X0
            c = do_experiment(x0, objf, jacT; descent_rule, max_iter=MAX_ITER)
            push!(res[label], c)
        end
    end
    res
end
````

For plotting, we use CairoMakie and some custom settings:

````@example konstantins_problems
using CairoMakie
include(joinpath(joinpath("..", "literate_jl"), "makie_theme.jl"))
nothing #hide
````

Produce the figure in a local scope:

````@example konstantins_problems
let
    fig = Figure()
    ax = Axis3(fig[1,1];
        title="$(NUM0) points, $(NUM_VARS) vars, $(MAX_ITER) its.",
        titlegap=15f0,
        azimuth=1.55*pi,
        titlesize=35f0,
        xlabel=L"f_1",
        ylabel=L"f_2",
        zlabel=L"f_3",
        xlabelsize=40f0,
        ylabelsize=40f0,
        zlabelsize=40f0,
    )
    for (label, _, pk) in experiment_settings
        final_points = reduce(hcat, c.fx_arr[end] for c in experiment_results[label])
        scatter!(ax, final_points; color=DOC_COLORS[pk], label, strokewidth=1.0f0, markersize=16f0)
    end
    rowsize!(fig.layout, 1, Relative(5/6))
    axislegend(ax; position=:lt)
    fig
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

