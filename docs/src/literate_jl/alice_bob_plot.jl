# This file is meant to be parsed by Literate.jl #src
if !(joinpath(@__DIR__, "..", "..") in LOAD_PATH) #src
    pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "..")) #src
end #src
using Pkg #src
Pkg.activate(@__DIR__) #src
include(joinpath(@__DIR__, "makie_theme.jl")) #hide
nothing #hide

SAVEFIGS=true #src
SAVEPATH=joinpath(@__DIR__, "..", "slides", "assets") #src

#=
# Pareto-Optimality
Consider the formal optimization problem
```math
\min_{\symbf{x}\in ℝ^N}
\begin{bmatrix}
    f_1(\symbf{x})
    \\
    \vdots
    \\
    f_K(\symbf{x})
\end{bmatrix}
\tag{MOP}
```
What are solutions of a problem with multiple objectives?
Well, acceptable trade-offs between the objective function values.
Formally, we use the concept of Pareto-optimality.

A point $\symbf{x}^*\in ℝ^N$ is called **Pareto-optimal** if 
there is no $\symbf{x}\in ℝ^N$
* that is at least as good as $\symbf{x}^*$, i.e.,  
  no $\symbf{x}$ with $f_ℓ(\symbf{x}) ≤ f_ℓ(\symbf{x}^*)$ for all $ℓ=1,…,K$,
* and that is strictly better in at least one objective, i.e., no  
  $\symbf{x}$ for which there is some $ℓ\in\{1,…,K\}$ with  
  $f_ℓ(\symbf{x})<f_ℓ(\symbf{x}^*)$.

The set of all optimal solutions is the **Pareto-Set** and its 
image is the **Pareto-Front**.

# Example

Alice 👩 and Bob 👨 want to meet.
Alice lives at ``(1, 1)`` and Bob lives at ``(-1, -1)``.
So the distance to Alice's home is
```math
d_A(x_1, x_2) = \sqrt{(x_1-1)^2 + (x_2-1)^2}.
```
For Bob, it is
```math
d_B(x_1, x_2) = \sqrt{(x_1+1)^2 + (x_2+1)^2}.
```

Deciding on a meeting venue is the bi-objective optimization problem
```math
\min_{\symbf{x}\in ℝ2}
\begin{bmatrix}
    d_A(\symbf x)
    \\
    d_B(\symbf x)
\end{bmatrix}
```

Let's visualize the situation:
=#
using CairoMakie
using FileIO

# Define the distance functions:
dA(x1, x2) = sqrt((x1-1)^2 + (x2-1)^2)
dB(x1, x2) = sqrt((x1+1)^2 + (x2+1)^2)

# A function to give us a basic figure with Alice 
# and Bob in it:

markerA = load(joinpath(@__DIR__,"woman.png"))
markerB = load(joinpath(@__DIR__,"man.png"))

function setup_fig()
    global markerA, markerB
    set_theme!(DOC_THEME2) #hide
    fig = Figure()

    ## set a title
    Label(fig[1,1:2], "Alice and Bob Want to Meet.")

    ## left axis - decision space, where Alice and Bob live
    ax1 = Axis(fig[2,1]; aspect=1.0)
    xlims!(ax1, (-1.5, 1.5))
    ylims!(ax1, (-1.5, 1.5))

    ## set markers for Alice and Bob
    scatter!(ax1, (1, 1); marker=markerA, markersize=30)
    scatter!(ax1, (-1, -1); marker=markerB, markersize=30)

    ## right axis - objective space, distance values
    ax2 = Axis(fig[2,2]; aspect=1.0, xlabel=L"d_A", ylabel=L"d_B")
    xlims!(ax2, (-0.1, 3.5))
    ylims!(ax2, (-0.1, 3.5))
    return fig, ax1, ax2
end

## show the basic image:
fig0 = first(setup_fig())
fig0
SAVEFIGS && save(joinpath(SAVEPATH, "alice_bob_plot0.png"), fig0) #src

#=
To compare various meeting positions, we build a function
testing for Pareto-optimality in a discrete set
with value vectors Y.
=#

function check_optimal(Y)
    _Y = empty(Y)               # array of processed value vectors
    isoptimal = zeros(Bool, 0)  # flags
    for fi in Y
        fi_isoptim = true
        for (j,fj) in enumerate(_Y)
            if all(fj .<= fi) && any(fj .< fi)
                ## fj is better than fi
                fi_isoptim = false
                break
            end
            ## test, if fi is better than fj 
            if isoptimal[j]
                if all(fi .<= fj) && any( fi .< fj)
                    isoptimal[j] = false
                end        
            end
        end
        push!(_Y, fi)
        isoptimal = vcat(isoptimal, fi_isoptim)
    end
    return isoptimal
end

# Let's look at some meeting places.
# First, only consider Alices' home:
X = [(1.0, 1.0),]
Y = [(dA(x[1], x[2]), dB(x[1], x[2])) for x in X]
# Obviously, the point is optimal **in our discrete sample**:
isopt = check_optimal(Y)
all(isopt) == true

# With more points, it gets more complicated.
# We want to scatter the `X` and `Y` points and use 
# different markers for optimal and non-optimal points.

function scatterOpt!(axX, axY, X, Y, isopt; colors)
    @assert length(X) == length(Y) == length(isopt)

    ## plot non-optimal points first
    si = sortperm(isopt)

    for i=si
        color = colors[i]
        marker, strokewidth, strokecolor = if isopt[i]
            (:circle, 0.5, :black)
        else
            (:xcross, 1.5, :orange)
        end

        scatter!(axX, (X[i]...); color, marker, strokewidth, strokecolor)
        scatter!(axY, (Y[i]...); color, marker, strokewidth, strokecolor)
    end
    nothing
end

# Now, Alice's place should get a round marker.
let
    fig, ax1, ax2 = setup_fig()
    scatterOpt!(ax1, ax2, X, Y, isopt; colors=Makie.wong_colors()) 
    SAVEFIGS && save(joinpath(SAVEPATH, "alice_bob_plot1.png"), fig) #src
    fig
end

# Now we include Bob's home.
X = [
    (1.0, 1.0),
    (-1.0, -1.0)
]
Y = [(dA(x[1], x[2]), dB(x[1], x[2])) for x in X]
# Still, both points are Pareto-optimal in our sample set.
isopt = check_optimal(Y)
isopt == [true, true]
# This is reflected in the plot.
# Note, that individual minima are by definition always Pareto-optimal,
# also in the non-discrete case.
let
    fig, ax1, ax2 = setup_fig()
    scatterOpt!(ax1, ax2, X, Y, isopt; colors=Makie.wong_colors())
    SAVEFIGS && save(joinpath(SAVEPATH, "alice_bob_plot2.png"), fig) #src
    fig
end

# By adding a third point, we see that it is important
# to distinguish our discrete comparisons and the continuous case. 
X = [
    (1.0, 1.0),
    (-1.0, -1.0),
    (-1.0, 1.0),
]
Y = [(dA(x[1], x[2]), dB(x[1], x[2])) for x in X]
let
    fig, ax1, ax2 = setup_fig()
    scatterOpt!(ax1, ax2, X, Y, check_optimal(Y); colors=Makie.wong_colors()) 
    SAVEFIGS && save(joinpath(SAVEPATH, "alice_bob_plot3.png"), fig) #src
    fig
end
# Surprisingly, ``(-1, 1)`` is considered optimal, despite it 
# being fare away from Alice and Bob.
# ``(0,0)`` would certainly be preferrable for both:

X = [
    (1.0, 1.0),
    (-1.0, -1.0),
    (-1.0, 1.0),
    (0.0, 0.0),
]
Y = [(dA(x[1], x[2]), dB(x[1], x[2])) for x in X]

fig, ax1, ax2 = setup_fig()
scatterOpt!(ax1, ax2, X, Y, check_optimal(Y); colors=Makie.wong_colors()) 
SAVEFIGS && save(joinpath(SAVEPATH, "alice_bob_plot4.png"), fig) #src
fig

# That's it! With more samples we get a better sense for the
# optimal points of the continuous problem (MOP).
# Thus, let's use a pseudo-random sampling of 
# decision space and see if we can identify the Pareto Set.

using HaltonSequences
using ColorSchemes
num_X = 200
X = [Tuple( -1.4 .+ 2.8 .* p  ) for p in HaltonPoint(2; length=num_X)]
Y = [(dA(x[1], x[2]), dB(x[1], x[2])) for x in X]
scatterOpt!(
    ax1, ax2, X, Y, check_optimal(Y); 
    colors=[get(ColorSchemes.acton, (i-1)/(num_X-1)) for i=1:num_X]
) 
fig
SAVEFIGS && save(joinpath(SAVEPATH, "alice_bob_plot5.png"), fig) #src

# Ok, we got a rough sense for the Pareto Set.
# Indeed, we can analitically show that its the line
# from ``(-1, -1)`` to ``(1,1)``.
lines!(ax1, [(-1, -1), (1, 1)]; linewidth=5f0, color=:blue)
PF = [(dA(x,x), dB(x, x)) for x=1:-0.1:-1]
lines!(ax2, PF; linewidth=5f0, color=:blue)
SAVEFIGS && save(joinpath(SAVEPATH, "alice_bob_plot6.png"), fig) #src
fig
