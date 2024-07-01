using Pkg
current_env = first(Base.load_path())
try
Pkg.activate(@__DIR__)

using Documenter
using MultiobjectiveNonlinearCG

using Literate

begin
    cp(
        joinpath(@__DIR__, "src", "literate_jl", "makie_theme.jl"),
        joinpath(@__DIR__, "src", "makie_theme.jl");
        force=true
    )
    cp(
        joinpath(@__DIR__, "src", "literate_jl", "PlotHelpers.jl"),
        joinpath(@__DIR__, "src", "PlotHelpers.jl");
        force=true
    )
    Literate.markdown(
        joinpath(@__DIR__, "src", "literate_jl", "pareto_optimality.jl"),
        joinpath(@__DIR__, "src");
        documenter=true
    )
    Literate.markdown(
        joinpath(@__DIR__, "src", "literate_jl", "steepest_descent_plots.jl"),
        joinpath(@__DIR__, "src");
        documenter=true
    )
end
makedocs(
    sitename = "MultiobjectiveNonlinearCG",
    format = Documenter.HTML(;
        mathengine=Documenter.MathJax3(),
    ),
    modules = [MultiobjectiveNonlinearCG, ], 
    pages = [
        "Home" => "index.md",
        "Pareto Optimality" => "pareto_optimality.md",
    ],
    warnonly = [:missing_docs,]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/manuelbb-upb/MultiobjectiveNonlinearCG.git",
    devbranch = "main"
)
finally
Pkg.activate(current_env)
end