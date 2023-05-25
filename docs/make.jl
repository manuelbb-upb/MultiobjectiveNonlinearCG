using Documenter
using MultiobjectiveNonlinearCG

using Literate

#DIR_PATH = "\"$(joinpath(@__DIR__, "src", "literate_jl"))\""
DIR_PATH = "joinpath(\"..\", \"literate_jl\")"
function dir_preprocess(content)
    return replace(content, "@__DIR__" => DIR_PATH)
end


function make_literate()
    Literate.markdown(
        joinpath(@__DIR__, "..", "src", "dir_rules", "multidir_frank_wolfe.jl"),
        joinpath(@__DIR__, "src", "generated");
        documenter=true
    )
    Literate.markdown(
        joinpath(@__DIR__, "src" , "literate_jl", "two_parabolas.jl"),
        joinpath(@__DIR__, "src", "generated");
        documenter=true, preprocess=dir_preprocess
    )
    Literate.markdown(
        joinpath(@__DIR__, "src" , "literate_jl", "two_rosenbrock.jl"),
        joinpath(@__DIR__, "src", "generated");
        documenter=true, preprocess=dir_preprocess,
    )
    Literate.markdown(
        joinpath(@__DIR__, "src" , "literate_jl", "alice_bob_plot.jl"),
        joinpath(@__DIR__, "src", "generated");
        documenter=true, preprocess=dir_preprocess,
    )
    return nothing
end

make_literate()

makedocs(
    sitename = "MultiobjectiveNonlinearCG",
    format = Documenter.HTML(;
        mathengine=Documenter.MathJax3(),
        assets = ["assets/custom.css"],
    ),
    modules = [MultiobjectiveNonlinearCG], 
    pages = [
        "Pareto Optimality" => "generated/alice_bob_plot.md",
        "Frank-Wolfe Solver" => "generated/multidir_frank_wolfe.md",
        "2 Parabolas" => "generated/two_parabolas.md",
        "2 Rosenbrocks" => "generated/two_rosenbrock.md",
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/manuelbb-upb/MultiobjectiveNonlinearCG.jl.git",
    devbranch = "main"
)
