using Documenter
using MultiobjectiveNonlinearCG

using Literate

function make_literate()
    Literate.markdown(
        joinpath(@__DIR__, "..", "src", "dir_rules", "multidir_frank_wolfe.jl"),
        joinpath(@__DIR__, "src", "generated");
        documenter=true
    )
    Literate.markdown(
        joinpath(@__DIR__, "src" , "literate_jl", "two_parabolas.jl"),
        joinpath(@__DIR__, "src", "generated");
        documenter=true
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
        "Frank-Wolfe Solver" => "generated/multidir_frank_wolfe.md",
        "2 Parabolas" => "generated/two_parabolas.md",
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/manuelbb-upb/MultiobjectiveNonlinearCG.jl.git"
)
