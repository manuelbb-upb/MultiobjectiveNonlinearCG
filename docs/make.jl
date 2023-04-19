using Documenter
using MultiobjectiveNonlinearCG

makedocs(
    sitename = "MultiobjectiveNonlinearCG",
    format = Documenter.HTML(),
    modules = [MultiobjectiveNonlinearCG]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/manuelbb-upb/MultiobjectiveNonlinearCG.jl.git"
)
