using Pkg
current_env = first(Base.load_path())
Pkg.activate(@__DIR__)

# Pkg.develop(PackageSpec(;path=joinpath(@__DIR__, "..")))

include("make.jl")

Pkg.activate(current_env)