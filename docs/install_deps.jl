using Pkg

#=
# This does not work without a custom registry -- I am too lazy for that and 
# upload the Manifest.toml instead...
Pkg.develop(;url="https://github.com/manuelbb-upb/MultiMLDatasets.jl.git");
Pkg.develop(;url="https://github.com/manuelbb-upb/MultiTaskLearning.jl.git"); 
=#

Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()