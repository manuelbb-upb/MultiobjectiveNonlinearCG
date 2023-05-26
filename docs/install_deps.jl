using Pkg

Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))

Pkg.add(;url="https://github.com/manuelbb-upb/MultiTaskLearning.jl.git"); 
Pkg.add(;url="https://github.com/manuelbb-upb/MultiMLDatasets.jl.git");

Pkg.instantiate()