```@meta
EditURL = "<unknown>/docs/src/literate_jl/multi_mnist.jl"
```

````@example multi_mnist
using Pkg #hide
nothing #hide

if isnothing(Base.find_package("MultiMLDatasets")) #hide
    Pkg.add(;url="https://github.com/manuelbb-upb/MultiMLDatasets.jl.git")
end #hide

using MultiMLDatasets
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

