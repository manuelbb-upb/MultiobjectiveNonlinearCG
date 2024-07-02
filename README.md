**There is a lot going on in branch `remake`. The main branch is outdated. Will update soon.**

# MultiobjectiveNonlinearCG

This is a package meant primarily for academic and research purposes.
For my own sanity, I have incorporated a few convenience functions,
but don't expect anything particularly usable or stable.

As things are changing fast behind the scenes, please refer to the [docs](https://manuelbb-upb.github.io/MultiobjectiveNonlinearCG/)
for usage hints.
The slides are linked there, too.

The package is not (yet) registered, but you can install it from master via
```julia
using Pkg
Pkg.add(;url="https://github.com/manuelbb-upb/MultiobjectiveNonlinearCG.git")
```

A draft PDF can be found in the `tex` folder, aptly named “template.pdf”.

The Julia code is licensed under the MIT license.
The slides use Reveal.js, which is also licensed under MIT.
The contents of the slides, as well as the LaTeX code and the article draft PDF are licensed under CC BY 4.0.

We hope to put something in the arXiv soon.
In the meantime:
```
@misc{Berkemeier_Peitz, 
    title={Nonlinear Conjugate Gradient Methods with Guaranteed Descent for Multi-Objective Optimzation},
    author={Berkemeier, Manuel Bastian and Peitz, Sebastian},
    year={2023},
    howpublished = \url{https://github.com/manuelbb-upb/MultiobjectiveNonlinearCG},
} 
```
