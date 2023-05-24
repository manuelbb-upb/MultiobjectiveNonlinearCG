using Pkg
Pkg.activate(@__DIR__)
import MultiobjectiveNonlinearCG as M
#%%
f1(x) = sum((x .- 1).^2)
f2(x) = sum((x .+ 1).^2)

df1(x) = 2 .* (x .- 1)
df2(x) = 2 .* (x .+ 1)

f(x) = vcat(f1(x), f2(x))
DfT(x) = hcat(df1(x), df2(x))
x0 = [1, 2, 3]
#%%
#descent_rule = M.SteepestDescentRule(M.StandardArmijoRule())
descent_rule = M.PRP(;
    #stepsize_rule=M.StandardArmijoRule(),
    stepsize_rule=M.ModifiedArmijoRule(),
    criticality_measure=:cg
)
descent_rule = M.PRPGradProjection()
#descent_rule = M.PRP(M.ModifiedArmijoRule())
#descent_rule = M.SteepestDescentRule(M.FixedStepsizeRule(0.1))
M.optimize(x0, f, DfT; descent_rule)
#%%
#=
xtolrel = M.RelativeStoppingX(; eps=-1)
fxtolrel = M.RelativeStoppingFx(; eps=0.1)
critstop = M.CriticalityStop(; eps_crit = 1)
M.optimize(x0, fx0, objf!, jacT!;descent_rule, stopping_criteria=[critstop])
=#