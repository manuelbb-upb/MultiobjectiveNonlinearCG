using Pkg
Pkg.activate(@__DIR__)
import MultiobjectiveNonlinearCG as M
#%%
f1(x) = sum((x .- 1).^2)
f2(x) = sum((x .+ 1).^2)

df1(x) = 2 .* (x .- 1)
df2(x) = 2 .* (x .+ 1)

function objf!(y, x)
    y[1] = f1(x)
    y[2] = f2(x)
    nothing
end

function jacT!(DfxT, x)
    DfxT[:, 1] = df1(x)
    DfxT[:, 2] = df2(x)
end

x0 = [1, 2, 3]
fx0 = zeros(2)
objf!(fx0, x0)
#%%
#descent_rule = M.SteepestDescentRule(M.StandardArmijoRule())
descent_rule = M.PRP(;
    stepsize_rule=M.StandardArmijoRule(),
    #stepsize_rule=M.ModifiedArmijoRule(),
    criticality_measure=:cg
)
#descent_rule = M.PRP(M.ModifiedArmijoRule())
#descent_rule = M.SteepestDescentRule(M.FixedStepsizeRule(0.1))
M.optimize(x0, fx0, objf!, jacT!;descent_rule)
#%%

xtolrel = M.RelativeStoppingX(; eps=-1)
fxtolrel = M.RelativeStoppingFx(; eps=0.1)
critstop = M.CriticalityStop(; eps_crit = 1)
M.optimize(x0, fx0, objf!, jacT!;descent_rule, stopping_criteria=[critstop])