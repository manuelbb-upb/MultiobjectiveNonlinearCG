# # MultiobjectiveNonlinearCG Tutorials
import MultiobjectiveNonlinearCG as MCG
#%%#src
mop = MCG.BasicMOP(;
    dim_in = 3,
    dim_out = 2,
    objectives! = function (y, x)
        y[1] = sum( (x .- 1).^2 )
        y[2] = sum( (x .+ 1).^2 )
        nothing
    end,
    jac! = function(Dy, x)
        Dy[1, :] .= 2 .* (x .- 1)
        Dy[2, :] .= 2 .* (x .+ 1)
        nothing
    end
)

x0 = [3.142, -2.718, 0]
dir_rule = MCG.SteepestDescentDirection(MCG.StandardArmijoBacktracking(; factor=.8))
#dir_rule = MCG.SteepestDescentDirection()

MCG.optimize(x0, mop; dir_rule, max_iter=10, x_tol_abs=1e-10)