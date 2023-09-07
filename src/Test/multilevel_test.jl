include("../utility.jl")
include("../myConstants.jl")
include("../strength_psparse.jl")
include("../splitting_psparse.jl")
include("../smoother_psparse.jl")
#include("matmatmul_psparse.jl")
include("../matmatmul_helper.jl")
include("../classical_psparse.jl")
include("../multilevel_psparse.jl")
include("../preconditioner_psparse.jl")

using SparseArrays
using PartitionedArrays
#=
using MPI 
mpiexec(cmd->run(`$cmd -np 4 julia --project=. src/Test/multilevel_test.jl`))
=#

np = 4
n = 50
const ranks = distribute_with_mpi(LinearIndices((np,)))
#const ranks = LinearIndices((np,))

A = ppoisson(n);

ml = ruge_stuben(A);

out = _solve(ml, rhs(A));
map(own_values(out)) do o
    @show o
end

# RS solver
out2 = solve(A, rhs(A), RugeStubenAMG(), maxiter = 10, abstol = 1e-6)
map(own_values(out2)) do RS_o
    @show RS_o
end

import IterativeSolvers: cg
p = aspreconditioner(ml)
cg_out = cg(A, rhs(A), Pl = p)
map(own_values(cg_out)) do CG_pre_out
    @show CG_pre_out
end
