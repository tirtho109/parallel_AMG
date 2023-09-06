include("../utility.jl")
include("../myConstants.jl")
include("../strength_psparse.jl")
include("../splitting_psparse.jl")
include("../smoother_psparse.jl")
#include("matmatmul_psparse.jl")
include("../matmatmul_helper.jl")
include("../classical_psparse.jl")
include("../multilevel_psparse.jl")


using SparseArrays
using PartitionedArrays

#=
using MPI 
mpiexec(cmd->run(`$cmd -np 4 julia --project=. src/Test/gather_pvecmat_test.jl`))
=#

np = 4
n = 15
const ranks = distribute_with_mpi(LinearIndices((np,)))

A = ppoisson(n);
b = rhs(A)
x = pzeros(A.col_partition)

A_new = accumulate_psparse2(A)
b_new = accumulate_pvector(b)

map(b_new, A_new) do B, a 
    @show B, a
end

b_emited = emit(b_new;source=MAIN)

# map(b_emited) do be
#     @show be
# end