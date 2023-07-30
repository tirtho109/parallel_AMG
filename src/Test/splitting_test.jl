include("../utility.jl")
include("../strength_psparse.jl")
include("../splitting_psparse.jl")
using SparseArrays
using PartitionedArrays
#=
using MPI 
mpiexec(cmd->run(`$cmd -np 4 julia --project=. test.jl`))
=#

np = 4
n = 301
const ranks = distribute_with_mpi(LinearIndices((np,)))
#const ranks = LinearIndices((np,))

A = ppoisson(n);
b = rhs(A)
x = create_x(A,b)
strength = Classical(0.25)
S,T = strength(A);

CF = RS()
splitting = CF(S)

map(splitting.index_partition) do split
    @show length(split)
end
