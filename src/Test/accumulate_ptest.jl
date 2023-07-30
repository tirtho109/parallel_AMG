include("../utility.jl")
include("../multilevel_psparse.jl")

using SparseArrays
using PartitionedArrays
#=
using MPI 
mpiexec(cmd->run(`$cmd -np 4 julia --project=. src/Test/accumulate_ptest.jl`))
=#

np = 4
n = 31
const ranks = distribute_with_mpi(LinearIndices((np,)))
#const ranks = LinearIndices((np,))

A = ppoisson(n);


out = accumulate_psparse(A)


map(out) do o
    @show o
end


















# map(local_values(A), A.row_partition, A.col_partition) do Sparse_mat, rows, cols
#     @show findnz(ordered_local_transposed_full_SparseMatrixCSC(Sparse_mat,rows, cols, size(A,2),transpose=false))
#     @show findnz(ordered_local_SparseMatrixCSC(Sparse_mat,rows, cols))
# end
# b_acc = accumulate_pvector(b)
# dense_A = accumulate_psparse(A)
# Is,Js,Vs = extract_IJV(A)
# check_symmetric(A)