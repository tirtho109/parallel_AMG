include("../utility.jl")
include("../myConstants.jl")
include("../strength_psparse.jl")
include("../splitting_psparse.jl")
include("../smoother_psparse.jl")
#include("matmatmul_psparse.jl")
include("../matmatmul_helper.jl")
include("../classical_psparse.jl")
using SparseArrays
using PartitionedArrays
#=
using MPI 
mpiexec(cmd->run(`$cmd -np 4 julia --project=. src\\Test\\transpose_psparse_test.jl`))
=#

np = 4
n = 10
const ranks = distribute_with_mpi(LinearIndices((np,)))
#const ranks = LinearIndices((np,))

row_partition = uniform_partition(ranks,n) 
in_partition_rows = map(row_partition) do rows
        global_rows = own_to_global(rows)
        length(global_rows)
end             
row_partition = variable_partition(in_partition_rows, sum(in_partition_rows))

    # set up the Poisson matrix A
IJV = map(row_partition) do row_indices 
    I,J,V = Int[], Int[], Float64[]
    for global_row in local_to_global(row_indices) 
        push!(I, global_row)
        push!(J, global_row)
        push!(V, 1.0)
    end
    I,J,V
end
I,J,V = tuple_of_arrays(IJV)
col_partition = row_partition

A = psparse!(I,J,V, row_partition, col_partition) |> fetch

At = transpose_psparse(A)
map(ranks) do rank
    if rank == 1
        @show size(A), size(At)
    end
end


B = ppoisson(100)
Bt = transpose_psparse(B)

#check_symmetric(B)




















# map(local_values(A), A.row_partition, A.col_partition) do Sparse_mat, rows, cols
#     @show findnz(ordered_local_transposed_full_SparseMatrixCSC(Sparse_mat,rows, cols, size(A,2),transpose=false))
#     @show findnz(ordered_local_SparseMatrixCSC(Sparse_mat,rows, cols))
# end
# b_acc = accumulate_pvector(b)
# dense_A = accumulate_psparse(A)
# Is,Js,Vs = extract_IJV(A)
# check_symmetric(A)