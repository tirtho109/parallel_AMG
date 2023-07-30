# hello_mpi.jl
using PartitionedArrays
include("../utility.jl")


np = 4
n = 100
const ranks = distribute_with_mpi(LinearIndices((np,)))
#const ranks = LinearIndices((np,))
row_partition = uniform_partition(ranks,n)

IV = map(row_partition) do row_indices
    I,V = Int[], Float64[]
    for global_row in local_to_global(row_indices)
        if global_row == 1
            v = 1.0
        elseif global_row == n
            v = -1.0
        else
            continue
        end
        push!(I,global_row)
        push!(V,v)
    end
    I,V
end
I,V = tuple_of_arrays(IV)
b = pvector!(I,V,row_partition) |> fetch

map(local_values(b)) do local_b
   #println("I am proc $rank of $np.")
   @show size(local_b)
end

A = ppoisson(n);

map(local_values(A), A.col_partition, ranks) do local_A, 
                                                cols, 
                                                rank
    @show size(local_A), ghost_to_global(cols), rank
end

At = transpose_psparse(A);
map(local_values(At), At.col_partition, ranks) do local_A, 
    cols, 
    rank
    @show size(local_A), ghost_to_global(cols), rank
end
#=
using MPI 
mpiexec(cmd->run(`$cmd -np 4 julia --project=. hello_mpi.jl`))
=#



