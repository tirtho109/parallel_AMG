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
mpiexec(cmd->run(`$cmd -np 4 julia --project=. src/Test/pinv_test.jl`))
=#

np = 4
n = 15
const ranks = distribute_with_mpi(LinearIndices((np,)))

A = ppoisson(n);
b = rhs(A)
x = pzeros(A.col_partition)

coarse_x = accumulate_pvector(x)
coarse_b = accumulate_pvector(b)
coarse_A = deepcopy(A)
coarse_A = accumulate_psparse2(coarse_A)

coarse_solver = Pinv

map(ranks, coarse_x, coarse_b, coarse_solver(A)) do rank, cx, cb, mcs
    if rank == MAIN
        o = mul!(cx, mcs.pinvA, cb)
    else
        o = 0
    end
    o
end
map(coarse_x,ranks) do cx, rank
    if rank == 1
        @show size(cx)
    end
end

#coarse_x = gather(coarse_x, destination=:all)
map(coarse_x) do cx
    @show cx
    
end

coarse_x = emit(coarse_x, source=MAIN)
IV = map(coarse_x, A.col_partition) do cx, cols
    #@show cx
    I, V = Int[], Float64[]
    indices_and_values = [(index, value) for (index, value) in enumerate(cx)]
    indices = map(x -> x[1], indices_and_values)
    values = map(x -> x[2], indices_and_values)
    for global_col in own_to_global(cols)
        push!(I, global_col)
        push!(V, indices_and_values[global_col][2])
    end
    @show I,V
    I,V
end

I,value = tuple_of_arrays(IV)
coarse_x = pvector!(I,value,A.col_partition) |> fetch
consistent!(coarse_x) |>wait

map(local_values(coarse_x)) do iks
    @show iks
end
#coarse_x = vec_to_pvec(coarse_x, A)


# map(own_values(coarse_x)) do cx
#     @show size(cx), cx
# end

# map(ranks, coarse_solver(A)) do rank, cs
#     if rank==1
#         @show size(cs.pinvA)
#     end
# end

coarse_b = vec_to_pvec(coarse_b, A)

map(local_values(coarse_b)) do be
    @show be
end