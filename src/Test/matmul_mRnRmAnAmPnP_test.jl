include("../utility.jl")
include("../myConstants.jl")
include("../strength_psparse.jl")
include("../splitting_psparse.jl")
include("../smoother_psparse.jl")
include("../matmatmul_helper.jl")
include("../classical_psparse.jl")

using SparseArrays
using PartitionedArrays
#=
using MPI 
mpiexec(cmd->run(`$cmd -np 4 julia --project=. src/Test/matmul_mRnRmAnAmPnP_test.jl`))
=#

np = 4
n = 500
#const ranks = distribute_with_mpi(LinearIndices((np,)))
const ranks = LinearIndices((np,))

A = ppoisson(n);
# b = rhs(A)
# x = create_x(A,b)
# strength = Classical()
# S ,T = strength(A);
# CF = RS()
# splitting = CF(S)


P,R = create_PR(A);

result = mat_mul(A,P);
A2 = mat_mul(R,result);
map(ranks) do rank
    if rank==1
        @show size(A), size(A2)
    end
end

P2,R2 = create_PR(A2);
A3 = mat_mul(R2, mat_mul(A2,P2))


P3,R3 = create_PR(A3);
A4 = mat_mul(R3, mat_mul(A3,P3))

P4,R4 = create_PR(A4);
A5 = mat_mul(R4, mat_mul(A4,P4))
map(ranks) do rank
    if rank==1
        @show size(A), size(A2), size(A3), size(A4), size(A5)
    end
end

P5,R5 = create_PR(A5);
A6 = mat_mul(R5, mat_mul(A5,P5))
map(ranks) do rank
    if rank==1
        @show size(A), size(A2), size(A3), size(A4), size(A5), size(A6)
    end
end

P6,R6 = create_PR(A6);
A7 = mat_mul(R6, mat_mul(A6,P6))


# P3,R3 = create_PR(A3);
# A4 = mat_mul(R3, mat_mul(A3,P3))

map(ranks) do rank
    if rank==1
        @show size(A), size(A2), size(A3), size(A4), size(A5), size(A6), size(A7)
    end
end







# P = deepcopy(A)

# result = mat_mul(A,P);

# map(local_values(result), local_values(A), local_values(P)) do res, a, p
#     @show size(res), size(a), size(p)
# end
