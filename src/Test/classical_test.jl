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
mpiexec(cmd->run(`$cmd -np 4 julia --project=. src/Test/classical_test.jl`))
=#

np = 4
n = 31
const ranks = distribute_with_mpi(LinearIndices((np,)))
#const ranks = LinearIndices((np,))

A = ppoisson(n);
mA, nA = size(A)

b = rhs(A)
x = create_x(A,b)
strength = Classical(0.25)
S,T = strength(A);

CF = RS()
splitting = CF(S)
P,R = create_PR(A);

# map(local_values(P), local_values(R), ranks) do loc_P, loc_R, rank
#     if rank==1
#         @show loc_P, loc_R
#     end
# end

# It looks like transpose works. But recheck again.
Pt = transpose_psparse(P);
map(local_values(R), local_values(Pt)) do r, p
    @show r == p , size(p),size(r)
end





















# map(local_values(A), A.row_partition, A.col_partition) do Sparse_mat, rows, cols
#     @show findnz(ordered_local_transposed_full_SparseMatrixCSC(Sparse_mat,rows, cols, size(A,2),transpose=false))
#     @show findnz(ordered_local_SparseMatrixCSC(Sparse_mat,rows, cols))
# end
# b_acc = accumulate_pvector(b)
# dense_A = accumulate_psparse(A)
# Is,Js,Vs = extract_IJV(A)
# check_symmetric(A)