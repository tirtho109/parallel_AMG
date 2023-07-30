include("../utility.jl")
include("../strength_psparse.jl")
using SparseArrays
using PartitionedArrays
#=
using MPI 
mpiexec(cmd->run(`$cmd -np 4 julia --project=. strength_test.jl`))
=#

np = 4
n = 15
const ranks = distribute_with_mpi(LinearIndices((np,)))
#const ranks = LinearIndices((np,))

A = ppoisson(n);

b = rhs(A)

x = create_x(A,b)

strength = Classical(0.25)

S,T = strength(A);

map(local_values(S)) do loc
    @show findnz(loc)
end

map(local_values(S), local_values(T)) do loc_S, loc_T
    @show loc_S == loc_T
end

# c = Classical(0.25)
# θ = c.θ
# mA, nA = size(A)

# in_partition_cols = map(A.col_partition) do cols #changed
#     global_cols = own_to_global(cols)
#     length(global_cols)
# end  
# col_partition = variable_partition(in_partition_cols, sum(in_partition_cols))

# IJV = map(local_values(A), A.row_partition, A.col_partition) do A_loc, rows, cols
#     I, J, V = Int[], Int[], Float64[]
#     # considering transposed as A = At
#     At = ordered_local_transposed_full_SparseMatrixCSC(A_loc, rows, cols, nA)
#     m, n = size(At)
#     T = deepcopy(At)
#     # set min col val
#     col = minimum(own_to_global(cols))-1

#     for i = 1:n
#         _m = find_max_off_diag(T,i,col)
#         threshold = θ * _m
#         for j in nzrange(T, i)
#             row = T.rowval[j]
#             val = T.nzval[j]

#             if row != (i+col)
#                 if abs(val) >= threshold
#                     T.nzval[j] = abs(val)
#                 else
#                     T.nzval[j] = 0
#                 end
#             end

#         end
#     end
#     dropzeros!(T)
#     scale_cols_by_largest_entry!(T)
#     x,y,z = findnz(T)
#     y = y .+ col
#     for (i, j, v) in zip(x, y, z)  
#         # switching order to transpose nz_indices
#         # If Symmetric, it's alright, no further transpose                         
#         push!(I, j)                                         
#         push!(J, i)                                         
#         push!(V, v)                                         
#     end
#     I,J,V
#     #@show size(I), size(J)
# end
# I,J,V = tuple_of_arrays(IJV)
# T = psparse!(I,J,V,A.row_partition, A.col_partition) |> fetch 

# S = transpose_psparse(T)
# x,y,z = extract_offDiag_transposed_global_IJV(T)
# dest = find_dest_from_I(x,T)
# Is, Js, Vs, Ds = get_remote_IJVD(x,y,z,dest)
# rcv_I, rcv_J, rcv_V = exchange_remote_IJV(Is,Js, Vs, Ds)
# S = make_transpose(T, rcv_I, rcv_J, rcv_V)

# map(S.col_partition) do I
#     @show ghost_to_global(I)
# end
# map(dest) do i
#     @show i
# end
# own_global = map(A.col_partition) do cols
#     own_to_global(cols)
# end
# all_own_global = gather(own_global, destination=:all)
# map(all_own_global) do all_og
#     @show all_og
# end
























# map(local_values(A), A.row_partition, A.col_partition) do Sparse_mat, rows, cols
#     @show findnz(ordered_local_transposed_full_SparseMatrixCSC(Sparse_mat,rows, cols, size(A,2),transpose=false))
#     @show findnz(ordered_local_SparseMatrixCSC(Sparse_mat,rows, cols))
# end
# b_acc = accumulate_pvector(b)
# dense_A = accumulate_psparse(A)
# Is,Js,Vs = extract_IJV(A)
# check_symmetric(A)