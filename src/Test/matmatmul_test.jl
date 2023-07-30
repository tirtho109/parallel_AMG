include("../utility.jl")
include("../myConstants.jl")
include("../strength_psparse.jl")
include("../splitting_psparse.jl")
include("../smoother_psparse.jl")
#include("matmatmul_psparse.jl")
include("../matmatmul_helper.jl")
using SparseArrays
using PartitionedArrays
#=
using MPI 
mpiexec(cmd->run(`$cmd -np 4 julia --project=. src/Test/matmatmul_test.jl`))
=#

np = 4
n = 31
const ranks = distribute_with_mpi(LinearIndices((np,)))
#const ranks = LinearIndices((np,))

A = ppoisson(n);
P = deepcopy(A)

mP, nP = size(P)
mA, nA = size(A)

in_partition_cols = map(partition(axes(P,2))) do cols
    global_cols = own_to_global(cols)
    length(global_cols)
end             
col_partition = variable_partition(in_partition_cols, sum(in_partition_cols))


# b = rhs(A)
# x = create_x(A,b)
# #step 1
# pre = GaussSeidel()
# pre(A,x,b)
# #step 2
# strength = Classical(0.25)
# S,T = strength(A);
# #step 3
# CF = RS()
# splitting = CF(S)
# #step 4
# # try mat_mat_mul (A*A)

#global_rows, owner, dest = extract_numrow_owner_dest(A)
global_rows, remote_owner, asked_by = extract_remote_rowsnum_onwers_dests(A)
# map(global_rows, remote_owner, asked_by) do gr, ro, rd
#     @show gr, ro, rd
# end

# do exchange with global_rows & Real_owner
extract_row, next_dest = exchange_remote_rowval(global_rows,remote_owner,asked_by)
# now extract the 
# map(extract_row, next_dest) do rows, d
#     @show rows, d
# end

rcv_I, rcv_J, rcv_V = extract_remote_rowsP(P,extract_row, next_dest)

# map(rcv_I, rcv_J, rcv_V) do i,j,v
#     @show i,j,v
# end

global_Pr = create_local_Pr(rcv_I, rcv_J, rcv_V)
# map(local_mat) do mat
#     @show findnz(mat)
# end


IJV = map(partition(axes(A,1)), 
            col_partition, 
            own_values(A),
            own_ghost_values(A),
            partition(axes(A,2)),
            own_values(P),
            local_values(P),
            partition(axes(P,2)),
            global_Pr) do rows, cols, own_A, 
                                og_A, cols_A, own_P,
                                loc_P, cols_P, Pr
    I,J,V = Int[], Int[], Float64[]

    #set min row-col val
    row = minimum(own_to_global(rows))-1
    col = minimum(own_to_global(cols))-1

    #Left diag & off-diag
    Ad = sparse(own_A)
    Ao = sparse(og_A)
    col_indices = ghost_to_global(cols_A)
    Ao .= Ao[:, sortperm(col_indices)]
    
    #right diag & off-diag
    Pd = sparse(own_P)
    Po = extract_offDiagMat(loc_P, cols_P, nP)

    #Pr = global_Pr[rank]

    for i in 1:size(Ad, 1)
        current_i = row + i 
        R = Numeric_calculation_of_one_row_of_AP(i, Ad, Ao, Pd, Po, Pr, col)
        #@show R
        for (j,val) in R
            push!(I, current_i);
            push!(J, j);
            push!(V, val);
        end
    end
    I,J,V
end
I,J,V = tuple_of_arrays(IJV) 

result = psparse!(I,J,V, A.row_partition, col_partition) |> fetch

map(partition(axes(result,2)), local_values(result)) do part, res
    @show part, size(res)
end
























# map(local_values(A), A.row_partition, A.col_partition) do Sparse_mat, rows, cols
#     @show findnz(ordered_local_transposed_full_SparseMatrixCSC(Sparse_mat,rows, cols, size(A,2),transpose=false))
#     @show findnz(ordered_local_SparseMatrixCSC(Sparse_mat,rows, cols))
# end
# b_acc = accumulate_pvector(b)
# dense_A = accumulate_psparse(A)
# Is,Js,Vs = extract_IJV(A)
# check_symmetric(A)