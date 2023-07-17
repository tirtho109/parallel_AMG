include("utility.jl")

#=
need to fix ruge_stuben. A is different
coarse_presmoother = CoarseGaussSeidel(), #added
coarse_postsmoother = CoarseGaussSeidel(), #added
=#
function ruge_stuben(_A::TA;
                    strength = Classical(0.25),
                    CF = RS(),
                    presmoother = GaussSeidel(),
                    postsmoother = GaussSeidel(),
                    max_levels = 10,
                    max_coarse = 10,
                    coarse_solveer = Pinv, kwargs...)
    A = _A
    levels = Vector{Level{TA,TA,TA}}()

    w = MultiLevelWorkspace(Val{bs}, eltype(A))
    residual!(w, size(A,1)) ### need to varify

    # need to set last final_A as dense 
    while length(levels) + 1 < max_levels && size(A,1) > max_coarse
        A = extend_heirarchy!(levels, strength, CF, A)
        coarse_x!(w, size(A,1))
        coarse_b!(w, size(A,1))
        residual!(w, size(A,1))
    end

    MultiLevel(levels, A, coarse_solveer(A), presmoother, postsmoother,coarse_presmoother, coarse_postsmoother, w)
end


function extend_heirarchy!(levels, strength, CF, A::PSparseMatrix{Ti,Tv}) where {Ti,Tv}
    At = A #considered symmetric
    # if symmetric
    #     At = A
    # else
    #     At = transpose_psparse(A)
    # end
    # S = strength(At)
    S,T = strength(At); # in-our-case: S == T' == T
    splitting = CF(S)
    P, R = direct_interpolation(At, T, splitting);
    push!(levels, Level(A, P, R))
    B = mat_mat_mul(R, A)
    return mat_mat_mul(B,P)
end


function direct_interpolation(At, T, splitting)
    np = length(T.row_partition)
    ranks = LinearIndices((np,)) 

    map(ranks) do rank
        T_rank = T.matrix_partition[rank]
        At_rank = At.matrix_partition[rank]

        fill!(T_rank.nzval, eltype(At_rank)(1))
        T_rank .= At_rank .* T_rank
    end
    
    Pp = rs_direct_interpolation_pass1(T, splitting) 
    
    R = rs_direct_interpolation_pass2(At, T, splitting, Pp)

    P = transpose_psparse(R)

    P, R
end

function rs_direct_interpolation_pass1(T::PSparseMatrix, splitting::PVector)
    np = length(T.row_partition)
    ranks = LinearIndices((np,)) 

    in_partition_indices = map(splitting.index_partition) do indices #changed
        global_indices = own_to_global(indices)
        length(global_indices)+1
    end             
    index_partition_Bp = variable_partition(in_partition_indices, sum(in_partition_indices))

    global_Bp = []

    map(ranks) do rank
        # T_own is we need, as boundaries are considered C_NODE
        T_loc = sparse(own_values(T)[rank])
        splitting_loc = splitting.vector_partition[rank]

        n_loc = size(T_loc, 2)
        Bp_loc = ones(Int, n_loc+1)
        nnzplus1 = 1

        for i = 1:n_loc
            if splitting_loc[i] == C_NODE
                nnzplus1 += 1
            else
                for j in nzrange(T_loc,i)
                    row = T_loc.rowval[j]
                    if splitting_loc[row] == C_NODE
                        nnzplus1 += 1
                    end
                end
            end
            Bp_loc[i+1] = nnzplus1
        end
        push!(global_Bp, Bp_loc);
    end
    Bp = PVector(global_Bp, index_partition_Bp)

    return Bp
end


# # calculates the number of nonzeros in each column of the interpolation matrix
# function rs_direct_interpolation_pass12(T::PSparseMatrix, splitting::PVector)
    
#     np = length(T.row_partition)
#     ranks = LinearIndices((np,)) 

#     mT, nT = size(T)
    
#     global_Bp = []

#     len_Bp = size(T,1)+ np
#     #index_partition_Bp = uniform_partition(ranks, len_Bp) # changed

#     in_partition_indices = map(splitting.index_partition) do indices #changed
#         global_indices = own_to_global(indices)
#         length(global_indices)
#     end             
#     index_partition_Bp = variable_partition(in_partition_indices, sum(in_partition_indices))
    
#     #index_partition_Bp = splitting.index_partition


#     map(ranks) do rank
#         #starting_row = rank==1 ? 0 : sum(length(T.row_partition[i-1]) for i in 2:rank)
#         starting_row = minimum(own_to_global(T.row_partition[rank]))-1
#         #@show starting_row
#         local_T = ordered_local_transposed_full_SparseMatrixCSC(T.matrix_partition[rank], T.row_partition[rank], T.col_partition[rank], nT)
#         local_splitting = splitting.vector_partition[rank]
#         local_n = size(local_T,2)
#         # nnzplus1 = rank==1 ? 1 : 0
#         # local_Bp = rank == 1 ? ones(Int, local_n + 1) : ones(Int, local_n)
#         nnzplus1 = 1
#         local_Bp = ones(Int, local_n + 1)

#         for i =1:local_n
#             if local_splitting[i] == C_NODE
#                 nnzplus1 += 1
#             else
#                 for j in nzrange(local_T,i)
#                     row = local_T.rowval[j]
#                     if local_splitting[row - starting_row] == C_NODE 
#                         nnzplus1 += 1
#                     end
#                 end
#             end
#             local_Bp[i+1] = nnzplus1
#             # if rank == 1
#             #     local_Bp[i+1] = nnzplus1
#             # else
#             #     local_Bp[i] = nnzplus1
#             # end
#         end
#         push!(global_Bp, local_Bp)
#     end
#     #@show global_Bp

#     global_Bp_psparse = PVector(global_Bp, index_partition_Bp)

#     # for i in 1:length(global_Bp)-1                                                                      
#     #     max_val = maximum(global_Bp[i])                                                                 
#     #     global_Bp[i+1] .+= max_val                                                                      
#     # end
#     # #@show global_Bp
#     # global_Bp = reduce(vcat, global_Bp)

#     return global_Bp_psparse
#  end

function rs_direct_interpolation_pass2(At::PSparseMatrix,
                                        T::PSparseMatrix,
                                        splitting::PVector,
                                        Bp::PVector)

    np = length(T.row_partition)
    ranks = LinearIndices((np,)) 
    
    # set row_partition for next A or current R_P of R or current C_P of P
    num_of_rows_next_A = [count(!iszero,splitting.vector_partition[i]) for i in ranks]
    row_partition_next_A = variable_partition(num_of_rows_next_A, sum(num_of_rows_next_A))

    Tv = Float64
    Ti = Int64

    IJV = map(ranks) do rank

        I,J,V = Int[], Int[], Float64[]

        
        starting_col = minimum(own_to_global(T.row_partition[rank])) -1
        starting_row = minimum(own_to_global(row_partition_next_A[rank]))-1
        local_Bp = Bp.vector_partition[rank]
        local_splitting = splitting.vector_partition[rank]
        
        local_T = sparse(own_values(T)[rank])
        local_At = sparse(own_values(At)[rank])
        
        local_Bx = zeros(Tv, local_Bp[end] - 1)
        local_Bj = zeros(Ti, local_Bp[end] - 1)

        # local_T = T.matrix_partition[rank]
        # local_At = At.matrix_partition[rank]
        #starting_col = rank==1 ? 0 : sum(length(T.row_partition[i-1]) for i in 2:rank)
        #starting_row = rank==1 ? 0 : maximum(row_partition_next_A[rank-1])

        n = size(local_At, 1)

        for i = 1:n
            if local_splitting[i] == C_NODE
                local_Bj[local_Bp[i]] = i
                local_Bx[local_Bp[i]] = 1
            else
                sum_strong_pos = zero(Tv)
                sum_strong_neg = zero(Tv)
                for j in nzrange(local_T, i)
                    row = local_T.rowval[j]
                    sval = local_T.nzval[j]
                    if local_splitting[row] == C_NODE
                        if sval < 0
                            sum_strong_neg += sval
                        else
                            sum_strong_pos += sval
                        end
                    end
                end
                sum_all_pos = zero(Tv)
                sum_all_neg = zero(Tv)
                diag = zero(Tv)
                for j in nzrange(local_At, i)
                    row = local_At.rowval[j]
                    aval = local_At.nzval[j]
                    if row == i
                        diag += aval
                    else
                        if aval < 0
                            sum_all_neg += aval
                        else
                            sum_all_pos += aval
                        end
                    end
                end

                if sum_strong_pos == 0
                    beta = zero(diag)
                    if diag >= 0
                        diag += sum_all_pos
                    end
                else
                    beta = sum_all_pos / sum_strong_pos
                end

                if sum_strong_neg == 0
                    alpha = zero(diag)
                    if diag < 0
                        diag += sum_all_neg
                    end
                else
                    alpha = sum_all_neg / sum_strong_neg
                end

                if isapprox(diag, 0, atol=eps(Tv))
                    neg_coeff = Tv(0)
                    pos_coeff = Tv(0)
                else
                    neg_coeff = alpha / diag
                    pos_coeff = beta / diag
                end

                nnz = local_Bp[i]
                for j in nzrange(local_T, i)
                    row = local_T.rowval[j]
                    sval = local_T.nzval[j]
                    if local_splitting[row] == C_NODE
                        local_Bj[nnz] = row
                        if sval < 0
                            local_Bx[nnz] = abs(neg_coeff * sval)
                        else
                            local_Bx[nnz] = abs(pos_coeff * sval)
                        end
                        nnz += 1
                    end
                end
            end
        end

        m = zeros(Ti, n)
        summ = zero(Ti)
        for i = 1:n
            m[i] = summ
            summ += local_splitting[i]
        end
        local_Bj .= m[local_Bj] .+ 1

        local_Bx, local_Bj, local_Bp

        local_R = SparseMatrixCSC(isempty(local_Bj) ?  0 : maximum(local_Bj), size(local_At, 1), local_Bp, local_Bj, local_Bx)

        local_I, local_J, local_V = findnz(local_R)
        local_I = local_I .+ starting_row
        local_J = local_J .+ starting_col

        for (i,j, v) in zip(local_I, local_J, local_V)
            push!(I,i)
            push!(J,j)
            push!(V,v)
        end
        #@show summ
        I,J,V
    end 
    I,J,V = tuple_of_arrays(IJV)
    R = psparse!(I,J,V, row_partition_next_A, T.row_partition) |> fetch
    return R
end

function create_PR(A::PSparseMatrix)
    t = PTimer(ranks)
    tic!(t)

    At = deepcopy(A)
    strength = Classical(0.25)
    S,T = strength(At);
    CF = RS();
    splitting = CF(S)
    P,R = direct_interpolation(At, T, splitting);
    
    toc!(t, "sleep")
    display(t)
    return P,R
end
