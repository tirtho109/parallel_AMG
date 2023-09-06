include("utility.jl")
include("strength_psparse.jl")
include("splitting_psparse.jl")
include("smoother_psparse.jl")
#include("matmatmul_psparse.jl")
include("matmatmul_helper.jl")

const bs = 1

function ruge_stuben(_A::TA,
                    ::Type{Val{bs}}=Val{1};
                    strength = Classical(0.25),
                    CF = RS(),
                    presmoother = GaussSeidel(),
                    postsmoother = GaussSeidel(),
                    max_levels = 10,
                    max_coarse = 10,
                    coarse_solver = Pinv, kwargs...) where {TA<:PSparseMatrix}
    A = _A
    levels = Vector{Level{TA,TA,TA}}()

    w = MultiLevelWorkspace(Val{bs}, eltype(A))
    residual!(w, partition(axes(A,1)))

    # need to set last final_A as dense???
    while length(levels) + 1 < max_levels && size(A,1) > max_coarse
        A = extend_heirarchy!(levels, strength, CF, A)
        #coarse_x!(w, size(A,1))
        coarse_x!(w, A.col_partition)
        #coarse_b!(w, size(A,1))
        coarse_b!(w, A.row_partition)
        #residual!(w, size(A,1))
        residual!(w, A.row_partition)
    end

    MultiLevel(levels, A, coarse_solver(A), presmoother, postsmoother, w)
end


function extend_heirarchy!(levels, strength, CF, A::PSparseMatrix{Ti,Tv}) where {Ti,Tv}

    At = deepcopy(A) #considered symmetric
    S,T = strength(At); # in-our-case: S == T' == T
    splitting = CF(S)
    P, R = direct_interpolation(At, T, splitting);
    push!(levels, Level(A, P, R))
    return mat_mul(R, mat_mul(A,P))
end


function direct_interpolation(At::PSparseMatrix, T::PSparseMatrix, splitting::PVector)

    map(local_values(T), local_values(At)) do T_rank, At_rank
        fill!(T_rank.nzval, eltype(At_rank)(1))
        T_rank .= At_rank .* T_rank
    end
    
    Pp = rs_direct_interpolation_pass1(T, splitting) 
    
    R = rs_direct_interpolation_pass2(At, T, splitting, Pp)

    P = transpose_psparse(R)

    P, R
end

function rs_direct_interpolation_pass1(T::PSparseMatrix, splitting::PVector)
    # np = length(T.row_partition)
    # ranks = LinearIndices((np,)) 

    in_partition_indices = map(partition(axes(splitting,1))) do indices 
        global_indices = own_to_global(indices)
        length(global_indices)+1        # colptr has [size(T,1) + 1]
    end             
    index_partition_Bp = variable_partition(in_partition_indices, sum(in_partition_indices))

    #global_Bp = []

    global_Bp = map(own_values(T), local_values(splitting)) do T_loc, splitting_loc
        # only T_own is needed, as boundaries are considered C_NODE
        T_loc = sparse(T_loc)
        # splitting_loc = splitting.vector_partition[rank]

        n_loc = size(T_loc, 2)
        Bp_loc = ones(Int, n_loc+1)
        nnzplus1 = 1

        # if C_NODE add nnzplus1 += 1 and continue
        # if F_NODE check #of strongly connected C_NODEs (nnzplus1 += 1)
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
        #push!(global_Bp, Bp_loc);
        Bp_loc
    end
    Bp = PVector(global_Bp, index_partition_Bp)
    #consistent!(Bp) |> wait
    return Bp
end

function rs_direct_interpolation_pass2(At::PSparseMatrix,
                                        T::PSparseMatrix,
                                        splitting::PVector,
                                        Bp::PVector)

    # np = length(T.row_partition)
    # ranks = LinearIndices((np,)) 
    
    # set row_partition for next A or current R_P of R or current C_P of P
    num_of_rows_next_A = map(own_values(splitting)) do splitting_loc
                                count(!iszero, splitting_loc)
                        end
    row_partition_next_A = variable_partition(num_of_rows_next_A, sum(num_of_rows_next_A))

    Tv = Float64
    Ti = Int64

    IJV = map(partition(axes(T,1)), 
                row_partition_next_A, 
                own_values(Bp), 
                own_values(splitting),
                own_values(T),
                own_values(At)) do T_rows, 
                                        next_A_rows, 
                                        local_Bp, 
                                        local_splitting,
                                        local_T,
                                        local_At

        I,J,V = Int[], Int[], Float64[]
        #=
        Bp = colptr
        Bx = val
        Bj = rowval
        =#
        
        starting_col = minimum(own_to_global(T_rows)) -1
        starting_row = minimum(own_to_global(next_A_rows))-1

        # local_Bp = Bp.vector_partition[rank]
        # To match the typeof in creating of SparseMatrixCSC
        local_Bp = collect(local_Bp)
        # local_splitting = splitting.vector_partition[rank]
        
        local_T = sparse(local_T)
        local_At = sparse(local_At)
        
        local_Bx = zeros(Tv, local_Bp[end] - 1)
        local_Bj = zeros(Ti, local_Bp[end] - 1)

        n = size(local_At, 1)

        for i = 1:n
            if local_splitting[i] == C_NODE
                local_Bj[local_Bp[i]] = i
                local_Bx[local_Bp[i]] = 1
            else
                sum_strong_pos = zero(Tv)   # (S+)
                sum_strong_neg = zero(Tv)   # (S-)

                for j in nzrange(local_T, i)
                    row = local_T.rowval[j]
                    sval = local_T.nzval[j]

                    if local_splitting[row] == C_NODE
                        if sval < 0
                            sum_strong_neg += sval          # ∑aᵢₖ(-)
                        else
                            sum_strong_pos += sval          # ∑aᵢₖ(+)
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
                            sum_all_neg += aval             # ∑aᵢⱼ(-)
                        else
                            sum_all_pos += aval             # ∑aᵢⱼ(+)
                        end
                    end
                end

                if sum_strong_pos == 0                  
                    beta = zero(diag)
                    if diag >= 0
                        diag += sum_all_pos                 # if Pᵢ(+) = ∅, aᵢᵢ = ∑aᵢⱼ(+)
                    end
                else
                    beta = sum_all_pos / sum_strong_pos     # βᵢ = ∑aᵢⱼ(+)  ÷ ∑aᵢₖ(+)
                end

                if sum_strong_neg == 0                  
                    alpha = zero(diag)
                    if diag < 0
                        diag += sum_all_neg                 # if Pᵢ(-) = ∅, aᵢᵢ = ∑aᵢⱼ(-)
                    end
                else
                    alpha = sum_all_neg / sum_strong_neg     # αᵢ = ∑aᵢⱼ(-)  ÷ ∑aᵢₖ(-)
                end

                if isapprox(diag, 0, atol=eps(Tv))
                    neg_coeff = Tv(0)
                    pos_coeff = Tv(0)
                else
                    neg_coeff = alpha / diag #---------(1)
                    pos_coeff = beta / diag  #---------(2)
                end

                nnz = local_Bp[i]
                for j in nzrange(local_T, i)
                    row = local_T.rowval[j]
                    sval = local_T.nzval[j]
                    if local_splitting[row] == C_NODE
                        local_Bj[nnz] = row
                        if sval < 0
                            local_Bx[nnz] = (-1) * (neg_coeff * sval)   #(1): wᵢⱼ = - (αᵢ * aᵢₖ) ÷ aᵢᵢ  [ϵ Pᵢ(-)] 
                        else                                            # multiplying (-1)
                            local_Bx[nnz] = (-1) * (pos_coeff * sval)   #(2): wᵢⱼ = - (βᵢ * aᵢₖ) ÷ aᵢᵢ  [ϵ Pᵢ(+)]
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
        #@show typeof(local_Bp), typeof(local_Bj), typeof(local_Bx)
        local_R = SparseMatrixCSC(isempty(local_Bj) ?  0 : maximum(local_Bj), size(local_At, 1), local_Bp, local_Bj, local_Bx)

        # normalize columns to 1
        normalize_columns!(local_R)

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
    R = psparse!(I,J,V, row_partition_next_A, partition(axes(T,1))) |> fetch
    return R
end


"""
normalize columns of R to 1
"""
function normalize_columns!(x::SparseMatrixCSC{Float64, Int64})                                                                                      
    ncols = size(x, 2)                                                                                                                               
    for col in 1:ncols                                                                                                                               
        col_sum = sum(x[:, col])                                                                                                                     
        if abs(col_sum - 1.0) > 1e-8                                                                                                                 
            x[:, col] ./= col_sum                                                                                                                    
        end                                                                                                                                          
    end                                                                                                                                              
    return x                                                                                                                                         
end

function create_PR(A::PSparseMatrix)
    # t = PTimer(ranks)
    # tic!(t)

    At = deepcopy(A)
    strength = Classical()
    S,T = strength(At);
    CF = RS();
    splitting = CF(S)
    P,R = direct_interpolation(At, T, splitting);
    
    # toc!(t, "sleep")
    # display(t)
    return P,R
end
