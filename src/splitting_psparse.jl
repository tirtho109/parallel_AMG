include("utility.jl")
include("inner_splitting.jl")

# const F_NODE = 0
# const C_NODE = 1
# const U_NODE = 2

# struct RS
# end

function remove_diag!(A::PSparseMatrix)
    map(local_values(A)) do A_loc
        num_rows = size(A_loc, 1)
        #@show num_rows
        for i in 1:num_rows
            A_loc[i,i] = 0.0
        end
        dropzeros!(A_loc)
    end
    A    
end

function (::RS)(S)
	remove_diag!(S)
    T = transpose_psparse(S)
	RS_CF_splitting(S, T)
end

function RS_CF_splitting(SP::PSparseMatrix, TP::PSparseMatrix)
    
    whole_splitting= map(own_ghost_values(SP), own_values(SP)) do og_val_SP, o_val_SP

        ghost_rows,_,_ = findnz(sparse(og_val_SP)) #changed
        ghost_rows = unique(ghost_rows)

        S = sparse(o_val_SP)
        #T = deepcopy(S)
      
        n = size(S,1)

        splitting = fill(U_NODE, n)
        for row in ghost_rows
            splitting[row] = C_NODE         # set boundaries as C_NODE
        end

        rows_to_remove = sort(ghost_rows, rev=true)
        cols_to_remove = sort(ghost_rows, rev=true)
        S = S[setdiff(1:end, rows_to_remove), setdiff(1:end, cols_to_remove)]
	    T = deepcopy(S)
        split = RS_CF_splitting_inner(S,T)

        # Merge the inner_split vector into the split vector
        j = 1
        for i in 1:n
            if splitting[i] == U_NODE
                splitting[i] = split[j]
                j += 1
            end
        end
        splitting
    end
    splitting = PVector(whole_splitting, SP.row_partition)
end



"""
    CLJP - Parallel Coarsening
    References
    ----------
    .. [8] David M. Alber and Luke N. Olson
       "Parallel coarse-grid selection"
       Numerical Linear Algebra with Applications 2007; 14:611-643.

    Algorithm 4: CLJP
    ----------------------------------------
    where, D = {i : wᵢ > wⱼ, ∀ j ∈ Sᵢ ∪ Sᵢᵀ}
    and initialized, wᵢ = |Sᵢᵀ| + rand(i)
    where rand(i) is a random number in (0,1)
    ----------------------------------------
    Initialize: F = ∅, C = ∅
    1. for all i ∈ Ω do
    2.    wᵢ <- initial values
    3. end for
    4. while |C| + |F| ≠ n do
    5.      select independent set D
    6.      for all j ∈ D do 
    7.          C = C ∪ j 
    8.          for all k in set local to j do 
    9.              update wₖ
    10.             if wₖ == 0 then 
    11.                 F = F ∪ k
    12.             end if 
    13.          end for 
    14.     end for 
    15. end while
    """

    function inner_CLJP(S::SparseMatrixCSC, T::SparseMatrixCSC)
        n = size(S,1)
        unassigned = size(S,1)
        nonz = nnz(S)

        edgemark = ones(Int, nnz)
        weight = zeros(Float64, n)
        D = zeros(Int, n)
        Dlist = zeros(Int, n)
        splitting = fill(U_NODE, n)

        for i in 1:n
            weights[i] = length(nzrange(S,i)) + rand()  # check later i or j// S or T
        end

        # selection loop
        pass = 0
        while(unassigned > 0)
            pass += 1
            nD = 1
            for i in 1:n
                if(splitting[i] == U_NODE)
                    D[i] = 1
                    for j in nzrange(S,i)
                        if (splitting[j]==U_NODE && weight[j] > weight[i])
                            D[i] = 0;
                            break
                        end
                    end
                    if(D[i] == 1)
                        for j in nzrange(T,i)
                            if (splitting[j]==U_NODE && weight[j]>weight[i])
                                D[i] = 0
                                break
                            end
                        end
                    end
                    if (D[i] == 1)
                        Dlist[nD] = i
                        unassigned -= 1
                        nD += 1
                    end
                else
                    D[i] = 0
                end
            end #end for
            # nD number of C_NODEs are selected for current iteration 
            # Dlist contain each batch of newly selected C_NODE
            for i in 1:nD
                splitting[Dlist[i]] = C_NODE   
            end
            # end select independent set
            
            ## TODO
            # Update weights
            # nodes that influences C points are not good C points.
        end
    end

    # function CLJP(Strength::PSparseMatrix, color::Bool=False)
    #     remove_diag!(S)
    #     Strength_T = transpose_psparse(S);
    
    #     n = size(S,1)
        
    #     whole_splitting = map(local_values(Strength), local_values(Strength_T)) do S,T
    #         # N = size(S,1)
    #         F = Set{Int}(); # initialize F & C == ∅
    #         C = Set{Int}();

    #         S = sparse(S')
    #         T = sparse(T')
    #         Sp = S.colptr
    #         Sj = S.rowval
    #         Tp = T.colptr
    #         Tj = T.rowval

    #         Ω = size(S,2)
    #         weights = zeros(Float64, Ω)
    #         for j in 1:Ω
    #             weights[j] = length(nzrange(S,j)) + rand()
    #         end
    #         while (length(C) + length(F) ≠ Ω)
    #             #D = 
    #         end
    # end

    """
    Algorithm 5. CLJP-update-weights
    #########################################
    H1: Values at C-points are not interpolated; hence, neighbours that strongly
        influences a C-point are less valuable as potential C-points themselves.
    H2: If k and j both strongly depend on i ∈ C and j storngly influences k, 
        then j is less valueable as a potential C-point since k can be
        interpolated from i.
    #########################################
    1. for all i ∈ D do 
    2.      for all j : Sᵢⱼ ≠ 0 do [Heuristic 1]
    3.          wⱼ <- wⱼ -1
    4.          Sᵢⱼ <- 0
    5.      end for 
    6.      for all j : Sⱼᵢ ≠ 0 do [Heuristic 2] [s.t. Sⱼᵢ ⩵ Sᵢⱼᵀ]
    7.          Sⱼᵢ <- 0
    8.          for all k : Sₖⱼ ≠ 0 do
    9.              if Sₖⱼ ≠ 0 then 
    10.                 wⱼ <- wⱼ -1
    11.                 Sₖⱼ <- 0
    12.             end if
    13.         end for 
    14.     end for 
    15. end for
    """
