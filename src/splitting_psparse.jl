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
