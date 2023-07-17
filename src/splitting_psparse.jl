include("utility.jl")

const F_NODE = 0
const C_NODE = 1
const U_NODE = 2

struct RS
end


function remove_diag1!(A::PSparseMatrix)

    ranks = LinearIndices((length(A.row_partition),))
	mA, nA = size(A)
    row_partition_A = uniform_partition(ranks, mA)
    col_partition_A = uniform_partition(ranks, nA)

    I,J,V = map(ranks) do rank
        I, J, V = Int[], Int[], Float64[]
        row = 0;
        row = rank==1 ? 0 : sum(length(row_partition_A[i-1]) for i in 2:rank)

        ordered_local_mat = ordered_local_transposed_full_SparseMatrixCSC(A.matrix_partition[rank], A.row_partition[rank], A.col_partition[rank], nA, transpose=false)
        n = size(ordered_local_mat, 2) # over the columns

        for i = 1:n
            for j in nzrange(ordered_local_mat, i)
                if ordered_local_mat.rowval[j] == (i-row)
                    ordered_local_mat.nzval[j] = 0
                end
            end
        end
        dropzeros!(ordered_local_mat)
        x,y,z = findnz(ordered_local_mat)
        x = x.+ row
        for (i,j,v) in zip(x,y,z)
            push!(I,i)
            push!(J,j)
            push!(V,v)
        end
        I,J,V
    end |> tuple_of_arrays
    #I,J,V = tuple_of_arrays(IJV)
    A = psparse!(I,J,V, row_partition_A, col_partition_A) |> fetch
    A
end

function remove_diag!(A::PSparseMatrix)
    map(ranks) do rank
        A_loc = A.matrix_partition[rank]
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
	S = remove_diag!(S)
    T = transpose_psparse(S)
	RS_CF_splitting(S, T)
end

function RS_CF_splitting(SP::PSparseMatrix, TP::PSparseMatrix)
    np = length(SP.row_partition)
    ranks = LinearIndices((np,)) 
    whole_splitting = []

    map(ranks) do rank
        # ghost_col_sp = ghost_to_local(SP.col_partition[rank])
        # local_matrix = SP.matrix_partition[rank]
        # ghost_rows,_,_ = findnz(local_matrix[:,ghost_col_sp])
        ghost_rows,_,_ = findnz(sparse(own_ghost_values(SP)[rank])) #changed
        ghost_rows = unique(ghost_rows)
        S = sparse(own_values(SP)[rank])
        #S = local_matrix[:,1:ghost_col_sp[1]-1]
        T = deepcopy(S)
        # SPp = SP.matrix_partition[rank].colptr
        # SPr = SP.matrix_partition[rank].rowval
        # ghost_rows = []
        # for col in ghost_col_sp
        #     ghost_col_indices = SPp[col]:(SPp[col+1]-1)
        #     ghost_row_indices = SPr[ghost_col_indices]
        #     for row in ghost_row_indices
        #         push!(ghost_rows, row)
        #     end
        # end

        n = size(S,1)

        splitting = fill(U_NODE, n)
        for row in ghost_rows
            splitting[row] = C_NODE
        end

	    lambda = zeros(Int, n)
        Tp = T.colptr
        Tj = T.rowval
        Sp = S.colptr
        Sj = S.rowval

        interval_ptr = zeros(Int, n+1)
        interval_count = zeros(Int, n+1)
        index_to_node = zeros(Int,n)
        node_to_index = zeros(Int,n)

        #@show rank, Sp

        for i = 1:n
            # compute lambda[i] - the number of nodes strongly coupled to node i
            lambda[i] = Sp[i+1] - Sp[i]
            interval_count[lambda[i] + 1] += 1
        end
        
        # initial interval_ptr
        @views accumulate!(+, interval_ptr[2:end], interval_count[1:end-1])

        interval_count .= 0 # temporarily zeroed, goes back to its original at end of loop
        for i = 1:n
            lambda_i = lambda[i] + 1
            interval_count[lambda_i] += 1
            index = interval_ptr[lambda_i] + interval_count[lambda_i]
            index_to_node[index] = i
            node_to_index[i]     = index
        end

        # all nodes which no other nodes are strongly coupled to become F nodes
        for i = 1:n
            if lambda[i] == 0
                splitting[i] = F_NODE
            end
        end
        # i = index_to_node[top_index] can either refer to an F node or to the U node with the
        #	highest lambda[i].

        # index_to_node[interval_ptr[i]+1 : interval_ptr[i+1]] includes the set of U nodes with 
        #	i-1 nodes strongly coupled to them, and other "inactive" F and C nodes.
        
        # C nodes are always in index_to_node[top_index:n]. So at the end of the last 
        #	non-empty interval_ptr[i]+1 : interval_ptr[i+1] will be all the C nodes together 
        #	with some other inactive F nodes.
        
        # when lambda_k+1 > lambda_i, i.e. lambda_k == lambda_i,  where lambda_k+1 = lambda_i+1 
        #	is the new highest lambda[i], the line: `interval_ptr[lambda_k+1] = new_pos - 1`
        #	pushes the all the inactive C and F points to the end of the next now-non-empty 
        #	interval.
        for top_index = n:-1:1
            i = index_to_node[top_index]
            lambda_i = lambda[i] + 1
            interval_count[lambda_i] -= 1

            splitting[i] == F_NODE && continue
            # added 
            if i ∉ ghost_rows
                @assert splitting[i] == U_NODE
            end
            splitting[i] = C_NODE
            for j in nzrange(S, i)
                row = S.rowval[j]
                if splitting[row] == U_NODE
                    splitting[row] = F_NODE

                    # increment lambda for all U nodes that node `row` is strongly coupled to
                    for k in nzrange(T, row)
                        rowk = T.rowval[k]

                        if splitting[rowk] == U_NODE
                            # to ensure `intervalcount` is inbounds
                            lambda[rowk] >= n - 1 && continue

                            # move rowk to the end of its current interval
                            lambda_k = lambda[rowk] + 1
                            old_pos  = node_to_index[rowk]
                            new_pos  = interval_ptr[lambda_k] + interval_count[lambda_k]

                            swap_node = index_to_node[new_pos]
                            (index_to_node[old_pos], index_to_node[new_pos]) = (swap_node, rowk)
                            node_to_index[rowk] = new_pos
                            node_to_index[swap_node] = old_pos

                            # increment lambda[rowk]
                            lambda[rowk] += 1

                            # update intervals
                            interval_count[lambda_k]   -= 1
                            interval_count[lambda_k+1] += 1
                            interval_ptr[lambda_k+1]    = new_pos - 1
                        end
                    end
                end
            end

            # decrement lambda for all U nodes that node i is strongly coupled to
            for j in nzrange(T, i)
                row = T.rowval[j]
                if splitting[row] == U_NODE
                    # to ensure `intervalcount` is inbounds
                    lambda[row] == 0 && continue

                    # move row to the beginning of its current interval
                    lambda_j = lambda[row] + 1
                    old_pos  = node_to_index[row]
                    new_pos  = interval_ptr[lambda_j] + 1

                    swap_node = index_to_node[new_pos]
                    (index_to_node[old_pos], index_to_node[new_pos]) = (swap_node, row)
                    node_to_index[row] = new_pos
                    node_to_index[swap_node] = old_pos

                    # decrement lambda[row]
                    lambda[row] -= 1

                    # update intervals
                    interval_count[lambda_j]   -= 1
                    interval_count[lambda_j-1] += 1
                    interval_ptr[lambda_j]     += 1
                end
            end
        end
        splitting
        push!(whole_splitting, splitting);
    end
    splitting = PVector(whole_splitting, S.row_partition)
end



