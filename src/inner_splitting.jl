function RS_CF_splitting_inner(S::SparseMatrixCSC, T::SparseMatrixCSC)

	n = size(S,1)

	lambda = zeros(Int, n)

	Tp = T.colptr
	Tj = T.rowval
	Sp = S.colptr
	Sj = S.rowval

	interval_ptr = zeros(Int, n+1)
	interval_count = zeros(Int, n+1)
	index_to_node = zeros(Int,n)
	node_to_index = zeros(Int,n)

    for i = 1:n
		# compute lambda[i] - the number of nodes strongly coupled to node i
		lambda[i] = Sp[i+1] - Sp[i]
        interval_count[lambda[i] + 1] += 1	# count how many node has the same lamda value
		#@show lambda[i], (lambda[i]+1), interval_count'
    end
	"""
	SP = 10x10
	(lambda[i], lambda[i] + 1, interval_count') = (1, 2, [0 1 0 0 0 0 0 0 0 0 0])
	(lambda[i], lambda[i] + 1, interval_count') = (2, 3, [0 1 1 0 0 0 0 0 0 0 0])
	(lambda[i], lambda[i] + 1, interval_count') = (2, 3, [0 1 2 0 0 0 0 0 0 0 0])
	(lambda[i], lambda[i] + 1, interval_count') = (2, 3, [0 1 3 0 0 0 0 0 0 0 0])
	(lambda[i], lambda[i] + 1, interval_count') = (2, 3, [0 1 4 0 0 0 0 0 0 0 0])
	(lambda[i], lambda[i] + 1, interval_count') = (2, 3, [0 1 5 0 0 0 0 0 0 0 0])
	(lambda[i], lambda[i] + 1, interval_count') = (2, 3, [0 1 6 0 0 0 0 0 0 0 0])
	(lambda[i], lambda[i] + 1, interval_count') = (2, 3, [0 1 7 0 0 0 0 0 0 0 0])
	(lambda[i], lambda[i] + 1, interval_count') = (2, 3, [0 1 8 0 0 0 0 0 0 0 0])
	(lambda[i], lambda[i] + 1, interval_count') = (1, 2, [0 2 8 0 0 0 0 0 0 0 0])
	"""
	
	# initial interval_ptr
	@views accumulate!(+, interval_ptr[2:end], interval_count[1:end-1])
	"""
	julia> interval_ptr'
	1×11 adjoint(::Vector{Int64}) with eltype Int64:
	0  0  2  10  10  10  10  10  10  10  10
	In this example, the value 2 at interval_ptr[3] means that 
	the interval with lambda[i] being 1 ends at index 2 in the interval_count array. 
	Similarly, the value 10 at interval_ptr[4] means that
	the interval with lambda[i] being 2 ends at index 10 in the interval_count array.
	"""

	# sort the nodes by the number of nodes strongly coupled to them:
	#   `index_to_node[idx]` is the node with `idx`th least number of nodes coupled to it
	#   `node_to_index[idx]` is the position of the `idx`th node in `index_to_node`
	# linear time and allocation-free equivalent to:
	#   sortperm!(index_to_node, lambda)
	#   node_to_index[index_to_node] .= 1:n
	interval_count .= 0 # temporarily zeroed, goes back to its original at end of loop
	"""
	julia> interval_count'
	1×11 adjoint(::Vector{Int64}) with eltype Int64:
	0  0  0  0  0  0  0  0  0  0  0
	"""
    for i = 1:n
        lambda_i = lambda[i] + 1
        interval_count[lambda_i] += 1
        index = interval_ptr[lambda_i] + interval_count[lambda_i]
        index_to_node[index] = i
        node_to_index[i]     = index
		#@show i,lambda_i, interval_count[lambda_i], index
		#@show index_to_node[index], node_to_index[i]
    end
	"""
	julia> lambda'
	1×10 adjoint(::Vector{Int64}) with eltype Int64:
	1  2  2  2  2  2  2  2  2  1
	julia> lambda_i
	1×10 adjoint(::Vector{Int64}) with eltype Int64:
	2  3  3  3  3  3  3  3  3  2
	julia> interval_ptr'
	1×11 adjoint(::Vector{Int64}) with eltype Int64:
	0  0  2  10  10  10  10  10  10  10  10
	julia> interval_count'
	1×11 adjoint(::Vector{Int64}) with eltype Int64:
	0  2  8  0  0  0  0  0  0  0  0 <--------- incremented in each step
	
	(i, interval_count') = (1, [0 1 0 0 0 0 0 0 0 0 0])
	(i, interval_count') = (2, [0 1 1 0 0 0 0 0 0 0 0])
	(i, interval_count') = (3, [0 1 2 0 0 0 0 0 0 0 0])
	(i, interval_count') = (4, [0 1 3 0 0 0 0 0 0 0 0])
	(i, interval_count') = (5, [0 1 4 0 0 0 0 0 0 0 0])
	(i, interval_count') = (6, [0 1 5 0 0 0 0 0 0 0 0])
	(i, interval_count') = (7, [0 1 6 0 0 0 0 0 0 0 0])
	(i, interval_count') = (8, [0 1 7 0 0 0 0 0 0 0 0])
	(i, interval_count') = (9, [0 1 8 0 0 0 0 0 0 0 0])
	(i, interval_count') = (10, [0 2 8 0 0 0 0 0 0 0 0])

	(i, lambda_i, interval_count[lambda_i], index) = (1, 2, 1, 1)
	(index_to_node[index], node_to_index[i]) = (1, 1)

	(i, lambda_i, interval_count[lambda_i], index) = (2, 3, 1, 3)
	(index_to_node[index], node_to_index[i]) = (2, 3)

	(i, lambda_i, interval_count[lambda_i], index) = (3, 3, 2, 4)
	(index_to_node[index], node_to_index[i]) = (3, 4)

	(i, lambda_i, interval_count[lambda_i], index) = (4, 3, 3, 5)
	(index_to_node[index], node_to_index[i]) = (4, 5)

	(i, lambda_i, interval_count[lambda_i], index) = (5, 3, 4, 6)
	(index_to_node[index], node_to_index[i]) = (5, 6)

	(i, lambda_i, interval_count[lambda_i], index) = (6, 3, 5, 7)
	(index_to_node[index], node_to_index[i]) = (6, 7)

	(i, lambda_i, interval_count[lambda_i], index) = (7, 3, 6, 8)
	(index_to_node[index], node_to_index[i]) = (7, 8)

	(i, lambda_i, interval_count[lambda_i], index) = (8, 3, 7, 9)
	(index_to_node[index], node_to_index[i]) = (8, 9)

	(i, lambda_i, interval_count[lambda_i], index) = (9, 3, 8, 10)
	(index_to_node[index], node_to_index[i]) = (9, 10)

	(i, lambda_i, interval_count[lambda_i], index) = (10, 2, 2, 2)
	(index_to_node[index], node_to_index[i]) = (10, 2)

	julia> index_to_node'
	1×10 adjoint(::Vector{Int64}) with eltype Int64:
	1  10  2  3  4  5  6  7  8  9

	julia> node_to_index'
	1×10 adjoint(::Vector{Int64}) with eltype Int64:
	1  3  4  5  6  7  8  9  10  2
	"""
	splitting = fill(U_NODE, n)

    # all nodes which no other nodes are strongly coupled to become F nodes
	# in our case it's doing nothing
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
	"""
	julia> index_to_node'
	1×10 adjoint(::Vector{Int64}) with eltype Int64:
	1  10  2  3  4  5  6  7  8  9

	julia> node_to_index'
	1×10 adjoint(::Vector{Int64}) with eltype Int64:
	1  3  4  5  6  7  8  9  10  2
	"""
	for top_index = n:-1:1
		i = index_to_node[top_index]
		lambda_i = lambda[i] + 1		#lambda_i = 2  3  3  3  3  3  3  3  3  2
		interval_count[lambda_i] -= 1

		splitting[i] == F_NODE && continue

		@assert splitting[i] == U_NODE
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
end
