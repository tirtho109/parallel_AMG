using SparseArrays
using PartitionedArrays

"""
create Poisson matrix using PartitionedArrays
"""
function ppoisson(n::Int64)
    row_partition = uniform_partition(ranks,n) 
    in_partition_rows = map(row_partition) do rows
        global_rows = own_to_global(rows)
        length(global_rows)
    end             
    row_partition = variable_partition(in_partition_rows, sum(in_partition_rows))

    # set up the Poisson matrix A
    IJV = map(row_partition) do row_indices 
        I,J,V = Int[], Int[], Float64[]
        for global_row in local_to_global(row_indices) 
            if global_row in (1,n)
                push!(I,global_row)
                push!(J,global_row)
                push!(V,2.0)
    
                if global_row == 1
                    push!(I, global_row)
                    push!(J, global_row+1)
                    push!(V, -1.0)
                end
                if global_row == n 
                    push!(I, global_row)
                    push!(J, global_row-1)
                    push!(V, -1.0)
                end
            else
                push!(I,global_row)
                push!(J,global_row-1)
                push!(V,-1.0)

                push!(I,global_row)
                push!(J,global_row)
                push!(V,2.0)

                push!(I,global_row)
                push!(J,global_row+1)
                push!(V,-1.0) # use -1 usually
            end
        end
        I,J,V
    end
    I,J,V = tuple_of_arrays(IJV)
    col_partition = row_partition

    pmat = psparse!(I,J,V, row_partition, col_partition) |> fetch

    return pmat
end

"""
create transpose of PSparseMatrix
"""
function transpose_psparse(A::PSparseMatrix;boo::Bool=true)
    # time = PTimer(ranks)
    # tic!(time)

    # I and J are switched 
    I,J,V = extract_offDiag_transposed_global_IJV(A)
    #check emptiness
    empty = map(I) do i
        boo =  isempty(i)
    end
    #@show size(A)

    if !boo
        # find dest from given i
        dest = find_dest_from_I(I,A)
        # get remote Is,Js,Vs,Ds
        Is, Js, Vs, Ds = get_remote_IJVD(I,J,V,dest)
        #do exchange
        rcv_I, rcv_J, rcv_V = exchange_remote_IJV(Is,Js, Vs, Ds)
        #maek transpose
        At = make_transpose(A, rcv_I, rcv_J, rcv_V)
        At
    else
        row_partition = A.row_partition
        in_partition_cols = map(A.col_partition) do cols
            global_cols = own_to_global(cols)
            length(global_cols)
        end             
        col_partition = variable_partition(in_partition_cols, sum(in_partition_cols))

        IJV = map(A.row_partition, A.col_partition, own_values(A)) do rows, cols,own_val_A
            I,J,V = Int[], Int[], Float64[]
            row = minimum(own_to_global(rows))-1
            col = minimum(own_to_global(cols))-1

            #@show length(rows)
            # extract own IJV
            Js,Is,Vs = findnz(sparse(own_val_A))
            Js.+=row
            Is.+=col

            for (i,j,v) in zip(Is,Js,Vs)
                push!(I,i);
                push!(J,j);
                push!(V,v);
            end
            I,J,V
        end
        I,J,V = tuple_of_arrays(IJV)
        At = psparse!(I,J,V, col_partition, row_partition) |> fetch
        At
    end

    # toc!(time, "Transpose Comm.")
    # display(time)
    
    return At
end

"""
To extract only the offDiagonal indices in each partition
"""
function extract_offDiag_transposed_global_IJV(A::PSparseMatrix; transpose=true)

    IJV = map(A.row_partition, A.col_partition, own_ghost_values(A)) do rows, cols, own_ghost_vals
        min_row = minimum(own_to_global(rows))-1
        
        I,J,V = Int[], Int[], Float64[]
        global_cols = ghost_to_global(cols)
        Is,Js,Vs = findnz(sparse(own_ghost_vals))
        
        for (i,j,v) in zip(Is,Js,Vs)
            push!(I,i+min_row);
            push!(J,global_cols[j]);
            push!(V,v);
        end
        #@show min_row, ghost_to_global(cols), size(own_ghost_vals) 
        I,J,V
    end
    if transpose
        J,I,V = tuple_of_arrays(IJV);
    else
        I,J,V = tuple_of_arrays(IJV);
    end
    return I,J,V
end

"""
to find the node #, based on the given I.
"""
function find_dest_from_I(I,A)
    own_global = map(A.col_partition) do cols
        own_to_global(cols)
    end
    map(own_global) do og
        # @show og
    end
    all_own_global = gather(own_global, destination=:all)


    map(all_own_global,I) do all_og, I_loc
        dest = Vector{Int}([])
        for Is in I_loc
            for i in eachindex(all_og)
                if Is in all_og[i]
                    push!(dest,i);
                end
            end
        end
        # @show dest
        dest
    end
end

"""
get remote IJVD
"""
function get_remote_IJVD(I,J,V,dest)
    IJVD =map(I, J, V, dest) do I_rank, J_rank, V_rank, dest_rank

        I_send_buffer = Vector{Vector{Int}}([])
        J_send_buffer = Vector{Vector{Int}}([])
        V_send_buffer = Vector{Vector{Float64}}([])
        destination = Vector{Int}([])
        unique_dest = unique(dest_rank)
        # @show I_rank, J_rank, V_rank, dest_rank
        for ud in unique_dest
            Is, Js, Vs = Int[], Int[], Float64[]
                positions = findall(x -> x == ud, dest_rank)
                for i in positions
                    push!(Is, I_rank[i]);
                    push!(Js, J_rank[i]);
                    push!(Vs, V_rank[i]);
                end
            push!(I_send_buffer, Is)
            push!(J_send_buffer, Js)
            push!(V_send_buffer, Vs)
            push!(destination, ud)
        end
        I_send_buffer, J_send_buffer, V_send_buffer, destination
    end

    Is,Js,Vs,Ds = tuple_of_arrays(IJVD)
    return Is, Js, Vs, Ds 
end

"""
To do exchange remote rowval & colval
dest: 4-element Vector{Vector{Int64}}: 
data_send:4-element Vector{Vector{Vector{Int64}}}
"""
function exchange_remote_IJV(Is,Js,Vs,Ds)

    graph = ExchangeGraph(Ds)

    r = exchange(Is, graph);
    s = exchange(Js, graph);
    t = exchange(Vs, graph);


    rcv_I = fetch(r)
    rcv_J = fetch(s)
    rcv_V = fetch(t)

    return rcv_I, rcv_J, rcv_V
end

"""
do transpose using onw and remote IJV
"""
function make_transpose(A::PSparseMatrix, rcv_I, rcv_J, rcv_V)
    row_partition = A.row_partition
    in_partition_cols = map(A.col_partition) do cols
        global_cols = own_to_global(cols)
        length(global_cols)
    end             
    col_partition = variable_partition(in_partition_cols, sum(in_partition_cols))

    IJV = map(A.row_partition, A.col_partition, own_values(A),rcv_I, rcv_J, rcv_V) do rows, 
                                                                                    cols, 
                                                                                    own_val_A, 
                                                                                    rcv_I_rank,
                                                                                    rcv_J_rank,
                                                                                    rcv_V_rank

        I,J,V = Int[], Int[], Float64[]
        row = minimum(own_to_global(rows))-1
        col = minimum(own_to_global(cols))-1

        #@show length(rows)
        # extract own IJV
        Js,Is,Vs = findnz(sparse(own_val_A))
        Js.+=row
        Is.+=col

        for (i,j,v) in zip(Is,Js,Vs)
            push!(I,i);
            push!(J,j);
            push!(V,v);
        end

        for ele in eachindex(rcv_I_rank)
            for (i,j,v) in zip(rcv_I_rank[ele], rcv_J_rank[ele], rcv_V_rank[ele])
                push!(I, i);
                push!(J,j);
                push!(V,v);
            end
        end

        I,J,V
    end
    I,J,V = tuple_of_arrays(IJV);

    At = psparse!(I,J,V, col_partition, row_partition) |> fetch
end

"""
check_symmetric? If (A == At) [PSparseMatrix]
"""
function check_symmetric(A::PSparseMatrix)
    mA, nA = size(A)
    if mA != nA
        throw(DimensionMismatch("A has dimensions ($mA,$nA) which is not square matrix"))
    end

    At = transpose_psparse(A)
    map(local_values(A), local_values(At)) do A_loc, At_loc
        @show A_loc == At_loc
    end
end
"""
extract IJV from each partition of PSparseMatrix
"""
function extract_IJV(A::PSparseMatrix)

    IJV = map(A.col_partition, A.row_partition, local_values(A)) do cols, rows, A_loc

        I=Int[]; J=Int[]; V=Float64[]
        local_to_global_cols = local_to_global(cols)
        local_to_global_rows = local_to_global(rows)

        local_rows, local_cols, values = findnz(A_loc)

        global_rows = local_to_global_rows[local_rows]
        global_cols = local_to_global_cols[local_cols]
        # @show global_rows, global_cols
        for (i,j,v) in zip(global_rows, global_cols, values)
            push!(I, i)
            push!(J, j)
            push!(V, v)
        end 
        I,J,V
    end
    I,J,V = tuple_of_arrays(IJV)
    return I, J, V
end

"""
extract local_SparseMartixCSC from local_values(A::PSparseMatrix)
NOTE: Size is limited to the last global cols in that partition
"""
function ordered_local_SparseMatrixCSC(Sparse_mat::SparseMatrixCSC, rows, cols)
    # rearrange to the original partitioned matrix
    local_to_global_cols = local_to_global(cols)
    local_rows, local_cols, values = findnz(Sparse_mat)
    global_cols = local_to_global_cols[local_cols]

    return sparse(local_rows, global_cols, values)
end

"""
extract local_SparseMartixCSC from local_values(A::PSparseMatrix)
that has the original size in nA direction
"""
# Arguments = [A.matrix_partition[rank], A.row_partition[rank], A.col_partition[rank], nA]
function ordered_local_transposed_full_SparseMatrixCSC(Sparse_mat::SparseMatrixCSC, rows, cols, nA; transpose=true)
    # rearrange to the original partitioned matrix
    local_to_global_cols = local_to_global(cols)
    local_rows, local_cols, values = findnz(Sparse_mat)
    global_cols = local_to_global_cols[local_cols]

    if transpose
        return sparse(global_cols, local_rows, values, nA, maximum(local_rows))
    else
        return sparse(local_rows, global_cols, values, maximum(local_rows), nA)
    end
end

"""
extract offDiagonal matrix PETS style. 
Everything is counted except the diagonal block
Size remains the original PSparseMatrix size in nA
"""
function extract_offDiagMat(A::SparseMatrixCSC, cols, nA)
    Ao = spzeros(size(A,1), nA)
    global_cols = ghost_to_global(cols)
    #offDiag_cols = [1; diff(global_cols)]
    local_cols = ghost_to_local(cols)
    for (global_cols, local_col) in zip(global_cols, local_cols)
        Ao[:,global_cols] = A[:,local_col] 
    end
    Ao
end

"""
Accumulate PSparseMatrix as a Dense/SparseMatrixCSC format
##### Need to finalize later, in which partition we should send???
"""
#accumulate pspase into 1 partition
# function accumulate_psparse(A::PSparseMatrix; type::Type = SparseMatrixCSC)
#     Is, Js, Vs = extract_IJV(A)
#     mA, nA = size(A)
#     I_all = gather(Is,destination=:all)
#     J_all = gather(Js, destination=:all)
#     V_all = gather(Vs, destination=:all)
    
#     map(I_all, J_all, V_all, ranks) do i,j,v, rank
#         #if rank==1
#         if type==SparseMatrixCSC
#             sparse(vcat(i...), vcat(j...), vcat(v...), mA, nA)
#         elseif type==Matrix
#             Matrix(sparse(vcat(i...), vcat(j...), vcat(v...), mA, nA))
#         else
#              throw(ArgumentError("Invalid type argument."))
#         end
#         #end
#     end
# end
"""
accumulate pvector into one partition
### Need to clarify: In which partition we should send???
"""
#accumulate pvector into 1 partition
# function accumulate_pvector(b::PVector)
#     mA = size(b,1)
#     IV = map(own_values(b), b.index_partition) do own_b, ind
#         I=Int[]; V=Float32[];
#         own_to_global_ind = own_to_global(ind)
#         local_ind, local_val = findnz(sparse(own_b))
#         global_ind = own_to_global_ind[local_ind]
#         for (i,v) in zip(global_ind,local_val)
#             push!(I,i);
#             push!(V,v);
#         end
#         I,V
#     end
#     I,V = tuple_of_arrays(IV)
#     I_all = gather(I,destination=:all)
#     V_all = gather(V,destination=:all)
#     vecs=map(I_all, V_all) do i, v
#             Vector(sparsevec(vcat(i...), vcat(v...), mA))
#     end
# end

function rhs(A::PSparseMatrix)
    n = size(A,1)

    IV = map(A.row_partition) do row_indices
        I,V = Int[], Float64[]
        for global_row in local_to_global(row_indices)
            if global_row == 1
                v = 1.0
            elseif global_row == n
                v = 1.0
            else
                continue
            end
            push!(I,global_row)
            push!(V,v)
        end
        I,V
    end
    I,V = tuple_of_arrays(IV)
    b = pvector!(I,V,A.row_partition) |> fetch
end

function create_x(A::PSparseMatrix, b::PVector)
    x = similar(b,axes(A,2))
    x .= b
    #consistent!(x) |> wait
end

function rhs_parabola(A::PSparseMatrix; cols = false)
    if cols==true
        row_partition = A.col_partition
    else
        row_partition = A.row_partition
    end
    n = size(A,2)
    mid_index = div(n,2)+1

    IV = map(row_partition) do row_indices
        I,V = Int[], Float64[]
        for global_row in own_to_global(row_indices)
            push!(I, global_row)
            if global_row < mid_index  
                push!(V, global_row - 1)
            else
                push!(V, n - global_row)
            end
        end
        I,V
    end
    I,V = tuple_of_arrays(IV) 
    b = pvector!(I,V, row_partition) |> fetch
end