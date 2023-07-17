using SparseArrays
using PartitionedArrays
using LinearAlgebra




function extract_IJV(A::PSparseMatrix)#, row_partition, col_partition)
    np = length(A.row_partition)
    ranks = LinearIndices((np,))
    
    Is = Vector{Vector{Int64}}()
    Js = Vector{Vector{Int64}}()
    Vs = Vector{Vector{Float64}}()

    map(ranks) do rank

        local_to_global_cols = local_to_global(A.col_partition[rank])
        local_to_global_rows = local_to_global(A.row_partition[rank])

        local_rows, local_cols, values = findnz(A.matrix_partition[rank])

        global_rows = local_to_global_rows[local_rows]
        global_cols = local_to_global_cols[local_cols]
        #@show global_rows, global_cols

        push!(Is, global_rows)
        push!(Js, global_cols)
        push!(Vs, values) 
    end
    return Is, Js, Vs
end

# Hints: use col_partition of the matrix.
# mat_col_partition == mat.col_partition
function transposed_IJV(Is, Js, Vs, mat_col_partition)
    # transpose(column <==== row)
    IJV = map(mat_col_partition)  do row_indices
        I, J, V  = Int[], Int[], Float64[]
        #skip ghost, collect own
        for global_row in own_to_global(row_indices)
            for rows in eachindex(Js)
                for  (j,i,v) in zip(Js[rows], Is[rows], Vs[rows])
                    if j == global_row
                        push!(I, j)
                        push!(J, i)
                        push!(V, v)
                    end
                end
            end
        end
        I,J,V
    end
    I,J,V = tuple_of_arrays(IJV)
    return I,J,V
end

# check symmetric: A == A'
function check_symmetric(A::PSparseMatrix)
    mA, nA = size(A)
    if mA != nA
        throw(DimensionMismatch("A has dimensions ($mA,$nA) which is not square matrix"))
    end
    At = transpose_psparse(A)
    ranks = LinearIndices((length(A.row_partition),))
    bools = []
    map(ranks) do rank
        I, J, V = findnz(A.matrix_partition[rank])
        I_t, J_t, V_t = findnz(At.matrix_partition[rank])
        # @show I, I_t
        # @show J, J_t
        # @show V, V_t
        boo = (isequal(I,I_t) && isequal(J,J_t) && isequal(V, V_t))
        push!(bools, boo)
    end
    
    return all(bools)
end

function check_symmetric2(A)
    At = transpose_psparse(A)
    map(ranks) do rank
        A_loc = A.matrix_partition[rank]
        At_loc = At.matrix_partition[rank]

        @show boo = A_loc == At_loc

    end
end


function transpose_psparse2(A::PSparseMatrix)

    ranks = LinearIndices((length(A.row_partition),))

    time = PTimer(ranks)
    tic!(time)

    row_len, col_len = size(A)
    # col_partition = uniform_partition(ranks, col_len)
    in_partition_cols = map(A.col_partition) do cols
        global_cols = own_to_global(cols)
        length(global_cols)
    end             
    col_partition = variable_partition(in_partition_cols, sum(in_partition_cols))
    #row_partition = uniform_partition(ranks, row_len)
    row_partition = A.row_partition
    # extract IJV
    I,J,V = extract_IJV(A) #, row_partition, col_partition)
    # transpose indices
    # I_t, J_t, V_t = transposed_IJV(I, J, V, A.col_partition)
    I_t, J_t, V_t = transposed_IJV(I, J, V, col_partition)

    toc!(time, "Transpose Comm.")
    display(time)
    # create transpose_psparse
    A_t = psparse!(I_t, J_t, V_t, col_partition, row_partition) |> fetch

    return A_t   
end


function ordered_local_SparseMatrixCSC(Sparse_mat::SparseMatrixCSC, rows, cols)
    # rearrange to the original partitioned matrix
    local_to_global_cols = local_to_global(cols)
    local_rows, local_cols, values = findnz(Sparse_mat)
    global_cols = local_to_global_cols[local_cols]

    return sparse(local_rows, global_cols, values)
end


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


function rows_to_pvector!(rhvec::PVector, row,row_partition)
    
    nonzero_indices = findnz(row)[1]

    IV = map(row_partition) do row_indices
        I,V = Int[], Float64[]
        I = intersect(row_indices, nonzero_indices)
        V = collect(row[I])
        I, V
    end
    I,V = tuple_of_arrays(IV) 

    rhvec .= pvector!(I,V, rhvec.index_partition) |> fetch
    consistent!(rhvec) |> wait
    rhvec
end


function mat_mat_mul(A::PSparseMatrix, B::PSparseMatrix)
    np = length(A.row_partition)    # processor
    ranks = LinearIndices((np,))    # of ranks

    mA, nA = size(A)
    mB, nB = size(B)
    
    if nA != mB
        throw(DimensionMismatch("A has dimensions ($mA,$nA) but B has dimensions ($mB,$nB)"))
    end
    
    row_partition_A = uniform_partition(ranks, mA)
    col_partition_A = uniform_partition(ranks, nA)
    
    row_partition_B = uniform_partition(ranks, mB)
    col_partition_B = uniform_partition(ranks, nB)
    
    # transpose B, to prepare each row for multiplication with A
    B_T =  transpose_psparse(B)   
    t = PTimer(ranks)
    tic!(t)

    IJV = map(ranks) do rank
        I,J,V = Int[], Int[], Float64[]

        col = 0;
        # check later row_partition of ???
        col = rank==1 ? 0 : sum(length(col_partition_B[i-1]) for i in 2:rank)

        #local
        local_mat = B_T.matrix_partition[rank]
        local_rows = B_T.row_partition[rank]
        local_cols = B_T.col_partition[rank]

        ordered_local_mat = ordered_local_SparseMatrixCSC(local_mat, local_rows, local_cols)
        # define rhvec and resultant pvec
        rhvec = PVector{Vector{Float64}}(undef,partition(axes(A,2)))
        pvec = PVector{Vector{Float64}}(undef,partition(axes(A,1)))

        for r in 1: ordered_local_mat.m

            # assign element based on row_partition_B and then converted into rhvec.index_partition
            rows_to_pvector!(rhvec, ordered_local_mat[r, :], row_partition_B)
            mul!(pvec, A, rhvec)

            j = r + col

            map(ranks) do part
                nz_indices = findall(!iszero, pvec.vector_partition[part])
                values = pvec.vector_partition[part][nz_indices]
                global_row_indices = local_to_global(pvec.index_partition[part])[nz_indices]
                
                for (i,v) in zip(global_row_indices, values)
                    push!(I, j)
                    push!(J, i)
                    push!(V, v)
                end
            end
        end
        I,J,V 
    end
    I,J,V = tuple_of_arrays(IJV)
    I,J,V = transposed_IJV(I,J,V, row_partition_A)
    result = psparse!(I,J,V, row_partition_A, col_partition_B) |> fetch

    toc!(t, "sleep")
    display(t)
    return result
end


"""
np = #of processor
n = #of rows/cols
"""
function ppoisson(n::Int64, np::Int64)
    
    ranks = LinearIndices((np,))

    row_partition = uniform_partition(ranks,n) 

    # set up the Poisson matrix A
    IJV = map(row_partition) do row_indices # index position (I,J) and value (V) in global 
        I,J,V = Int[], Int[], Float64[]
        for global_row in local_to_global(row_indices) # local to global mapper
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
    #@show IJV
    I,J,V = tuple_of_arrays(IJV)
    col_partition = row_partition

    pmat = psparse!(I,J,V, row_partition, col_partition) |> fetch

    return pmat
end


# function extract_offDiagMat(A::SparseMatrixCSC, cols, nA)
#     Ao = spzeros(size(A,1), nA)
#     global_cols = ghost_to_global(cols)
#     offDiag_cols = [1; diff(global_cols)]
#     local_cols = ghost_to_local(cols)
#     for (offDiag_col, local_col) in zip(offDiag_cols, local_cols)
#         Ao[:,offDiag_col] = A[:,local_col] 
#     end
#     Ao
# end

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

#accumulate pspase into 1 partition
function accumulate_psparse(A::PSparseMatrix; type::Type = SparseMatrixCSC)
    Is, Js, Vs = extract_IJV(A)
    mA, nA = size(A)
    if type == SparseMatrixCSC
        return sparse(vcat(Is...), vcat(Js...), vcat(Vs...), mA, nA)
    elseif type == Matrix
        return Matrix(sparse(vcat(Is...), vcat(Js...), vcat(Vs...), mA, nA))
    else
        throw(ArgumentError("Invalid type argument."))
    end
end
#accumulate pvector into 1 partition
function accumulate_pvector(b::PVector)
    out = map(ranks) do rank
        b_loc = own_values(b)[rank]
    end
    return vcat(out...)
end

function transpose_psparse(A::PSparseMatrix)

    ranks = LinearIndices((length(A.row_partition),))
    
    time = PTimer(ranks)
    tic!(time)

    #extract row_col partition
    row_partition = A.row_partition
    in_partition_cols = map(A.col_partition) do cols
        global_cols = own_to_global(cols)
        length(global_cols)
    end             
    col_partition = variable_partition(in_partition_cols, sum(in_partition_cols))

    # I and J are swapped
    I,J,V = extract_offDiag_transposed_global_IJV(A, ranks)

    #get destination for each I val
    dest = find_dest_from_I(I,A, ranks)

    # do Communication
    rcv_I, rcv_J, rcv_V = rcv_remote_IJV(I,J,V, dest,ranks)

    # setup new matrix
    IJV = map(ranks) do rank
        I,J,V = Int[], Int[], Float64[]
        row = minimum(own_to_global(A.row_partition[rank]))-1
        col = minimum(own_to_global(col_partition[rank]))-1
        

        # extract own IJV
        Js,Is,Vs = findnz(sparse(own_values(A)[rank]))
        Js.+=row
        Is.+=col

        for (i,j,v) in zip(Is,Js,Vs)
            push!(I, i);
            push!(J,j);
            push!(V,v);
        end

        # extract from rcv_buffer
        rcv_I_rank = rcv_I[rank]
        rcv_J_rank = rcv_J[rank]
        rcv_V_rank = rcv_V[rank]

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
    toc!(time, "Transpose Comm.")
    display(time)

    At = psparse!(I,J,V, col_partition, row_partition) |> fetch
end

function rcv_remote_IJV(I,J,V,dest,ranks)
    IJVD =map(ranks) do rank
        dest_rank = dest[rank]
        I_rank = I[rank]
        J_rank = J[rank]
        V_rank = V[rank]
        I_send_buffer = []
        J_send_buffer = []
        V_send_buffer = []
        destination = []
        unique_dest = unique(dest_rank)
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

    graph = ExchangeGraph(Ds)

    r = exchange(Is, graph);
    s = exchange(Js, graph);
    t = exchange(Vs, graph);


    rcv_I = fetch(r)
    rcv_J = fetch(s)
    rcv_V = fetch(t)

    return rcv_I, rcv_J, rcv_V
end

function find_dest_from_I(I,A, ranks)
    map(ranks) do rank
        I_loc = I[rank]
        dest =[]
        for I in I_loc
            for i in 1:length(A.col_partition)
                if I in own_to_global(A.col_partition[i])
                    push!(dest,i);
                end
            end
        end
        dest
    end  
end

function extract_offDiag_transposed_global_IJV(A::PSparseMatrix, ranks; transpose=true)
    IJV = map(ranks) do rank
        min_row = minimum(own_to_global(A.row_partition[rank]))-1
        I,J,V = Int[], Int[], Float64[]
        global_cols = ghost_to_global(A.col_partition[rank])
        Is,Js,Vs = findnz(sparse(own_ghost_values(A)[rank]))
        for (i,j,v) in zip(Is,Js,Vs)
            push!(I,i+min_row);
            push!(J,global_cols[j]);
            push!(V,v);
        end
        I,J,V
    end
    if transpose
        J,I,V = tuple_of_arrays(IJV);
    else
        I,J,V = tuple_of_arrays(IJV);
    end
    return I,J,V
end


function rhs(A::PSparseMatrix)

    row_partition = A.row_partition
    n = size(A,2)

    IV = map(row_partition) do row_indices
        I,V = Int[], Float64[]
        for global_row in local_to_global(row_indices)
            if global_row == 1
                v = 10.0
            elseif global_row == n
                v = -5.0
            else
                continue
            end
            push!(I,global_row)
            push!(V,v)
        end
        I,V
    end
    I,V = tuple_of_arrays(IV)
    b = pvector!(I,V,row_partition) |> fetch
end

function create_x(A::PSparseMatrix, b::PVector)
    x = similar(b,axes(A,2))
    x .= b
end