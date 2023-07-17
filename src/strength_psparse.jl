using PartitionedArrays
using SparseArrays
include("utility.jl")

abstract type Strength end

struct Classical{T} <: Strength
    θ::T
end
Classical(;θ = 0.25) = Classical(θ)

function (c::Classical)(A::PSparseMatrix; Symmetric::Bool=true) where {Ti, Tv}
    θ = c.θ

    np = length(A.row_partition)
    ranks = LinearIndices((np,)) 
    mA, nA = size(A)
    #row_partition_A = uniform_partition(ranks, mA)
    #col_partition_A = uniform_partition(ranks, nA)

    in_partition_cols = map(A.col_partition) do cols #changed
        global_cols = own_to_global(cols)
        length(global_cols)
    end             
    col_partition = variable_partition(in_partition_cols, sum(in_partition_cols))

    # if A is Symmetric, we don't need to transpose it, A = A'.
    # As we have to find out maximum based on each column
    if !Symmetric
        A = transpose_psparse(A)
    end
    
    IJV = map(ranks) do rank
        I, J, V = Int[], Int[], Float64[]
        #At = sparse(A.matrix_partition[rank]')
        At = ordered_local_transposed_full_SparseMatrixCSC(A.matrix_partition[rank], A.row_partition[rank], A.col_partition[rank], nA)
        m, n = size(At)
        T = copy(At)
        
        #col = 0;
        #col = rank==1 ? 0 : sum(length(col_partition_A[i-1]) for i in 2:rank)
        col = minimum(own_to_global(col_partition[rank]))-1
        
        # i == columns
        for i = 1:n
            _m = find_max_off_diag(T,i,col)
            threshold = θ * _m
            for j in nzrange(T, i)
                row = T.rowval[j]
                val = T.nzval[j]

                if row != (i+col)
                    if abs(val) >= threshold
                        T.nzval[j] = abs(val)
                    else
                        T.nzval[j] = 0
                    end
                end

            end
        end
        dropzeros!(T)
        scale_cols_by_largest_entry!(T)
        x,y,z = findnz(T)
        y = y .+ col
        for (i, j, v) in zip(x, y, z)  
            # switching order to transpose nz_indices
            # If Symmetric, it's alright, no further transpose                         
            push!(I, j)                                         
            push!(J, i)                                         
            push!(V, v)                                         
        end
        I,J,V
    end

    I,J,V = tuple_of_arrays(IJV)
    if !Symmetric
        I,J,V = transposed_IJV(I,J,V, col_partition_A)
    end
    T = psparse!(I,J,V,A.row_partition, col_partition) |> fetch 
    S = transpose_psparse(T);
    return S, T
    # return transpose_psparse(Strength_mat), Strength_mat
end

# of a column from (At) part A
function find_max_off_diag(A, i, col)
    m = zero(eltype(A))
    for j in nzrange(A, i)
        row = A.rowval[j]
        val = A.nzval[j]
        if row != i+col  #col based on the rank
            m = max(m, abs(val))
        end
    end
    m
end

function find_max(A, i)
    m = zero(eltype(A))
    for j in nzrange(A, i)
        row = A.rowval[j]
        val = A.nzval[j]
        m = max(m, val)
    end
    m
end

function scale_cols_by_largest_entry!(A::SparseMatrixCSC)
    n = size(A, 2) # for partitioned matrix case, size(A,2)
    for i = 1:n
        _m = find_max(A, i)
        for j in nzrange(A, i)
            A.nzval[j] /= _m
        end
    end
    A
end
