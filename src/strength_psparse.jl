using PartitionedArrays
using SparseArrays
include("utility.jl")
#include("myConstants.jl")

# abstract type Strength end

# struct Classical{T} <: Strength
#     θ::T
# end
# Classical(;θ = 0.25) = Classical(θ)

"""
To return the Strength matrix and it's transpose
"""
function (c::Classical)(A::PSparseMatrix; Symmetric::Bool=true) where {Ti, Tv}
    θ = c.θ
    mA, nA = size(A)

    in_partition_cols = map(A.col_partition) do cols #changed
        global_cols = own_to_global(cols)
        length(global_cols)
    end  
    col_partition = variable_partition(in_partition_cols, sum(in_partition_cols))

    if !Symmetric
        A = transpose_psparse(A)
    end

    IJV = map(local_values(A), A.row_partition, A.col_partition) do A_loc, rows, cols
        I, J, V = Int[], Int[], Float64[]
        # considering transposed as A = At
        At = ordered_local_transposed_full_SparseMatrixCSC(A_loc, rows, cols, nA)
        m, n = size(At)
        T = deepcopy(At)

        #set col val
        col = minimum(own_to_global(cols))-1

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

    T = psparse!(I,J,V,A.row_partition, A.col_partition) |> fetch 
    S = transpose_psparse(T)
    return S, T
end

"""
fin the max off diagonal vals
"""
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

"""
Find max in a col
"""
function find_max(A, i)
    m = zero(eltype(A))
    for j in nzrange(A, i)
        row = A.rowval[j]
        val = A.nzval[j]
        m = max(m, val)
    end
    m
end

"""
scaling nzval by max
"""
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