using PartitionedArrays
using SparseArrays
include("utility.jl")

"""
    In PartitionedArrays distributed local matrices are SparseMatrixCSC
    So, we cal culate transpose of the S matrix first then 
    transpose back to Strength matrix(S).
    Communication needed to calculate maximum in shared cols.
    [send_ghost_max -> compare_with_own_max -> select_the_final_max -> send_back_max_to_the_ghost_col]
    Ref: Pyamg: abs(aᵢⱼ) > θ × max(abs(aᵢₖ)), where i ≠ k 
    https://pyamg.readthedocs.io/en/latest/generated/pyamg.strength.html
    Ref: RS: 7.1.1 Standard coarsening (aᵢₖ<0)) & 7.1.3 Strong positive connectio
                θ = 0.25                &           θ = 0.5[not fixed]
    https://www.scai.fraunhofer.de/content/dam/scai/de/documents/AllgemeineDokumentensammlung/SchnelleLoeser/SAMG/AMG_Introduction.pdf
"""

function (c::Classical)(A::PSparseMatrix; Symmetric::Bool=true) where {Ti, Tv}
    θ = c.θ
    mA, nA = size(A)

    in_partition_cols = map(A.col_partition) do cols #changed
        global_cols = own_to_global(cols)
        length(global_cols)
    end  
    col_partition = variable_partition(in_partition_cols, sum(in_partition_cols))

    T = deepcopy(A)
    Offdiag_max_in_shared_cols = get_set_of_max_in_shared_columns(T, "offdiag")
    # @show Offdiag_max_in_shared_cols

    # check the strengh of connection. Consider the shared cols max in all partition
    map(local_values(T), Offdiag_max_in_shared_cols) do local_T, OMSC
        n = size(local_T, 2)
        for i in 1:n
            if i ∈ keys(OMSC)
                _m = OMSC[i]
            else
                _m = find_max_off_diag(local_T, i)
            end
            threshold = θ * _m  # θ × max(abs(aᵢₖ))
            for j in nzrange(local_T, i)
                row = local_T.rowval[j]
                val = local_T.nzval[j]

                if row != i
                    if abs(val) >= threshold
                        local_T.nzval[j] = abs(val)
                    else
                        local_T.nzval[j] = 0
                    end
                end
            end
        end
        dropzeros!(local_T)
    end
    
    # Now scale the cols by the largest value in the cols
    # Consider the column max on the shared cols in all partition
    Max_in_shared_col = get_set_of_max_in_shared_columns(T, "all")
    # @show Max_in_shared_col
    map(local_values(T), Max_in_shared_col) do local_T, MSC
        # Go col-by-col
        n = size(local_T, 2)
        for i in 1:n
            if i ∈ keys(MSC)
                _m = MSC[i]
            else
                _m = find_max(local_T, i) #no type arg, as it's already +ve
            end
            for j in nzrange(local_T, i)
                local_T.nzval[j] /= _m
            end
        end
    end
    transpose_psparse(T), T # T' = S, strengh matrix
end


"""
    find_max_off_diag(A::SparseMatrixCSC, i::Int)
    fin the max off diagonal vals in cols of SparseMatrixCSC
    i = col
"""
function find_max_off_diag(A::SparseMatrixCSC, i::Int)
    m = zero(eltype(A))
    for j in nzrange(A, i)
        row = A.rowval[j]
        val = A.nzval[j]
        if row != i 
            m = max(m, abs(val))
        end
    end
    m
end

"""
    find_max(A::SparseMatrixCSC, i::Int; type="")
    Find max in a col of a sparse matrix
"""
function find_max(A::SparseMatrixCSC, i::Int; type::String = "")
    m = zero(eltype(A))
    for j in nzrange(A, i)
        row = A.rowval[j]
        val = A.nzval[j]
        if type == "abs"
            m = max(m, abs(val))
        else
            m = max(m, val)
        end
    end
    m
end


"""
    Single function that gives only dictionary
    Either contains max off-diag, or max-all(including diagonal element) 
    for respective shared ghost and own columns in all partition.
    Choose type::"all" or "offdiag"
    "all" gives max(col) & "offdiag" gives max(abs(col)) of offdiag
"""

############## Try for more than one ghost_cols and max to send ##########
function get_set_of_max_in_shared_columns(S::PSparseMatrix, type::String)

    if !(type in ["all", "offdiag"])
        throw(ArgumentError("Invalid 'type' argument. Please select 'all' or 'offdiag'."))
    end
    # Step 1: find max in ghost col
    out = map(local_values(S), S.col_partition, ranks) do local_S, cols, rank
        m = Float64[]
        owner = Int[]
        destinatin = Int[]
        for (i,j) in  zip(ghost_to_local(cols), ghost_to_owner(cols))
            if type == "all"
                _m = find_max(local_S, Int64(i))
            elseif type=="offdiag"
                _m =  find_max(local_S, Int64(i), type="abs") # abs(aᵢⱼ) > θ × max(abs(aᵢₖ)), where i ≠ k 
            end

            push!(m, _m)
            push!(owner, rank)
        end
        m, ghost_to_global(cols), ghost_to_owner(cols), owner
    end
    ghost_max, global_cols, destination, owner = tuple_of_arrays(out)

    # Step 2: Send max of ghost_col to the original owner.
    maxs, cols, dss, oss = map(ghost_max, global_cols, destination, owner) do gm, gc, dest, ow
        max_group = Vector{Float64}[]
        col_group = Vector{Int}[]
        ds_group = Int[]
        ow_group = Vector{Int}[]

        for (max, col, d, o) in zip(gm,gc,dest,ow)
            if d in ds_group
                d_inx = findfirst(isequal(d), ds_group)
                push!(max_group[d_inx], max)
                push!(col_group[d_inx], col)
                push!(ow_group[d_inx], o)
            else
                push!(max_group, [max])
                push!(col_group, [col])
                push!(ow_group, [o])
                push!(ds_group, d)
            end
        end
        max_group, col_group, ds_group, ow_group
    end |> tuple_of_arrays

    graph = ExchangeGraph(dss)
    t = exchange(maxs,graph);
    s = exchange(cols, graph);
    r = exchange(oss, graph)

    exchanged_max = fetch(t)
    exchanged_col = fetch(s)
    exchanged_owner = fetch(r)

    # Step 3: Compare the maximum from max_in_ghost_in_other_partiton
    # and max_in_owned_partition.
    # Make a dictionary that contains the max in the owned column
    set_of_shared_col, col_set, max_set, dest_set = map(S.col_partition,local_values(S),
                exchanged_max,exchanged_col,
                exchanged_owner,ranks) do colsP,local_S,
                                        e_max,e_col,
                                        e_owner,rank
        min_col = minimum(own_to_global(colsP))-1
        R = Dict{Int, Float64}()

        max_group = Vector{Float64}[]
        col_group = Vector{Int}[]
        ds_group = Int[]
        ow_group = Vector{Int}[]
        local_maxes = Float64[]

        # check for shared cols maximum 
        for k in eachindex(e_owner)
            in_partition_col_group = Int[]
            in_partition_max_group = Int[]
            for (c_max, col) in zip(e_max[k], e_col[k])
                m = 0.0
                if type == "all"
                    m = find_max(local_S, col - min_col)
                    R[col] = max(c_max, m)
                    #push!(local_maxes,  max(c_max, m))
                    push!(in_partition_col_group, col)
                    push!(in_partition_max_group, max(c_max, m))

                elseif type=="offdiag"
                    for j in nzrange(local_S, col - min_col)
                        row = local_S.rowval[j]
                        val = local_S.nzval[j]
                        if row != col - min_col
                            m = max(m, abs(val))
                        end
                    end
                    #@show m, rank
                    R[col] = max(c_max, m)
                    push!(in_partition_col_group, col)
                    push!(in_partition_max_group, max(c_max, m))
                    #push!(local_maxes,  max(c_max, m))
                end
            end
            push!(col_group, in_partition_col_group)
            push!(max_group, in_partition_max_group)
            push!(ds_group, e_owner[k][1])
        end
        R, col_group, max_group, ds_group
    end |> tuple_of_arrays

    # Step 4: Now send back the maximum to the ghost cols.
    graph = ExchangeGraph(dest_set)
    returned_global_cols = exchange(col_set, graph) |>fetch
    returned_global_max_by_col = exchange(max_set, graph) |> fetch

    # Step 5: Collect the maximum by column in the defined dictionary
    map(returned_global_cols, returned_global_max_by_col, set_of_shared_col) do gc, maxval, R
        for i in eachindex(gc)
            for (g,m) in zip(gc[i], maxval[i])
                R[g] = m
            end
        end
    end

    # Step 6: change keys to global to local col number
    local_set = map(S.col_partition, set_of_shared_col, ranks) do cols, s, rank        
        R = Dict{Int, Float64}()
        for keys in keys(s)
            indices = findfirst(x->x==keys, cols)      
            R[indices] = s[keys]
        end
        R
    end 
end
