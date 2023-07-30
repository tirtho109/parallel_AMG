"""
Do matXmat (A*P)
"""
function mat_mul(A::PSparseMatrix, P::PSparseMatrix)
    mP, nP = size(P)
    mA, nA = size(A)

    in_partition_cols = map(partition(axes(P,2))) do cols
        global_cols = own_to_global(cols)
        length(global_cols)
    end             
    col_partition = variable_partition(in_partition_cols, sum(in_partition_cols))

    global_rows, remote_owner, asked_by = extract_remote_rowsnum_onwers_dests(A)
    extract_row, next_dest = exchange_remote_rowval(global_rows,remote_owner,asked_by)
    rcv_I, rcv_J, rcv_V = extract_remote_rowsP(P,extract_row, next_dest)

    global_Pr = create_local_Pr(rcv_I, rcv_J, rcv_V)
    
    IJV = map(partition(axes(A,1)), 
        col_partition, 
        own_values(A),
        own_ghost_values(A),
        partition(axes(A,2)),
        own_values(P),
        local_values(P),
        partition(axes(P,2)),
        global_Pr) do rows, cols, own_A, 
                            og_A, cols_A, own_P,
                            loc_P, cols_P, Pr

        I,J,V = Int[], Int[], Float64[]

        #set min row-col val
        row = minimum(own_to_global(rows))-1
        col = minimum(own_to_global(cols))-1

        #Left diag & off-diag
        Ad = sparse(own_A)
        Ao = sparse(og_A)
        col_indices = ghost_to_global(cols_A)
        Ao .= Ao[:, sortperm(col_indices)]

        #right diag & off-diag
        Pd = sparse(own_P)
        # Po can be improved by only taking nz values with it's global coordinates.
        # Row-wise dict could be a good sol'n
        Po = extract_offDiagMat(loc_P, cols_P, nP)

        #Pr = global_Pr[rank]

        for i in 1:size(Ad, 1)
            current_i = row + i 
            R = Numeric_calculation_of_one_row_of_AP(i, Ad, Ao, Pd, Po, Pr, col)
            #@show R
            for (j,val) in R
                push!(I, current_i);
                push!(J, j);
                push!(V, val);
            end
        end
        I,J,V
    end
    I,J,V = tuple_of_arrays(IJV) 

    result = psparse!(I,J,V, A.row_partition, col_partition) |> fetch

    return result

end


"""
Extract required row number, it's original owner and the destination[asked by]
"""
function extract_remote_rowsnum_onwers_dests(A::PSparseMatrix)
    request_Pr = map(A.col_partition, ranks) do cols, rank
        global_row_indices = Int[]
        remote_owner = Int[]
        asked_by = Int[]        
        for(i,j) in zip(ghost_to_global(cols),ghost_to_owner(cols))
            push!(global_row_indices,i);
            push!(remote_owner, j);
            push!(asked_by, rank);
        end
        #push!(destination, rank);
        (global_row_indices, remote_owner, asked_by)
    end
    global_rows, remote_owner, asked_by = tuple_of_arrays(request_Pr) 
    return global_rows, remote_owner, asked_by
end

#step:1-------> Eo the exchange: Send row number to it's owner, along with the asked by partition
function exchange_remote_rowval(global_rows, remote_owner, asked_by)
    x = map(global_rows, remote_owner, asked_by) do grs, os, ab
        rows_group = Vector{Int}[]
        owner_group = Int[]
        asked_by_group = Vector{Int}[]
        for (row, owner, asked) in zip(grs, os, ab)
            # Check if the owner is already in the owner_group
            if owner in owner_group
                # Find the index of the owner in the owner_group
                owner_idx = findfirst(isequal(owner), owner_group)
                # Push the row to the corresponding rows_group
                push!(rows_group[owner_idx], row)
                push!(asked_by_group[owner_idx], asked)
            else
                # If the owner is not in owner_group, create a new entry
                push!(owner_group, owner)
                push!(rows_group, [row])
                push!(asked_by_group, [asked])
            end
        end
        rows_group, owner_group, asked_by_group
    end
    rows, own, askedby = tuple_of_arrays(x)

    #do rowval excahnge and it's next destination
    graph = ExchangeGraph(own);
    t = exchange(rows, graph);
    row_asked = fetch(t)
    s = exchange(askedby, graph);
    dest_asked = fetch(s)

    extract_row, next_dest =map(row_asked, dest_asked) do rows, dest
        vcat(rows...), vcat(dest...)       
    end |> tuple_of_arrays

    return extract_row, next_dest
end

# step 2------> extract remote rows from P and send(I,J,V) it back to it's "asked by" using exchange
function extract_remote_rowsP(P,extract_row, next_dest)

    set = map(extract_row, next_dest) do global_rows, current_dest
        R = Dict{Int, Set{Int}}()
        for i in eachindex(current_dest)
            if !haskey(R, current_dest[i])
                R[current_dest[i]] = Set{Int}()        
            end
            push!(R[current_dest[i]], global_rows[i])  
        end
        R
    end

    mP,nP = size(P)
    IJVD = map(partition(axes(P,1)), 
                partition(axes(P,2)), 
                local_values(P),
                set) do rs, cs, local_P, s

        #s = set[rank]
        I_send_buffer = Vector{Int}[]
        J_send_buffer = Vector{Int}[]
        V_send_buffer = Vector{Float64}[]
        destination = Int[]

        local_own = local_to_own(rs)
        local_global = local_to_global(rs)

        local_mat = ordered_local_transposed_full_SparseMatrixCSC(local_P, rs, cs, nP, transpose = false)
        for key in keys(s) # rank=2, key= 3,1
            I, J, V = Int[], Int[], Float64[]
            rows = s[key]
            row_count = 1
            corresponding_local_own = local_own[findall(x -> x in rows, local_global)]
            for row in corresponding_local_own
                y, val = findnz(local_mat[row, :])
                for (m, n) in zip(y, val)
                    push!(I, row_count)
                    push!(J, m)
                    push!(V, n)
                end
               
                row_count += 1
            end
            push!(I_send_buffer, I)
            push!(J_send_buffer, J)
            push!(V_send_buffer, V)
            push!(destination, key)
            #@show key, rows, corresponding_local_own
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

"""
Gather remote rows.
To create local Pr matrix for multiplying with Ao
"""
function create_local_Pr(rcv_I, rcv_J, rcv_V)

    map(rcv_I, rcv_J, rcv_V) do Is, Js, Vs
        I_concat = Int64[]
        J_concat = Int64[]
        V_concat = Float64[]

        max_value = 0
        for i in eachindex(Is)
            # increase max_value if the row is empty
            if isempty(Is[i]) || isempty(Js[i]) || isempty(Vs[i])
                max_value += 1
            else
                Is[i] .= Is[i] .+ max_value
                I_concat = vcat(I_concat, Is[i])
                J_concat = vcat(J_concat, Js[i])
                V_concat = vcat(V_concat, Vs[i])
                max_value = maximum(Is[i])
            end
        end
        # create Pr matrix.
        if isempty(I_concat) || isempty(J_concat) || isempty(V_concat)
            loc_mat = sparse(I_concat, J_concat, V_concat, max_value, 0) 
        else
            loc_mat = sparse(I_concat, J_concat, V_concat, max_value, maximum(J_concat))
        end
        loc_mat
    end
end

"""
Do the row-wise multiplication for each partition
"""
function Numeric_calculation_of_one_row_of_AP(i, Ad, Ao, Pd, Po, Pr, col)
    R = Dict{Int, Float64}() # Initialize r

    # Step 3: Iterate over nonzero columns k in Ad(i, :)
    for k in findnz(Ad[i, :])[1]
        # Step 4: Iterate over nonzero columns j in Pd(k, :)
        for j in findnz(Pd[k, :])[1]
            # Step 5: Add Ad(i, k) * Pd(k, j) to R[j]
            if haskey(R, j+col)
                R[j+col] += Ad[i, k] * Pd[k, j]
            else
                R[j+col] = Ad[i, k] * Pd[k, j]
            end
        end

        # Step 7: Iterate over nonzero columns j in Po(k, :)
        if !isempty(Po)
            for j in findnz(Po[k, :])[1]
                # Step 8: Add Ad(i, k) * Po(k, j) to R[j]
                if haskey(R, j)
                    R[j] += Ad[i, k] * Po[k, j] # need to varify with A*A
                    #@show j
                else
                    R[j] = Ad[i, k] * Po[k, j] # need to varify with A*A
                    #@show j
                end
            end
        end
    end
    # Step 11: Iterate over nonzero columns k in Ao(i, :)
    for k in findnz(Ao[i, :])[1]
        # Step 12: Iterate over nonzero columns j in P˜r(k, :)
        for j in findnz(Pr[k, :])[1]
            # we can add condition 
            # such that, if j < min(global_to_own(P)[rank])
            # Step 13: Add Ao(i, k) * P˜r(k, j) to R[j]
            if haskey(R, j)
                R[j] += Ao[i, k] * Pr[k, j]
            else
                R[j] = Ao[i, k] * Pr[k, j]
            end
        end
    end
    return R
end

############# Extras ############
function exchange_remote_rowval2(global_rows, owner)
    
    x = map(global_rows, owner) do grs, os
        grouped_rows_dict =  Dict{Int, Vector{Int}}()
        for (row, owners) in zip(grs,os)
            if haskey(grouped_rows_dict, owners)
                push!(grouped_rows_dict[owners], row)
            else
                grouped_rows_dict[owners] = [row]
            end
        end
        grouped_rows_dict
    end
    #get dest
    dest =map(x) do iks
        d = []
        #x =
        collect(keys(iks))
        # for ds in x
        #     push!(d, ds)
        # end
        # d
    end
    row_vals =map(x) do iks
        collect(values(iks))
    end
    graph = ExchangeGraph(dest)
    t = exchange(row_vals, graph);
    row_rcv = fetch(t)
    return row_rcv
end
#= 
####### Goal ######

julia> map(global_rows, owner, dest) do g,o,d
                  @show g, o, d
              end
(g, o, d) = ([8], [2], [2])
(g, o, d) = ([7, 16], [1, 3], [1, 3])
(g, o, d) = ([15, 24], [2, 4], [2, 4])
(g, o, d) = ([23], [3], [3])
4-element Vector{Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}}}:
 ([8], [2], [2])
 ([7, 16], [1, 3], [1, 3])
 ([15, 24], [2, 4], [2, 4])
 ([23], [3], [3])

 julia> x
4-element Vector{Vector{Int64}}:
 [7]
 [8, 15]
 [16, 23]
 [24]

julia> z
4-element Vector{Vector{Int64}}:
 [2]
 [1, 3]
 [2, 4]
 [3]

 Dict(2 => Set([7]))
 Dict(3 => Set([15]), 1 => Set([8]))
 Dict(4 => Set([23]), 2 => Set([16]))
 Dict(3 => Set([24]))

 =#
