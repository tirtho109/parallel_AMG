using SparseArrays
using PartitionedArrays
using LinearAlgebra
include("utility.jl")

function mat_mul(A::PSparseMatrix, P::PSparseMatrix)
    ranks  = LinearIndices((length(A.row_partition),));

    mP, nP = size(P)
    mA, nA = size(A)

    @show mA, nA, mP, nP, length(ranks)

    if nA != mP
        throw(DimensionMismatch("A has dimensions ($mA,$nA) but P has dimensions ($mP,$nP)"))
    end

    #set Output row & col partition
    #row_partition = A.row_partition #w/o assigning value is better
    in_partition_cols = map(P.col_partition) do cols
                        global_cols = own_to_global(cols)
                        length(global_cols)
                    end             
    col_partition = variable_partition(in_partition_cols, sum(in_partition_cols))


    time = PTimer(ranks)
    tic!(time)

    #extract the remote rows here
    global_row, owner, dest = extract_numrow_owner_dest(A)
    if !isempty(global_row) && !isempty(owner) && !isempty(dest)
        rcv_I, rcv_J, rcv_V = extract_remote_rows(P, global_row, dest)
        global_Pr = create_local_Pr(rcv_I, rcv_J, rcv_V ,ranks)
    else
        global_Pr = []
        map(ranks) do rank
            push!(global_Pr, spzeros(0,0));
        end
    end

    toc!(time, "Communication")
    display(time)

    t = PTimer(ranks)
    tic!(t)

    IJV = map(ranks) do rank
        #A_local = A.matrix_partition[rank]
        I,J,V = Int[], Int[], Float64[]

        #set min row-col val
        row = minimum(own_to_global(A.row_partition[rank]))-1
        col = minimum(own_to_global(col_partition[rank]))-1

        #Left diag
        Ad = sparse(own_values(A)[rank])
        #Left off-diag
        Ao = sparse(own_ghost_values(A)[rank])
        col_indices = ghost_to_global(A.col_partition[rank])
        Ao .= Ao[:, sortperm(col_indices)]
        
        #right diag
        Pd = sparse(own_values(P)[rank])
        #right off-diag
        Po = extract_offDiagMat(P.matrix_partition[rank], P.col_partition[rank], nP)

        Pr = global_Pr[rank]
        #@show size(Ad), size(Po), size(Pd), size(Ao), size(Pr)
        #diagonals = local_to_global(A.row_partition[rank])
        #mA_loc = size(Ad,1)

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

    toc!(t, "Numerics")
    display(t)

    return result

end


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

function Rowwise_algorithm_for_numerical_calculation_of_Ap(Ad,Ao, Pd,Po,Pr, starting_row, starting_col)
    R = Dict{Int, Float64}()  # Initialize R as an empty hash table

    #whether is a function Arguments Cl
    Cl = spzeros(size(Ad, 1), size(P, 2))  # Initialize Cl as a sparse matrix

    for i in 1:size(Ad,1)
        R = Numeric_calculation_of_one_row_of_AP(i, Ad, Ao, Pd, Po, Pr,col)

        for (j,val) in R
            Cl[i,j] = val
        end
        R = Dict{Int, Float64}()  # Step 8: Clear R
    end
    return Cl
end

function Rowwise_algorithm_for_symbolic_calculation_of_AP(Ad, Ao, Pd, Po, Pr, nP)
    # Step 3: Initialize nzd = {0} and nzo = {0}
    nzd = [0]
    nzo = [0] 
    # Step 4: Initialize {Rd, Ro} = {∅, ∅}
    Rd = Dict{Int, Set{Int}}()
    Ro = Dict{Int, Set{Int}}()
    # Step 5: Initialize i = 1
    # Step 6: Iterate over rows of Cl
    for i in 1:size(Ad,1)
        # Step 7: Call Algorithm 1 to calculate {Rd, Ro}
        Rd, Ro = Symbolic_calculation_of_one_row_of_AP(i, Ad, Ao, Pd, Po, Pr)
        # Step 8: Update nzd and nzo with the sizes of Rd and Ro
        push!(nzd, length(Rd))
        push!(nzo, length(Ro))
        # Step 10: Clear {Rd, Ro}
        Rd = Dict{Int, Set{Int}}()
        Ro = Dict{Int, Set{Int}}()
    end
    # Step 13: Preallocate memory for Cl using nzd and nzo
    #Cl = spzeros(size(Ad,1), nP)
    # for i in 1:size(Ad, 1)
    #     nz = nzd[i] + nzo[i]
    #     Cl.colptr[i+1] = Cl.colptr[i] + nz
    #     Cl.nzval[Cl.colptr[i]+1:Cl.colptr[i+1]] .= 0.0
    # end
    return Cl
end

function Symbolic_calculation_of_one_row_of_AP(i,Ad, Ao, Pd, Po, Pr)
    # Step 2: Initialize sets Rd and Ro as empty sets
    Rd = Dict{Int, Set{Int}}()
    Ro = Dict{Int, Set{Int}}()
    # Step 3: Iterate over nonzero columns k in Ad(i, :)
    for k in findnz(Ad[i, :])[1]
        # Step 4: Iterate over nonzero columns j in Pd(k, :)
        for j in findnz(Pd[k, :])[1]
            # Step 5: Insert j into Rd
            push!(get!(Rd, k, Set{Int}()), j)
        end
        # Step 7: Iterate over nonzero columns j in Po(k, :)
        if !isempty(Po[k, :])
            for j in findnz(Po[k, :])[1]
                # Step 8: Insert j into Ro
                push!(get!(Ro, k, Set{Int}()), j)
            end
        end
    end
    # Step 11: Iterate over nonzero columns k in Ao(i, :)
    for k in findnz(Ao[i, :])[1]
        # Step 12: Iterate over nonzero columns j in Pr(k, :)
        for j in findnz(Pr[k, :])[1]
            # Step 13: Check if j is a diagonal column
            if i==j
                # Step 14: Insert j into Rd
                push!(get!(Rd, k, Set{Int}()), j)
            else
                # Step 16: Insert j into Ro
                push!(get!(Ro, k, Set{Int}()), j)
            end
        end
    end
    # Step 20: Output sets Rd and Ro
     return Rd, Ro
end



function extract_remote_rows(P, global_row, dest)
    ranks  = LinearIndices((length(P.row_partition),));

    set = map(ranks) do rank
        R = Dict{Int, Set{Int}}()
        global_rows = global_row[rank]
        current_dest = dest[rank]
        for i in eachindex(current_dest)
            if !haskey(R, current_dest[i])
                R[current_dest[i]] = Set{Int}()        
            end
            push!(R[current_dest[i]], global_rows[i])  
        end
        R
    end

    # data = map(ranks) do rank
    #     global_rows = global_row[rank]
    #     indices = []
    #     rows = []
    #     local_own = local_to_own(P.row_partition[rank])
    #     local_global = local_to_global(P.row_partition[rank])
    #     corresponding_local_own = local_own[findall(x -> x in global_rows, local_global)]
    #     #println(corresponding_local_own)
    #     local_mat = ordered_local_transposed_full_SparseMatrixCSC(P.matrix_partition[rank],P.row_partition[rank], P.col_partition[rank], size(P,2),transpose=false)
    #     for (i,j) in zip(corresponding_local_own, global_rows)
    #         push!(rows, local_mat[i,:])
    #         push!(indices, j)
    #     end
    #     rows
    # end

    #=
    set=Dict(2 => Set([7]))
        Dict(3 => Set([14]), 1 => Set([8]))
        Dict(4 => Set([22]), 2 => Set([15]))
        Dict(3 => Set([23]))
    set ---> keys = destination, value = row#
    =#

    IJVD = map(ranks) do rank
        s = set[rank]
        I_send_buffer = []
        J_send_buffer = []
        V_send_buffer = []
        destination = []
        local_own = local_to_own(P.row_partition[rank])
        local_global = local_to_global(P.row_partition[rank])
        #@show local_own, local_global
        local_mat = ordered_local_transposed_full_SparseMatrixCSC(P.matrix_partition[rank], P.row_partition[rank], P.col_partition[rank], size(P, 2), transpose = false)
        for key in keys(s) # rank=2, key= 3,1
            I, J, V = Int[], Int[], Float64[]
            rows = s[key]
            #@show rows
            row_count = 1
            corresponding_local_own = local_own[findall(x -> x in rows, local_global)]
            #@show corresponding_local_own
            for row in corresponding_local_own
                #@show row
                y, val = findnz(local_mat[row, :])
                for (m, n) in zip(y, val)
                    push!(I, row_count)
                    push!(J, m)
                    push!(V, n)
                end
               # @show row_count, y, val
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

    # graph = ExchangeGraph(dest)
    # t = exchange(data, graph);
    # rcv = fetch(t)
    # rcv   
    return rcv_I, rcv_J, rcv_V
end

function create_local_Pr(rcv_I, rcv_J, rcv_V ,ranks)
    # map(ranks) do rank
    #     local_rcv = rcv[rank]
    #     I = Int[]
    #     J = Int[]
    #     V = Float64[]
    #     #rcv_matrices = []
    #     #@show length(local_rcv)
    #     for i in eachindex(local_rcv)
    #         #@show i
    #         js, vs = findnz(local_rcv[i])
    #             for (j,v) in zip(js,vs)
    #                 push!(I,i);
    #                 push!(J,j);
    #                 push!(V,v);
    #         end
    #     end
    #     local_matrix = sparse(I,J,V)
    #     #push!(rcv_matrices, local_matrix);
    #     local_matrix
    # end
    map(ranks) do rank
        Is = rcv_I[rank]
        Js = rcv_J[rank]
        Vs = rcv_V[rank]

        I_concat = Int64[]
        J_concat = Int64[]
        V_concat = Float64[]

        max_value = 0
        for i in eachindex(Is)
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
        if isempty(I_concat) || isempty(J_concat) || isempty(V_concat)
            loc_mat = sparse(I_concat, J_concat, V_concat, max_value, 0) 
        else
            loc_mat = sparse(I_concat, J_concat, V_concat, max_value, maximum(J_concat))
        end
        loc_mat
    end
end


function extract_numrow_owner_dest(A::PSparseMatrix)
    ranks  = LinearIndices((length(A.row_partition),));
    request_Pr = map(ranks) do rank
        global_row_indices = Int[]
        owner = Int[]
        destination = Int[]        
        for(i,j) in zip(ghost_to_global(A.col_partition[rank]),ghost_to_owner(A.col_partition[rank]))
            push!(global_row_indices,i);
            push!(owner, j);
            push!(destination, rank);
        end
        (global_row_indices, owner, destination)
    end
    global_rows, owner, dest = tuple_of_arrays(request_Pr)
    global_rows = vcat(global_rows...)
    owner_copy = vcat(owner...)
    dest = vcat(dest...)
    indices = sortperm(owner_copy)
    global_rows_sorted = global_rows[indices]
    owner_sorted = owner_copy[indices]
    dest_sorted = dest[indices]
    # Group based on unique owners
    unique_owners = unique(owner_sorted)
    global_rows_grouped = [global_rows_sorted[findall(owner_sorted .== owner)] for owner in unique_owners]
    sorted_global_rows_grouped = [sort(rows) for rows in global_rows_grouped]
    owner_grouped = [owner_sorted[findall(owner_sorted .== owner)] for owner in unique_owners]
    dest_grouped = [dest_sorted[findall(owner_sorted .== owner)] for owner in unique_owners]
    sorted_global_rows_grouped, owner_grouped, dest_grouped#, owner
end




#=
julia> owner
1-element Vector{Vector{Int64}}:
 [3]

     map(ranks) do rank
            if owner[rank][1] != rank
                 insert!(owner, rank, Vector{Int64}())
            end
        end

julia> owner
3-element Vector{Vector{Int64}}:
 []
 []
 [3]
=#


