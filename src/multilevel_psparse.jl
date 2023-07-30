using PartitionedArrays
using Printf
using LinearAlgebra
include("utility.jl")

struct Level{TA, TP, TR}
    A::TA
    P::TP
    R::TR
end

struct MultiLevel{S, Pre, Post, TA,TP,TR, Tw}
    levels:: Vector{Level{TA,TP,TR}}
    final_A::TA
    coarse_solver::S
    presmoother::Pre
    postsmoother::Post
    workspace::Tw
end

struct MultiLevelWorkspace{Tx, bs}
    coarse_xs::Vector{Tx}
    coarse_bs::Vector{Tx}
    res_vecs::Vector{Tx}
end

function MultiLevelWorkspace(::Type{Val{bs}}, ::Type{T}) where {bs, T<:Number}
    TX = PVector{Vector{T}}
    MultiLevelWorkspace{TX, bs}(TX[], TX[], TX[])
end

Base.eltype(w::MultiLevelWorkspace{TX}) where TX = eltype(TX)
blocksize(w::MultiLevelWorkspace{TX, bs}) where {TX, bs} = bs

function residual!(m::MultiLevelWorkspace{TX, bs}, n) where {TX, bs}
    push!(m.res_vecs, pzeros(n)) #pzeros(index_partition)
end

function coarse_x!(m::MultiLevelWorkspace{TX, bs}, n) where {TX, bs}
    push!(m.coarse_xs, pzeros(n)) #pzeros(index_partition) = pzeros(n) <= pzeros(A.col_partition)
end

function coarse_b!(m::MultiLevelWorkspace{TX, bs}, n) where {TX, bs}
    push!(m.coarse_bs, pzeros(n)) #pzeros(index_partition)
end

abstract type CoarseSolver end
struct Pinv{T} <: CoarseSolver
    pinvA::Matrix{T}
    #Pinv{T}(A) where T = new{T}(pinv(Matrix(A))) #need to make dense in 1 partition
    Pinv{T}(A) where T = new{T}(pinv(accumulate_psparse(A,type=Matrix)))
end
Pinv(A) = Pinv{eltype(A)}(A)
Base.show(io::IO, p::Pinv) = print(io, "Pinv")

function (p::Pinv)(x, b)
    mul!(x, p.pinvA, b)
end

function Base.length(ml::MultiLevel)
    length(ml.levels) + 1
end

function Base.show(io::IO, ml::MultiLevel)
    op = operator_complexity(ml) 
    g = grid_complexity(ml) 
    c = ml.coarse_solver
    total_nnz = nonz(ml.final_A) 
    if !isempty(ml.levels)
        total_nnz += sum(nonz(level.A) for level in ml.levels) 
    end
    lstr = ""
    if !isempty(ml.levels)
        for (i, level) in enumerate(ml.levels)
            nonzLA = nonz(level.A)
            rows = size(level.A, 1)
            map(ranks, nonzLA, total_nnz) do rank, nonLA, tot_nnz
                if(rank==1)
                    lstr = lstr *
                        @sprintf "   %2d   %10d   %10d [%5.2f%%]\n" i rows nonLA (100 * nonLA / tot_nnz)
                end
            end
        end                                                                     #Done:: need to change the nnz
    end
    nnzfinalA = nonz(ml.final_A)
    final_rows = size(ml.final_A,1)
    map(ranks, nnzfinalA, total_nnz) do rank, nzfinalA, tot_nnz
        if(rank == 1)
            lstr = lstr *
                @sprintf "   %2d   %10d   %10d [%5.2f%%]" length(ml.levels) + 1 final_rows nzfinalA (100 * nzfinalA / tot_nnz)
                                                                                    # may not need to change, final_A == Dense
        end
    end
    opround = map(op) do o
            round(o, digits = 3)
        end
    ground = round(g, digits = 3)

    map(ranks, opround) do rank, ops
        if(rank == 1)
            str = """
            Multilevel Solver
            -----------------
            Operator Complexity: $ops
            Grid Complexity: $ground
            No. of Levels: $(length(ml))
            Coarse Solver: $c
            Level     Unknowns     NonZeros
            -----     --------     --------
            $lstr
            """
            print(io, str)
        end
    end
end


function grid_complexity(ml::MultiLevel)
    if !isempty(ml.levels)
        (sum(size(level.A, 1) for level in ml.levels) +
                size(ml.final_A, 1)) / size(ml.levels[1].A, 1)
    else
        1.
    end
end

function operator_complexity(ml::MultiLevel)
    if !isempty(ml.levels)
        nnz_final = nonz(ml.final_A)
        nnz_first = nonz(ml.levels[1].A)
        summ = map(ranks) do rank
                    0.0
                end  
        for level in ml.levels
            nnz_level = nonz(level.A)
            summ=map(summ, nnz_level) do s,nn
                s += nn
            end
        end
        summ = map(summ, nnz_final, nnz_first) do s, nnfi, nnfir
             s= (s+nnfi) / nnfir
        end
    else
        1.
    end
end

function nonz(A::PSparseMatrix)
    nz = map(local_values(A)) do A_loc
        loc_nnz = nnz(A_loc)
        loc_nnz
    end
    out = reduction(+, nz;init=0,destination=:all)
    return out
end

######################
####### Cycle ########
######################
abstract type Cycle end
struct V <: Cycle
end

struct W <: Cycle
end

struct F <: Cycle
end


function _solve(ml::MultiLevel, b::PVector, args...; kwargs...)
    n = length(ml) == 1 ? size(ml.final_A, 1) : size(ml.levels[1].A, 1)
    A = length(ml) == 1 ? ml.final_A : ml.levels[1].A
    v = promote_type(eltype(ml.workspace), eltype(b))
    x = pzeros(eltype(b), A.col_partition)
    consistent!(x) |> wait
    return _solve!(x, ml, b, args...; kwargs...)
end

function _solve!(x::PVector, ml::MultiLevel, b::PVector, # check type b
    cycle::Cycle = V();
    maxiter::Int = 1,
    abstol::Real = zero(real(eltype(b))),
    reltol::Real = sqrt(eps(real(eltype(b)))),
    verbose::Bool = false,
    log::Bool = false,
    calculate_residual = true, kwargs...) where {T}

    A = length(ml) == 1 ? ml.final_A : ml.levels[1].A

    v = promote_type(eltype(A), eltype(b))
    log && (residuals = Vector(v))

    normres = normb = norm(b)
    if normb != 0
        abstol = max(reltol * normb, abstol)
    end
    log && push!(residuals, normb)

    res = ml.workspace.res_vecs[1]
    itr = lvl = 1
    while itr <= maxiter && (!calculate_residual || normres > abstol)
        if length(ml) == 1
            ml.coarse_solver(x, b)
        else
            __solve!(x, ml, cycle, b, lvl)
        end
        if calculate_residual
            if verbose
                @printf "Norm of residual at iteration %6d is %.4e\n" itr normres
            end
            mul!(res, A, x)
            reshape(res, size(b)) .= b .- reshape(res, size(b))
            normres = norm(res)
            log && push!(residuals, normres)
        end
        itr += 1
    end
    log ? (x, residuals) : x
end


function __solve_next!(x, ml, cycle::V, b, lvl)
    __solve!(x, ml, cycle, b, lvl)
end

function __solve_next!(x, ml, cycle::W, b, lvl)
    __solve!(x, ml, cycle, b, lvl)
    __solve!(x, ml, cycle, b, lvl)
end

function __solve_next!(x, ml, cycle::F, b, lvl)
    __solve!(x, ml, cycle, b, lvl)
    __solve!(x, ml, V(), b, lvl)
end


function __solve!(x, ml, cycle::Cycle, b, lvl)
    
    A = ml.levels[lvl].A
    ml.presmoother(A, x, b)

    # cureent level 
    res = ml.workspace.res_vecs[lvl]
    mul!(res, A, x) #initialize: res.vector_partition = A.row_partition, but! x.index_partition == A.col_partition
    #reshape(res, size(b)) .= b .- reshape(res, size(b))
    res .= b .- res     # b.index_partition == res.index_partition

    # need to check whether we need to change res.index_partition to R.col_partition
    coarse_b = ml.workspace.coarse_bs[lvl]
    mul!(coarse_b, ml.levels[lvl].R, res)  # res.index_partition == A.row_partition == R.col_partition
                                           # b.index_partition == R.row_partition

    coarse_x = ml.workspace.coarse_xs[lvl] # coarse_x = P.col_partition
    coarse_x .= 0
    if lvl == length(ml.levels)
        coarse_x_acc = accumulate_pvector(coarse_x)
        ml.coarse_solver(coarse_x_acc, accumulate_pvector(coarse_b))
        coarse_x = vec_to_pvec(coarse_x_acc, ml.levels[lvl].P)
    else
        coarse_x = __solve_next!(coarse_x, ml, cycle, coarse_b, lvl + 1) #__solve!(x, ml, cycle::Cycle, b, lvl)
    end
    #@show lvl
    coarse_x_new = PVector(coarse_x.vector_partition, ml.levels[lvl].P.col_partition)
    mul!(res, ml.levels[lvl].P, coarse_x_new) # P.col_partition == coarse_x.index_partition
    #mul!(ml.workspace.res_vecs[lvl], ml.levels[lvl].P, coarse_x)
    x .+= res

    ml.postsmoother(A, x, b)

    x
end


function vec_to_pvec(x::Vector, P::PSparseMatrix)

    IV = map(partition(axes(P,2))) do cols
        I,V = Int[], Float64[]
        for global_col in own_to_global(cols)  
            push!(I, global_col)
            push!(V, x[global_col])
        end
        I,V
    end
    I,V = tuple_of_arrays(IV)
    out = pvector!(I,V, P.col_partition) |> fetch
    consistent!(out);
    return out
end


### CommonSolve.jl spec
struct AMGSolver{T}
    ml::MultiLevel
    b::Vector{T}
end

abstract type AMGAlg end

struct RugeStubenAMG  <: AMGAlg end
#struct SmoothedAggregationAMG  <: AMGAlg end

function solve(A::AbstractMatrix, b::Vector, s::AMGAlg, args...; kwargs...)
    solt = init(s, A, b, args...; kwargs...)
    solve!(solt, args...; kwargs...)
end
function init(::RugeStubenAMG, A, b, args...; kwargs...)
    AMGSolver(ruge_stuben(A; kwargs...), b)
end
function solve!(solt::AMGSolver, args...; kwargs...) 
    _solve(solt.ml, solt.b, args...; kwargs...)   
end


##################### Helpers ##########################

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
    out = map(own_values(b)) do own
        # b_loc = own_values(b)[rank]
        own
    end
    return vcat(out...)
end