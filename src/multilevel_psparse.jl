struct Level{TA, TP, TR}
    A::TA
    P::TP
    R::TR
end

struct MultiLevel{S, Pre, Post, TA, TP, TR, Tw} # Cpre, Cpost,
    levels:: Vector{Level{TA, TP, TR}}
    final_A::TA
    coarse_solver::S
    presmoother::Pre
    postsmoother::Post
    #coarse_presmoother::Cpre
    #coarse_postsmoother::Cpost
    workspase::TW
end

#=
need to check MultiLevelWorkspace
=#
struct MultiLevelWorkspace{TX, bs}
    coarse_xs::PVector{Vector{Tx}}
    coarse_bs::PVector{Vector{Tx}}
    res_vecs::PVector{Vector{Tx}}
end

function MultiLevelWorkspace(::Type{Val{bs}}, ::Type{T}) where {bs, T<:Number}
    if bs === 1
        TX = PVector{Vector{T}}
    end
    MultiLevelWorkspace{TX, bs}(TX[], TX[], TX[])
end

Base.eltype(w::MultiLevelWorkspace{TX}) where TX = eltype(TX)
blocksize(w::MultiLevelWorkspace{TX, bs}) where {TX, bs} = bs

function residual!(m::MultiLevelWorkspace{TX, bs}, n) where {TX, bs}
    push!(m.res_vecs, TX(undef, n))
end

function coarse_x!(m::MultiLevelWorkspace{TX, bs}, n) where {TX, bs}
    push!(m.coarse_xs, TX(undef, n))
end

function coarse_b!(m::MultiLevelWorkspace{TX, bs}, n) where {TX, bs}
    push!(m.coarse_bs, TX(undef, n))
end

abstract type CoarseSolver end
struct Pinv{T} <: CoarseSolver
    pinvA::Matrix{T}
    Pinv{T}(A) where T = new{T}(pinv(Matrix(A))) #need to make dense in 1 partition
end
Pinv(A) = Pinv{eltype(A)}(A)
Base.show(io::IO, p::Pinv) = print(io, "Pinv")

function (p::Pinv)(x, b)
    mul!(x, p.pinvA, b)
end
Base.length(ml::MultiLevel) = length(ml.levels) + 1

function Base.show(io::IO, ml::MultiLevel)
    op = operator_complexity(ml) #Done::need to define
    g = grid_complexity(ml) #Done::need to define
    c = ml.coarse_solver
    total_nnz = nnz(ml.final_A) #final_A will be dense. No need to define
    if !isempty(ml.levels)
        total_nnz += sum(nonz(level.A) for level in ml.levels) # need to check
    end
    lstr = ""
    if !isempty(ml.levels)
    for (i, level) in enumerate(ml.levels)
        lstr = lstr *
            @sprintf "   %2d   %10d   %10d [%5.2f%%]\n" i size(level.A, 1) nnz(level.A) (100 * nonz(level.A) / total_nnz)
    end                                                                     #Done:: need to change the nnz
    end
    lstr = lstr *
        @sprintf "   %2d   %10d   %10d [%5.2f%%]" length(ml.levels) + 1 size(ml.final_A, 1) nnz(ml.final_A) (100 * nnz(ml.final_A) / total_nnz)
                                                                            # may not need to change, final_A == Dense
    opround = round(op, digits = 3)
    ground = round(g, digits = 3)

    str = """
    Multilevel Solver
    -----------------
    Operator Complexity: $opround
    Grid Complexity: $ground
    No. of Levels: $(length(ml))
    Coarse Solver: $c
    Level     Unknowns     NonZeros
    -----     --------     --------
    $lstr
    """
    print(io, str)
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
        (sum(nnz(level.A) for level in ml.levels) +
                nnz(ml.final_A)) / nonz(ml.levels[1].A) # Done:: need to change nnz(not-final)
    else
        1.
    end
end

function nonz(A::PSparseMatrix)
    nz = map(ranks) do rank
        loc_nnz = nnz(A.matrix_partition[rank])
        loc_nnz
    end
    summ = 0
    for i in eachindex(nz)
        summ += nz[i]
    end
    return summ
end

abstract type Cycle end
struct V <: Cycle
end

struct W <: Cycle
end

struct F <: Cycle
end

function _solve(ml::MultiLevel, b::AbstractVector, args...; kwargs...) # check type b
    n = length(ml) == 1 ? size(ml.final_A, 1) : size(ml.levels[1].A, 1)
    V = promote_type(eltype(ml.workspace), eltype(b))
    x = zeros(V, size(b))
    return _solve!(x, ml, b, args...; kwargs...)
end

function _solve!(x, ml::MultiLevel, b::AbstractVector{T}, # check type b
    cycle::Cycle = V();
    maxiter::Int = 100,
    abstol::Real = zero(real(eltype(b))),
    reltol::Real = sqrt(eps(real(eltype(b)))),
    verbose::Bool = false,
    log::Bool = false,
    calculate_residual = true, kwargs...) where {T}

    A = length(ml) == 1 ? ml.final_A : ml.levels[1].A
    V = promote_type(eltype(A), eltype(b))
    log && (residuals = PVector{Vector{V}}()) #check residual type

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
    # @show residuals
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
    # if lvl == length(ml.levels)
    #     A = accumulate_psparse(ml.levels[lvl].A)

    #     x = accumulate_pvector(x)
    #     b = accumulate_pvector(b)
    #     ml.coarse_presmoother(A,x,b)

    #     res = accumulate_pvector(ml.workspace.res_vecs[lvl])
    #     mul!(res, A, x)
    #     reshape(res, size(b)) .= b .- reshape(res, size(b))

    #     coarse_b = accumulate_pvector(ml.workspace.coarse_bs[lvl])
    #     mul!(coarse_b, accumulate_psparse(ml.levels[lvl].R), res)

    #     coarse_x = accumulate_pvector(ml.workspace.coarse_xs[lvl])
    #     coarse_x .= 0
    #     ml.coarse_solver(coarse_x, coarse_b)

    #     mul!(res, ml.levels[lvl].P, coarse_x)
    #     x .+= res

    A = ml.levels[lvl].A
    #need to add, if lvl at length(ml.levels), accumulate x,A,b into 1 partition.
    # if lvl == length(ml.levels)
    #     ml.coarse_presmoother(A,x,b) #need to check
    # else
    #     ml.presmoother(A, x, b)
    # end
    ml.presmoother(A, x, b)

    res = ml.workspace.res_vecs[lvl]
    mul!(res, A, x) #A.col_partition != x.index_partition
    reshape(res, size(b)) .= b .- reshape(res, size(b))

    coarse_b = ml.workspace.coarse_bs[lvl]
    mul!(coarse_b, ml.levels[lvl].R, res)  #A.col_partition != res.index_partition

    coarse_x = ml.workspace.coarse_xs[lvl]
    coarse_x .= 0
    if lvl == length(ml.levels)
        coarse_x_acc = accumulate_pvector(coarse_x)
        ml.coarse_solver(coarse_x_acc, accumulate_pvector(coarse_b))
        coarse_x = vec_to_pvec(coarse_x_acc, ml.levels[lvl].P)
    else
        coarse_x = __solve_next!(coarse_x, ml, cycle, coarse_b, lvl + 1)
    end
    
    mul!(res, ml.levels[lvl].P, coarse_x) #A.col_partition != coarse_x.index_partition
    x .+= res

    # if lvl == length(ml.levels)
    #     ml.coarse_postsmoother(A,x,b) #need to check
    # else
    #     ml.postsmoother(A, x, b)
    #  end #fix smoother based on the level
    ml.postsmoother(A, x, b)

    x
end

function vec_to_pvec(x::Vector, P::PSparseMatrix)

    IV = map(P.col_partition) do cols
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
