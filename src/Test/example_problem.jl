include("../utility.jl")
include("../myConstants.jl")
include("../strength_psparse.jl")
include("../splitting_psparse.jl")
include("../smoother_psparse.jl")
include("../matmatmul_helper.jl")
include("../classical_psparse.jl")
include("../multilevel_psparse.jl")

using SparseArrays
using PartitionedArrays
#=
using MPI 
mpiexec(cmd->run(`$cmd -np 4 julia --project=. src/Test/example_problem.jl`))
=#

np = 3
n = 100
#const ranks = distribute_with_mpi(LinearIndices((np,)))
const ranks = LinearIndices((np,))

A = ppoisson(n)
x_sol = rhs_parabola(A;cols=true)

b = A*x_sol


ml = ruge_stuben(A;max_coarse=2, max_levels=2)
"""
Multilevel Solver
-----------------
Operator Complexity: 1.625
Grid Complexity: 1.667
No. of Levels: 2
Coarse Solver: Pinv
Level     Unknowns     NonZeros
-----     --------     --------
    1            6           16 [61.54%]
    2            4           10 [38.46%]
"""

# _solve
n = length(ml) == 1 ? size(ml.final_A, 1) : size(ml.levels[1].A, 1)
A = length(ml) == 1 ? ml.final_A : ml.levels[1].A
v = promote_type(eltype(ml.workspace), eltype(b))
#x = pzeros(eltype(b), A.col_partition)
x = x_sol
consistent!(x) |> wait

# _solve!
maxiter = 10
abstol = zero(real(eltype(b)))
reltol = sqrt(eps(real(eltype(b))))
log = false

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

while itr <= maxiter
    # _solve!(cycle)
    A = ml.levels[lvl].A
    #try without smoother 
    #@show x.vector_partition
    ml.presmoother(A, x, b)
    #@show x.vector_partition
    res = ml.workspace.res_vecs[lvl]
    mul!(res, A, x)
    res .= b .- res 
    coarse_b = ml.workspace.coarse_bs[lvl]
    mul!(coarse_b, ml.levels[lvl].R, res) 
    coarse_x = ml.workspace.coarse_xs[lvl]
    coarse_x .= 0

    # as level == 1, we directly solve in coarse level.
    coarse_x_acc = accumulate_pvector(coarse_x)
    ml.coarse_solver(coarse_x_acc, accumulate_pvector(coarse_b))
    coarse_x = vec_to_pvec(coarse_x_acc, ml.levels[lvl].P)

    # interpolate solution
    coarse_x_new = PVector(coarse_x.vector_partition, ml.levels[lvl].P.col_partition)
    mul!(res, ml.levels[lvl].P, coarse_x_new) 
    x .+= res
    consistent!(x) |> wait
    @show x.vector_partition
    ml.postsmoother(A, x, b)
    @show x.vector_partition
    global itr += 1
end


# Sol'n
# julia> A
# 6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 16 stored entries:
#   2.0  -1.0    ⋅     ⋅     ⋅     ⋅
#  -1.0   2.0  -1.0    ⋅     ⋅     ⋅
#    ⋅   -1.0   2.0  -1.0    ⋅     ⋅
#    ⋅     ⋅   -1.0   2.0  -1.0    ⋅
#    ⋅     ⋅     ⋅   -1.0   2.0  -1.0
#    ⋅     ⋅     ⋅     ⋅   -1.0   2.0

# julia> b
# 6-element Vector{Float64}:
#  0.0
#  1.0
#  2.0
#  2.0
#  1.0
#  0.0

# julia> A\b
# 6-element Vector{Float64}:
#  2.999999999999999
#  5.999999999999999
#  7.999999999999998
#  8.0
#  6.000000000000002
#  3.0