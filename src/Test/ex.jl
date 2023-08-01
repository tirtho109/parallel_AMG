include("../utility.jl")
include("../myConstants.jl")
include("../classical_psparse.jl")
include("../matmatmul_helper.jl")
include("../multilevel_psparse.jl")

np = 2
n = 6
#const ranks = distribute_with_mpi(LinearIndices((np,)))
const ranks = LinearIndices((np,))

A = ppoisson(n)
x_solution = rhs_parabola(A; cols=true)
b = A*x_solution
map(local_values(b)) do b_local
    @show b_local
end

#ml = ruge_stuben(A; max_coarse=2, max_levels=1)
g = GaussSeidel()
x = pzeros(A.col_partition)

for i in 1:100
    # map(local_values(x)) do x_local
    #     @show x_local
    # end
    g(A, x, b)
    # map(own_values(x), own_values(b)) do x_own, b_own
    #     x_own #.= x_local .- b_local
    #     @show 
    # end
    res = A*x - b
    @show norm(res)
    p = map(own_values(A), own_values(x), own_values(b)) do a, xs, bs
        @show a, xs, bs
    end

end
