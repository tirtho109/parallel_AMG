using AlgebraicMultigrid
include("../smoother_psparse.jl")
import IterativeSolvers: gauss_seidel!
A = poisson(3)

x_sol = [0.0,1.0,2.0]

b = A*x_sol
x = zeros(3)

D = spdiagm(0 => zeros(size(A,1))) 
D[3,3] = 1.0
A .+= D

gauss_seidel!(x,A, b)
x