using SparseArrays
using IterativeSolvers
import IterativeSolvers: gauss_seidel!

# abstract type Smoother end
# abstract type Sweep end
# struct SymmetricSweep <: Sweep
# end
# struct ForwardSweep <: Sweep
# end
# struct BackwardSweep <: Sweep
# end
# struct GaussSeidel{S} <: Smoother
#     sweep::S
#     iter::Int
# end

# GaussSeidel(; iter = 1) = GaussSeidel(SymmetricSweep(), iter)
# GaussSeidel(f::ForwardSweep) = GaussSeidel(f, 1)
# GaussSeidel(b::BackwardSweep) = GaussSeidel(b, 1)
# GaussSeidel(s::SymmetricSweep) = GaussSeidel(s, 1)

function (s::GaussSeidel{S})(A::PSparseMatrix, x::PVector, b::PVector) where {S<:Sweep}
   
    if S === ForwardSweep || S === SymmetricSweep || S === BackwardSweep
        #@show s.iter
        gs!(x, A, b, s.iter)
    end
    x
end


function gs!(x::PVector, A::PSparseMatrix, b::PVector, maxiter::Int) #, start, step, stop)
    n = size(A,1)
    z = zero(eltype(A))
    # ranks = LinearIndices((length(A.row_partition),))

    # preconditioner
    M_l1gs = extract_l1_gs_preconditioner(A)

    map(own_values(x), own_values(M_l1gs), own_values(b)) do own_x, own_M_l1gs, own_b
        gauss_seidel!(own_x, own_M_l1gs, own_b,maxiter=maxiter)   
    end
    consistent!(x) |> wait
    x
end

function extract_l1_gs_preconditioner(A::PSparseMatrix)
    # ranks = LinearIndices((length(A.row_partition),))

    IJV = map(A.row_partition, A.col_partition, own_values(A), local_values(A)) do rows, cols, own_A, local_A
        row = 0;
        row = minimum(own_to_global(rows))-1
        col = minimum(own_to_global(cols))-1
        I,J,V = Int[], Int[], Float64[]
        B_k = sparse(own_A)
        D_l1 = calculate_D(local_A)
        B_k .= B_k .+ D_l1
        is, js, vs = findnz(B_k)

        is .+= row
        js .+= col

        # for (i,j,v) in zip(is, js, vs)
        #     push!(I, i)
        #     push!(J, j)
        #     push!(V, v)
        # end

        append!(I,is)
        append!(J,js)
        append!(V,vs)
        I,J,V 
    end
    I,J,V = tuple_of_arrays(IJV)
    M_l1gs = psparse!(I,J,V, A.row_partition, A.row_partition) |> fetch
    M_l1gs
end


function calculate_D(A)
    n = size(A, 1)              # Get the dimension 
    D = spdiagm(0 => zeros(n))  # Initialize a sparse diagonal matrix

    for i in 1:n
        row_sum = sum(abs.(A[i, :])) 
        D[i, i] = row_sum  
    end
    return D
end
