using SparseArrays
using IterativeSolvers
import IterativeSolvers: gauss_seidel!, jacobi!,jacobi

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
    #M_l1gs = extract_l1_gs_preconditioner(A)

    map(local_values(x), local_values(A), own_values(b)) do local_x, local_A, own_b
        #gauss_seidel!(own_x, own_M_l1gs, own_b,maxiter=maxiter) 
        #own_x .=  0 
        #own_x .= jacobi(local_A, own_b; maxiter=10)
        res = zeros(size(own_b,1))
        for i in 1:size(local_A, 1)
            for j in 1:size(local_A,2)
                if i!= j
                    res[i]  -= local_A[i,j] * local_x[j]
                end
            end
        end

        res .+= own_b

        for i in 1:size(local_A,1)
            local_x[i] = res[i] / local_A[i,i]
        end
    end
    consistent!(x) |> wait
    x
end

function extract_l1_gs_preconditioner(A::PSparseMatrix)
    # ranks = LinearIndices((length(A.row_partition),))

    IJV = map(A.row_partition, A.col_partition, local_values(A), own_ghost_values(A)) do rows, cols, own_A, ghost_A
        row = 0;
        row = minimum(own_to_global(rows))-1
        col = minimum(own_to_global(cols))-1
        I,J,V = Int[], Int[], Float64[]
        #B_k = sparse(own_A)
        B_k = own_A
        #B_k = inv(B_k)
        #D_l1 = calculate_D(ghost_A)
        #B_k .= B_k .+ D_l1
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
