using SparseArrays
using IterativeSolvers


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
        gs_psparse!(x, A, b, s.iter)
        #jacobi_psparse!(x,A,b,s.iter)
    end
    x
end

""""
Aplly l₁ GS. Let  M_l1GS = M_HGS + Dₗ₁
"""
function gs_psparse!(x::PVector, A::PSparseMatrix, b::PVector, maxiter::Int)

    M_l1gs = extract_l1_gs_smoother(A)

    for iter in 1:maxiter
        delx = pzeros(x.index_partition)    # initial guess
        # Relaxation Methods
        # uₖ₊₁ = uₖ + M⁻¹ ⋅ rₖ; where rₖ = f -Auₖ

        map(local_values(delx), 
            local_values(M_l1gs), 
            own_values(b-A*x),) do local_delx, 
                                    local_M, 
                                    own_res
            #Hints: own_res ← b & σ ← A⋅x
            n = size(local_M, 1)
            for i in 1:size(local_M, 1)
                σ = 0.0 # Initialize: σ
                for j in 1:size(local_M,2)
                    if j ≠ i
                        σ += local_M[i, j] * local_delx[j]
                    end
                end
                local_delx[i] = (own_res[i] - σ) / local_M[i, i]
            end
            local_delx
        end
        x .+= delx
        consistent!(x) |> wait
    end
    x
end

function extract_l1_gs_smoother(A::PSparseMatrix)

    IJV = map(partition(axes(A,1)), partition(axes(A,2)), own_values(A), own_ghost_values(A)) do rows, cols, own_A, ghost_A
        row = 0;
        row = minimum(own_to_global(rows))-1
        col = minimum(own_to_global(cols))-1
        I,J,V = Int[], Int[], Float64[]

        # extract empty ghost to I,J,V 
        gg_cols = ghost_to_global(cols)
        i,j,v = findnz(sparse(ghost_A))
        j = gg_cols[j]
        v .= 0.0
        append!(I, i.+row)
        append!(J, j)
        append!(V, v)


        B_k = sparse(own_A)
        D_l1 = calculate_D(ghost_A)
        B_k .= B_k .+ D_l1
        is, js, vs = findnz(B_k)

        is .+= row
        js .+= col

        append!(I,is)
        append!(J,js)
        append!(V,vs)
        I,J,V 
    end
    I,J,V = tuple_of_arrays(IJV)
    M_l1gs = psparse!(I,J,V, A.row_partition, A.row_partition) |> fetch
    return M_l1gs
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

function jacobi_psparse!(x::PVector, A::PSparseMatrix, b::PVector, maxiter::Int)
    n = size(A,1)
    for itr in 1:maxiter
        map(local_values(x), local_values(A), own_values(b)) do local_x, local_A, own_b
            mA, nA = size(local_A)
            res = zeros(size(own_b,1))
            for i in 1:mA
                for j in 1:nA
                    if i!= j
                        res[i]  -= local_A[i,j] * local_x[j]
                    end
                end
            end

            res .+= own_b

            for i in 1:mA
                local_x[i] = res[i] / local_A[i,i]
            end
        end
        consistent!(x) |> wait
    end
    x
end
