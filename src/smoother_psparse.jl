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
    # preconditioner
    #M_l1gs = extract_l1_gs_preconditioner(A)
    M_l1gs = A
    map(local_values(x), local_values(M_l1gs), own_values(b), own_values(x), ranks) do local_x, 
                                                                                local_A, 
                                                                                own_b, 
                                                                                own_x, rank
        #iks = deepcopy(local_x) #need own x
        n = size(local_A, 1)
        for iter in 1:maxiter
            x_new = similar(local_x)  
            for i in 1:n
                    # σ = 0.0 
                    # for j in 1:n
                    #     if j ≠ i
                    #         σ += local_A[i, j] * local_x[j]
                    #     end
                    # end
                    # x_temp[i] = (own_b[i] - σ) / local_A[i, i]
                s1 = dot(local_A[i,1:i-1], x_new[1:i-1])
                s2 = dot(local_A[i, i+1:end], local_x[i+1:end])
                x_new[i] = (own_b[i] - s1 - s2) / local_A[i, i]
            end
            #@show iks, rank
            for i in 1:n
                local_x[i] = x_new[i]
            end
        end
        local_x
    end
    consistent!(x) |> wait
    x
end

function extract_l1_gs_preconditioner(A::PSparseMatrix)

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
        itr +=1 
    end
    x
end