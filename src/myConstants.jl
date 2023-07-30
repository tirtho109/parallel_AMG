const F_NODE = 0
const C_NODE = 1
const U_NODE = 2

struct RS
end

abstract type Smoother end
abstract type Sweep end
struct SymmetricSweep <: Sweep
end
struct ForwardSweep <: Sweep
end
struct BackwardSweep <: Sweep
end
struct GaussSeidel{S} <: Smoother
    sweep::S
    iter::Int
end

GaussSeidel(; iter = 1) = GaussSeidel(SymmetricSweep(), iter)
GaussSeidel(f::ForwardSweep) = GaussSeidel(f, 1)
GaussSeidel(b::BackwardSweep) = GaussSeidel(b, 1)
GaussSeidel(s::SymmetricSweep) = GaussSeidel(s, 1)
abstract type Strength end

struct Classical{T} <: Strength
    θ::T
end

Classical(;θ = 0.25) = Classical(θ)
