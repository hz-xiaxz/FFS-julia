module FastFermionSampling

# Write your package code here.
using Random
using LinearAlgebra

export FFS

function FFS(u::Matrix{ComplexF64}, L::Int, N::Int)
    # U: Matrix{ComplexF64}: the sampling ensemble
    # L: Int: the number of energy states
    # N: Int: the number of fermions
    v = randperm(N)
    U = u[:,v]
    # For x1 case, P(x1;m) = |U_{x1, m1}|^2 
end

end
