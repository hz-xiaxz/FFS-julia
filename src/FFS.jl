using GenericLinearAlgebra

function mynullspace(A::Matrix{Float64})
    m, n = size(A, 1), size(A, 2)
    (m == 0 || n == 0) && return Matrix{eigtype(eltype(A))}(I, n, n)

    SVD = GenericLinearAlgebra.svd(big.(A), full = true)
    atol = 0
    rtol = eps(eltype(A)) * max(m, n) * SVD.S[1]
    tol = max(atol, SVD.S[1] * rtol)
    indstart = sum(s -> s .> tol, SVD.S) + 1
    return copy((@view SVD.Vt[indstart:end, :])')
end


"""
    FFS([rng=default_rng()], ensemble_matrix::AbstractMatrix)

Employs the Fast Fermion Sampling Algorithm to sample free fermion states.

# Arguments
- `rng`: Random number generator (optional)
- `ensemble_matrix`: Matrix of size L×N where:
  * L is the number of energy states
  * N is the number of fermions
  * Columns represent single-particle states

# Returns
- `κ`: Vector of Int, length L indicating the order in which states were sampled

# Throws
- `ArgumentError`: If matrix dimensions are invalid or N ≤ 0
- `LinearAlgebra.SingularException`: If null space calculation fails
"""
@inline function FFS(rng::AbstractRNG, ensemble_matrix::AbstractMatrix)
    L, N = size(ensemble_matrix)

    # Input validation
    N > 1 || throw(ArgumentError("Number of fermions must be positive and greater than 1"))
    L ≥ N || throw(ArgumentError("Number of states must be ≥ number of fermions"))

    # Initialize
    perm = randperm(rng, N)
    U = ensemble_matrix[:, perm]
    κ = zeros(Int, L)
    available = trues(L)
    state_indices = collect(1:L)
    sampled_indices = Int[]

    # Sample first state
    probs = abs2.(U[:, 1])
    sum(probs) ≈ 1.0 || @warn "Probabilities don't sum to 1: $(sum(probs))"
    sampled_state = sample(rng, state_indices, Weights(probs))
    κ[sampled_state] = 1
    available[sampled_state] = false
    push!(sampled_indices, sampled_state)

    # Initial null vector
    null_vector = normalize([-U[sampled_state, 2] / U[sampled_state, 1], 1])

    # Sample remaining states
    @inbounds for i in 2:(N - 1)
        # Calculate probabilities
        probs = compute_probabilities(U, available, null_vector, i)

        # Sample next state
        sampled_state = sample(rng, state_indices[available], Weights(probs))
        κ[sampled_state] = i
        available[sampled_state] = false
        push!(sampled_indices, sampled_state)

        # Update null vector
        U_sampled = U[sampled_indices, 1:(i + 1)]
        null_vector = compute_null_vector(U_sampled)
    end

    # Sample final state
    probs = compute_probabilities(U, available, null_vector, N)
    sampled_state = sample(rng, state_indices[available], Weights(probs))
    κ[sampled_state] = N

    return κ
end

"""Helper function to compute sampling probabilities"""
function compute_probabilities(
        U::AbstractMatrix, available::BitVector, null_vector::Vector, i::Int)
    probs = abs2.((view(U, :, 1:i) * null_vector)[available])
    sum(probs) ≈ 1.0 || @warn "Probabilities don't sum to 1: $(sum(probs))"
    return probs
end

"""Helper function to compute null vector with stability checks"""
function compute_null_vector(U_sampled::AbstractMatrix)
    try
        null_vec = nullspace(U_sampled)
        # Check if nullspace returned valid result
        if size(null_vec, 2) != 1
            throw(LinearAlgebra.SingularException(size(U_sampled, 2)))
        end
        return vec(null_vec)
    catch e
        if e isa LinearAlgebra.SingularException
            return mynullspace(U_sampled)
        end
        rethrow(e)
    end
end

# Default RNG version
FFS(ensemble_matrix::AbstractMatrix) = FFS(Random.default_rng(), ensemble_matrix)
