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
    perm = randperm(rng, N)
    U = ensemble_matrix[:, perm]
    κ = zeros(Int, L)
    available = trues(L)
    state_indices = collect(1:L)
    sampled_indices = Int[]

    # For x1 case, P(x1;m) = |U_{x1, m1}|^2
    probs = abs2.(U[:, 1])
    sampled_state = sample(rng, state_indices, Weights(probs))
    κ[sampled_state] = 1
    available[sampled_state] = false
    push!(sampled_indices, sampled_state)
    null_vector = normalize([-U[sampled_state, 2] / U[sampled_state, 1], 1])
    @inbounds for i in 2:(N - 1)
        prob = abs2.((view(U, :, 1:i) * null_vector)[available])
        sampled_state = sample(rng, state_indices[available], Weights(prob))
        κ[sampled_state] = i
        available[sampled_state] = false
        push!(sampled_indices, sampled_state)
        # now compute next n_vec
        # I suggest not using the gaussian elimination
        U_x = U[sampled_indices, 1:i]
        B = -U[1:i, i + 1]
        null_vector = normalize([U_x \ B; 1])
    end
    prob = abs2.((view(U, :, 1:N) * null_vector)[available])
    sampled_state = sample(rng, state_indices[available], Weights(prob))
    κ[sampled_state] = N
    return κ
end

FFS(u::AbstractMatrix) = FFS(Random.default_rng(), u)
