function FFS(u::AbstractMatrix, L::Int, N::Int)::Vector{Bool}
    # U: Matrix{ComplexF64}: the sampling ensemble
    # L: Int: the number of energy states
    # N: Int: the number of fermions
    v = randperm(N)
    U = u[:, v]
    sampled = falses(L)
    avail = trues(L)
    groud_set = collect(1:L)
    # For x1 case, P(x1;m) = |U_{x1, m1}|^2 
    p = abs2.(U[:, 1])
    x_new = sample(1:L, Weights(p))
    sampled[x_new] = true
    avail[x_new] = false
    n_vec = normalize([-U[x_new, 2] / U[x_new, 1], 1])
    for i in 2:N
        prob = abs2.(U[avail, 1:i] * n_vec)
        x_new = sample(groud_set[avail], Weights(prob))
        sampled[x_new] = true
        avail[x_new] = false
        if i == N
            break
        end
        # now compute next n_vec
        # I suggest not using the gaussian elimination
        U_x = U[sampled, 1:i]
        B = -U[1:i, i+1]
        n_vec = normalize([U_x \ B; 1])
    end
    return sampled
end