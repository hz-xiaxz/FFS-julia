function FFS(r::AbstractRNG, u::AbstractMatrix)
    """
    U: Matrix{Float64}: the sampling ensemble
    """
    L, N = size(u)
    v = randperm(r, N)
    U = u[:, v]
    sampled = falses(L)
    avail = trues(L)
    groud_set = collect(1:L)
    # For x1 case, P(x1;m) = |U_{x1, m1}|^2 
    p = abs2.(U[:, 1])
    x_new = sample(r, 1:L, Weights(p))
    sampled[x_new] = true
    avail[x_new] = false
    n_vec = normalize([-U[x_new, 2] / U[x_new, 1], 1])
    for i in 2:N-1
        prob = abs2.(view(U, avail, 1:i) * n_vec)
        x_new = sample(r, groud_set[avail], Weights(prob))
        sampled[x_new] = true
        avail[x_new] = false
        # now compute next n_vec
        # I suggest not using the gaussian elimination
        U_x = U[sampled, 1:i]
        B = -U[1:i, i+1]
        n_vec = normalize([U_x \ B; 1])
    end
    prob = abs2.(view(U, avail, 1:N) * n_vec)
    x_new = sample(r, groud_set[avail], Weights(prob))
    sampled[x_new] = true
    avail[x_new] = false
    return sampled
end

FFS(u::AbstractMatrix) = FFS(Random.default_rng(), u)