abstract type AbstractOrbitals end

"""
Anderson-Hubbard Model
----------------------
lattice: LatticeRectangular{B}
    The lattice structure   
t: Float64
    Hopping parameter
W: Float64
    Disorder strength
U: Float64
    On-site interaction strength
Nup: Int
    Number of up spins
Ndown: Int
    Number of down spins
omega: Vector{Float64}
    Random on-site energies
U_up: Matrix
    Unitary matrix for up spins
U_down: Matrix
    Unitary matrix for down spins
"""
struct AHmodel{B, RT} <: AbstractOrbitals
    lattice::LatticeRectangular{B}
    t::Float64
    W::Float64
    U::Float64
    N_up::Int
    N_down::Int
    omega::Vector{Float64}
    U_up::Matrix{RT}
    U_down::Matrix{RT}
end

"""
Get the non-interacting Anderson model Hamiltonian Matrix to construct HF states
"""
function getHmat(lattice::LatticeRectangular, t::Float64, W::Float64, omega::Vector{Float64}, N_up::Int, N_down::Int)
    ns = lattice.ns
    N = N_up + N_down
    @assert ns / N == 1 "Should be hall-filling"
    tunneling = zeros(Float64, ns, ns)
    for (idx, neighbors) in enumerate(lattice.neigh)
        for j in neighbors
            tunneling[idx, j] = -t
        end
    end
    onsite = diagm(ones(Float64, ns) .* omega)
    H_mat = tunneling + onsite
    return H_mat
end


"""
generate Anderson-Hubbard model and get the sampling ensemble
"""
function AHmodel(lattice::LatticeRectangular, t::Float64, W::Float64, U::Float64, N_up::Int, N_down::Int)
    omega = randn(Float64, lattice.ns) * W / 2
    H_mat = getHmat(lattice, t, W, omega, N_up, N_down)
    # get sampling ensemble U_up and U_down
    vals, vecs = eigen(H_mat)
    # select N lowest eigenvectors as the sampling ensemble
    sorted_indices = sortperm(vals)
    U_up = vecs[:, sorted_indices[1:N_up]]
    U_down = vecs[:, sorted_indices[1:N_down]]
    return AHmodel(lattice, t, W, U, N_up, N_down, omega, U_up, U_down)

end

"""
return |x'> = H|x> where H = -t ∑_<i,j> c_i^† c_j + U ∑_i n_i↓ n_i↑ + ∑_i ω_i n_i
"""
function getxprime(orb::AHmodel, x::BitStr{N, T}) where {N, T}
    @assert N == 2 * length(orb.omega) "x should have the same 2x length as omega (2 x $(length(orb.omega))), got: $N"
    L = length(x) ÷ 2  # Int division
    xprime = Dict{typeof(x),Float64}() # HACK, I use Dict to store the Fock Basis and its coefficient, will it be more efficient to use two-element tuple? 
    # consider the spin up case
    @inbounds for i in 1:L
        if readbit(x, i) == 1
            xprime[x] = get!(xprime, x, 0.0) + orb.omega[i] # On-site energy
            if readbit(x, i) == 1 && readbit(x, i+L) == 1 #occp[i] == 2
                xprime[x] += orb.U # Hubbard Interaction
            end
            for neigh in orb.lattice.neigh[i]
                if readbit(x, neigh) == 0
                    _x = x
                    _x &= ~indicator(T, i)
                    _x |= indicator(T, neigh)
                    xprime[_x] = get!(xprime, _x, 0.0) - orb.t # Hopping 
                end
            end
        end
    end
    @inbounds for i in L+1:length(x)
        if readbit(x, i) == 1
            xprime[x]  = get!(xprime, x, 0.0) + orb.omega[i-L] # On-site energy
            for neigh in orb.lattice.neigh[i-L]
                if readbit(x, neigh) == 0
                    _x = x
                    _x &= ~indicator(T, i)
                    _x |= indicator(T, neigh+L)
                    xprime[_x] = get!(xprime, _x, 0.0) - orb.t # Hopping 
                end
            end
        end
    end
    return xprime
end