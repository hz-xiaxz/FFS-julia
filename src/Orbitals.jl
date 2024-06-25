abstract type AbstractOrbitals end

"""
Anderson-Hubbard Model
----------------------

* `lattice` : `LatticeRectangular{B}` The lattice structure   
* `t`: `Float64` Hopping parameter
* `W` : `Float64` Disorder strength, on site energy is sampled from `N(0, W/2)`
* `U` : `Float64` On-site interaction strength
* `Nup` : `Int` Number of up spins
* `Ndown` : `Int` Number of down spins
* `omega` : `Vector{Float64}` Random on-site energies
* `U_up` : `Matrix{Float64}` Unitary matrix for up spins
* `U_down` : `Matrix{Float64}` Unitary matrix for down spins
"""
struct AHmodel{B} <: AbstractOrbitals where {B}
    lattice::LatticeRectangular{B}
    t::Float64
    W::Float64
    U::Float64
    N_up::Int
    N_down::Int
    omega::Vector{Float64}
    U_up::Matrix{Float64}
    U_down::Matrix{Float64}
end

"""
    getHmat(lattice::LatticeRectangular{B}, t::Float64, omega::Vector{Float64}, N_up::Int, N_down::Int)'

Get the non-interacting Anderson model Hamiltonian Matrix to construct Slater Determinants 
"""
function getHmat(
    lattice::LatticeRectangular{B},
    t::Float64,
    omega::Vector{Float64},
    N_up::Int,
    N_down::Int,
) where {B}
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
    AHmodel(lattice::LatticeRectangular{B}, t::Float64, W::Float64, U::Float64, N_up::Int, N_down::Int)

Generate Anderson-Hubbard model and get the sampling ensemble
"""
function AHmodel(
    lattice::LatticeRectangular{B},
    t::Float64,
    W::Float64,
    U::Float64,
    N_up::Int,
    N_down::Int,
) where {B}
    omega = randn(Float64, lattice.ns) * W / 2
    # omega = ones(Float64, lattice.ns) * W / 2
    H_mat = getHmat(lattice, t, omega, N_up, N_down)
    # get sampling ensemble U_up and U_down
    vals, vecs = eigen(H_mat)
    # select N lowest eigenvectors as the sampling ensemble
    sorted_indices = sortperm(vals)
    U_up = vecs[:, sorted_indices[1:N_up]]
    U_down = vecs[:, sorted_indices[1:N_down]]
    return AHmodel{B}(lattice, t, W, U, N_up, N_down, omega, U_up, U_down)
end

# this function is for debugging
function fixedAHmodel(
    lattice::LatticeRectangular{B},
    t::Float64,
    W::Float64,
    U::Float64,
    N_up::Int,
    N_down::Int,
) where {B}
    omega = ones(Float64, lattice.ns) * W / 2
    H_mat = getHmat(lattice, t, omega, N_up, N_down)
    # get sampling ensemble U_up and U_down
    vals, vecs = eigen(H_mat)
    # select N lowest eigenvectors as the sampling ensemble
    # eigen function may have numerical instability problem, see issue(#21)
    sorted_indices = sortperm(vals)
    U_up = vecs[:, sorted_indices[1:N_up]]
    U_down = vecs[:, sorted_indices[1:N_down]]
    return AHmodel{B}(lattice, t, W, U, N_up, N_down, omega, U_up, U_down)
end

"""
    getxprime(orb::AHmodel{B}, x::BitStr{N,T}) where {B,N,T}

return ``|x'> = H|x>``  where ``H = -t ∑_{<i,j>} c_i^† c_j + U ∑_i n_{i↓} n_{i↑} + ∑_i ω_i n_i``
"""
function getxprime(orb::AHmodel{B}, x::BitStr{N,T}) where {B,N,T}
    @assert N == 2 * length(orb.omega) "x should have the same 2x length as omega (2 x $(length(orb.omega))), got: $N"
    L = length(x) ÷ 2  # Int division
    xprime = Dict{typeof(x),Float64}()
    # consider the spin up case
    @inbounds for i = 1:L
        if readbit(x, i) == 1
            xprime[x] = get!(xprime, x, 0.0) + orb.omega[i] # On-site energy
            if readbit(x, i) == 1 && readbit(x, i + L) == 1 # occp[i] == 2
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
    @inbounds for i = L+1:length(x)
        if readbit(x, i) == 1
            xprime[x] = get!(xprime, x, 0.0) + orb.omega[i-L] # On-site energy
            for neigh in orb.lattice.neigh[i-L]
                if readbit(x, neigh + L) == 0
                    _x = x
                    _x &= ~indicator(T, i)
                    _x |= indicator(T, neigh + L)
                    xprime[_x] = get!(xprime, _x, 0.0) - orb.t # Hopping 
                end
            end
        end
    end
    return xprime
end
