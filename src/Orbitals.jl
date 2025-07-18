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
        N_down::Int
) where {B}
    ns = lattice.ns
    N = N_up + N_down
    @assert ns==N "Should be hall-filling"
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
    check_shell(E::AbstractArray, Nup::Int, ns::Int)

Check if the whole degenerate shell of single particle eigenstates are filled.
"""
function check_shell(E::AbstractArray, Nup::Int, ns::Int)
    shell_pool = []
    # iteratively find degenerate spaces
    start_shell = 1
    while start_shell < length(E)
        num = findlast(x -> isapprox(x, E[start_shell], atol = 1e-10), E)
        push!(shell_pool, (start_shell, num))
        start_shell = num + 1
    end
    # the number of N_up and N_down should be at least > num
    shell = filter(
        x -> (x[1] <= Nup && x[2] > Nup) || (x[1] <= (ns - Nup) && x[2] > (ns - Nup)),
        shell_pool
    )
    return isempty(shell)
end

"""
    AHmodel(lattice::LatticeRectangular{B}, t::Float64, W::Float64, U::Float64, N_up::Int, N_down::Int)

Generate Anderson-Hubbard model and get the sampling ensemble

Raise warning if the shell of degenerate eigenstates are not whole-filled
"""
function AHmodel(
        lattice::LatticeRectangular{B},
        t::Float64,
        W::Float64,
        U::Float64,
        N_up::Int,
        N_down::Int
) where {B}
    omega = randn(Float64, lattice.ns) * W / 2
    # omega = ones(Float64, lattice.ns) * W / 2
    H_mat = getHmat(lattice, t, omega, N_up, N_down)
    # get sampling ensemble U_up and U_down
    H_mat_sparse = sparse(H_mat)
    # select N lowest eigenvectors as the sampling ensemble
    # note eigenvalues could be degenerate, so a better way is to use Arnoldi method
    # also Schur decomposition is more stable than eigen
    nev = max(N_up, N_down)
    decomp,
    history = ArnoldiMethod.partialschur(
        H_mat_sparse, nev = nev, which = :SR)
    U_up = decomp.Q[:, 1:N_up]
    U_down = decomp.Q[:, 1:N_down]
    if !check_shell(diag(decomp.R), N_up, lattice.ns)
        @warn "The shell of degenerate eigenstates are not whole-filled"
    end
    return AHmodel{B}(lattice, t, W, U, N_up, N_down, omega, U_up, U_down)
end

# this function is for debugging
function fixedAHmodel(
        lattice::LatticeRectangular{B},
        t::Float64,
        W::Float64,
        U::Float64,
        N_up::Int,
        N_down::Int
) where {B}
    omega = zeros(Float64, lattice.ns) * W / 2
    H_mat = getHmat(lattice, t, omega, N_up, N_down)
    # get sampling ensemble U_up and U_down
    H_mat_sparse = sparse(H_mat)
    # select N lowest eigenvectors as the sampling ensemble
    nev = max(N_up, N_down)
    decomp,
    history = ArnoldiMethod.partialschur(
        H_mat_sparse, nev = nev, which = :SR)
    U_up = decomp.Q[:, 1:N_up]
    U_down = decomp.Q[:, 1:N_down]
    return AHmodel{B}(lattice, t, W, U, N_up, N_down, omega, U_up, U_down)
end

"""
    getxprime(orb::AHmodel{B}, κup::Vector{Int}, κdown::Vector{Int})

Compute ``|x'> = H|x>`` where ``H = -t ∑_{<i,j>} c_i^† c_j + U ∑_i n_{i↓} n_{i↑} + ∑_i ω_i n_i``
"""
function getxprime(orb::AHmodel{B}, κup::Vector{Int}, κdown::Vector{Int}) where {B}
    @assert length(κup)==length(κdown) "Length of κ↑ and κ↓ should match the number of sites"

    # Initialize result dictionary with better key type
    xprime = Dict{Tuple{Int, Int, Int, Int}, Float64}()

    # Add different contributions
    add_onsite_terms!(xprime, orb, κup, κdown)
    add_hubbard_terms!(xprime, orb, κup, κdown)
    add_hopping_terms!(xprime, orb, κup, κdown)

    return xprime
end


"""Add on-site energy terms"""
function add_onsite_terms!(
        xprime::Dict{Tuple{Int, Int, Int, Int}, Float64},
        orb::AHmodel{B},
        κup::Vector{Int},
        κdown::Vector{Int}
) where {B}
    @inbounds for i in 1:(orb.lattice.ns)
        if is_occupied(κup, i)
            add_diagonal_term!(xprime, orb.omega[i])
        end
        if is_occupied(κdown, i)
            # should be modified if the disorder of spin up and down are different
            add_diagonal_term!(xprime, orb.omega[i])
        end
    end
end

"""Add Hubbard interaction terms"""
function add_hubbard_terms!(
        xprime::Dict{Tuple{Int, Int, Int, Int}, Float64},
        orb::AHmodel{B},
        κup::Vector{Int},
        κdown::Vector{Int}
) where {B}
    @inbounds for i in 1:(orb.lattice.ns)
        if is_occupied(κup, i) && is_occupied(κdown, i)
            add_diagonal_term!(xprime, orb.U)
        end
    end
end

"""Add a diagonal term to the Hamiltonian"""
function add_diagonal_term!(
        xprime::Dict{Tuple{Int, Int, Int, Int}, Float64},
        energy::Float64
)
    key = (-1, -1, -1, -1)
    xprime[key] = get!(xprime, key, 0.0) + energy
end

@enum Spin Up Down

"""Add hopping terms for both spin species"""
function add_hopping_terms!(
        xprime::Dict{Tuple{Int, Int, Int, Int}, Float64},
        orb::AHmodel{B},
        κup::Vector{Int},
        κdown::Vector{Int}
) where {B}
    add_spin_hopping!(xprime, orb, κup, Up)
    add_spin_hopping!(xprime, orb, κdown, Down)
end

"""Add hopping terms for a specific spin"""
function add_spin_hopping!(
        xprime::Dict{Tuple{Int, Int, Int, Int}, Float64},
        orb::AHmodel{B},
        κ::Vector{Int},
        spin::Spin
) where {B}
    @inbounds for (site, occupation) in enumerate(κ)
        iszero(occupation) && continue

        for neighbor in orb.lattice.neigh[site]
            if !is_occupied(κ, neighbor)
                K = neighbor
                l = occupation
                add_hop!(xprime, K, l, spin, -orb.t)
            end
        end
    end
end

"""Add a single hopping term to the Hamiltonian"""
function add_hop!(
        xprime::Dict{Tuple{Int, Int, Int, Int}, Float64},
        K::Int,
        l::Int,
        spin::Spin,
        energy::Float64
)
    key = if spin == Up
        (K, l, -1, -1)
    else
        (-1, -1, K, l)
    end

    xprime[key] = get!(xprime, key, 0.0) + energy
end
