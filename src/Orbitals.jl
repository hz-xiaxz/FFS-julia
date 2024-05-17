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
struct AHmodel <: AbstractOrbitals
    t::Float64
    W::Float64
    U::Float64
    N_up::Int
    N_down::Int
    omega::Vector{Float64}
    U_up::Matrix
    U_down::Matrix
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
    return AHmodel(t, W, U, N_up, N_down, omega, U_up, U_down)

end
