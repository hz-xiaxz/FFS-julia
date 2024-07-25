using QuantumLattices
using ExactDiagonalization
using LinearAlgebra: eigen

function doED(
        Lx::Int, Ly::Int, t::Float64, U::Float64, omega::Vector{Float64}, boundary::Char)
    @assert length(omega)==Lx * Ly "onsite disorder should be the same as the number of sites, got $(length(omega)) and $(Lx * Ly)"
    global omega_reshape = reshape(omega, Lx, Ly)

    # define the unitcell of the square lattice
    unitcell = Lattice([0.0, 0.0]; name = :Square, vectors = [[1.0, 0.0], [0.0, 1.0]])

    # define a finite 3×4 cluster of the square lattice with open boundary condition
    boundary = ntuple(i -> boundary, 2)
    lattice = Lattice(unitcell, (Lx, Ly), boundary)

    # define the Hilbert space (single-orbital spin-1/2 complex fermion)
    hilbert = Hilbert(site => Fock{:f}(1, 2) for site in 1:length(lattice))

    # define the quantum number of the sub-Hilbert space in which the computation to be carried out
    # here the particle number is set to be `length(lattice)` and Sz is set to be 0
    quantumnumber = SpinfulParticle(length(lattice), 0)

    # define the terms, i.e. the nearest-neighbor hopping and the Hubbard interaction
    t_term = Hopping(:t, -t, 1)
    U_term = Hubbard(:U, U)

    #########################

    function disorder(bond::Bond)
        coordinate = bond.points[1].rcoordinate
        onsite = omega_reshape[Int(coordinate[1]) + 1, Int(coordinate[2]) + 1]
        return onsite
    end

    ω_up = Term{:Onsite}(:ωup, 1.0, 0,
        Coupling(Index(:, FID{:f}(:, 1 // 2, 2)), Index(:, FID{:f}(:, 1 // 2, 1))),
        true; amplitude = disorder, ismodulatable = true)

    ω_down = Term{:Onsite}(:ωdown, 1.0, 0,
        Coupling(Index(:, FID{:f}(:, -1 // 2, 2)), Index(:, FID{:f}(:, -1 // 2, 1))),
        true; amplitude = disorder, ismodulatable = true)

    ###########################

    # define the exact diagonalization algorithm for the Fermi Hubbard model
    ed = ED(lattice, hilbert, (t_term, U_term, ω_up, ω_down), quantumnumber)

    # find the ground state and its energy
    eigensystem = eigen(ed; nev = 1)

    # Ground state energy should be -4.913259209075605
    eigensystem.values
end