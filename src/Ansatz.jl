abstract type AbstractAnsatz end
struct Gutzwiller <: AbstractAnsatz
    conf_up::BitVector
    conf_down::BitVector
    g::Float64
    G::Float64
    Og::Float64
end

"""
add Gutzwiller Ansatz where G  = exp(-g/2 ∑_i (n_i - n_mean)^2), Ψ_G = G Ψ_0
"""
function Gutzwiller(orbitals::AHmodel, conf_up::BitVector, conf_down::BitVector, g::Float64)
    occupation = conf_up + conf_down
    n_mean = (orbitals.N_up + orbitals.N_down) / orbitals.lattice.ns
    Og = -1 / 2 * sum(@. (occupation - n_mean)^2)
    G = exp(g * Og)
    return Gutzwiller(conf_up, conf_down, g, G, Og)
end