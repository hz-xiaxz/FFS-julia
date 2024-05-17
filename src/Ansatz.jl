abstract type AbstractAnsatz end
struct Gutzwiller <: AbstractAnsatz
    conf_up::Vector{Bool}
    conf_down::Vector{Bool}
    g::Float64
    G::Float64
    Og::Float64
end

function Gutzwiller(lattice::LatticeRectangular, orbitals::AHmodel, conf_up::Vector{Bool}, conf_down::Vector{Bool}, g::Float64)
    @assert length(conf_down) == length(conf_up)
    @assert sum(conf_down) == orbitals.N_down
    @assert sum(conf_up) == orbitals.N_up
    ns = lattice.ns
    occupation = conf_up + conf_down
    n_mean = (orbitals.N_up + orbitals.N_down) / ns
    @assert typeof(occupation) == Vector{Int64}
    Og = -1 / 2 * sum(@. (occupation - n_mean)^2)
    G = exp(g * Og)
    return Gutzwiller(conf_up, conf_down, g, G, Og)
end


