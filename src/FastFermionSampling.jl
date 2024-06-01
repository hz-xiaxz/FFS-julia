module FastFermionSampling

# Write your package code here.
using Random
using StatsBase
using LinearAlgebra
using BitBasis
using Carlo

export FFS, AHmodel, LatticeRectangular, getHmat, MC
export Periodic, Open, getxprime
export Gutzwiller

include("Lattices.jl")
include("Orbitals.jl")
include("FFS.jl")
include("Ansatz.jl")
include("MonteCarlo.jl")

end
