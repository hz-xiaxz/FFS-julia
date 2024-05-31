module FastFermionSampling

# Write your package code here.
using Random
using StatsBase
using LinearAlgebra
using BitBasis

export FFS, AHmodel, LatticeRectangular, getHmat
export Periodic, Open, getxprime
export Gutzwiller

include("Lattices.jl")
include("Orbitals.jl")
include("FFS.jl")
include("Ansatz.jl")

end
