module FastFermionSampling

# Write your package code here.
using Random
using StatsBase
using LinearAlgebra

export FFS

include("Lattices.jl")
include("FFS.jl")

end
