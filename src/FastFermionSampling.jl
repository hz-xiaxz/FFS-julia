module FastFermionSampling

# Write your package code here.
using Random
using StatsBase
using LinearAlgebra
using BitBasis
using Carlo
using HDF5
using GenericLinearAlgebra
using SparseArrays
using ArnoldiMethod

export FFS, AHmodel, LatticeRectangular, getHmat, fast_G_update, Spin, Up, Down
export Periodic, Open, getxprime
export MC, tilde_U, is_occupied, add_hop!, add_spin_hopping!

include("Lattices.jl")
include("Orbitals.jl")
include("FFS.jl")
include("Ansatz.jl")
include("MonteCarlo.jl")
# include("ED.jl")

end
