using FastFermionSampling
using Test

@testset "FastFermionSampling.jl" begin
    # Write your tests here.
    include("FFS.jl")
    include("Lattice.jl")
    include("Orbitals.jl")
    include("Ansatz.jl")
end