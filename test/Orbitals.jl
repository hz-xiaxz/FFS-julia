using FastFermionSampling
using Test

@testset "AHmodel" begin
    lattice = LatticeRectangular(4, 4, Periodic())
    model = AHmodel(lattice, 1.0, 1.0, 1.0, 8, 8)
    @test model.N_up == 8
    @test model.N_down == 8
    @test length(model.omega) == 16
    # how to test Hmat?
end