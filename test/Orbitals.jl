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

@testset "getxprime" begin
    lat = LatticeRectangular(2, 2, Periodic())
    orb = AHmodel(lat, 1.0, 1.0, 1.0, 2, 2)
    x = [true, true, false, false, false, true, true, false]
    xprime = getxprime(orb, x)
    occp = x[1:4] + x[5:end]
    @show occp
    @test length(keys(xprime)) == 5
    @test xprime[x] ≈ sum(occp .* orb.omega) + 1.0
    @test xprime[[false, true, true, false, false, true, true, false]]  == -2
    @test xprime[[true, false, false, true, false, true, true, false]]  == -2
    @test xprime[[true, true, false, false, false, true, false, true]]  == -2
    @test xprime[[true, true, false, false, false, false, true, true]]  == -2
end
