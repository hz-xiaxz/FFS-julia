using Test, FastFermionSampling
using LinearAlgebra

@testset "Gutzwiller" begin
    g = 1.0
    lat = LatticeRectangular(2, 2, Periodic())
    orb = AHmodel(lat, 1.0, 1.0, 1.0, 2, 2)
    conf_up = BitVector([1, 0, 1, 0])
    conf_down = BitVector([0, 1, 0, 1])
    ansatz = Gutzwiller(orb, conf_up, conf_down, g)
    @test ansatz.g == g
    @test ansatz.G == exp(g * ansatz.Og)
    @test ansatz.Og == -1 / 2 * sum(@. (conf_up + conf_down - (orb.N_down + orb.N_up) / lat.ns)^2)
    @test ansatz.conf_up == conf_up
    @test ansatz.conf_down == conf_down
    ansatz2 = Gutzwiller(orb, conf_up, conf_down, 0.0)
    @test ansatz2.G == 1.0
end

@testset "fast_update" begin
    conf_up = BitVector([1, 0, 1, 0])
    conf_down = BitVector([0, 1, 0, 1])
    orb = AHmodel(LatticeRectangular(2, 2, Periodic()), 1.0, 1.0, 1.0, 2, 2)
    U_upinvs = inv(orb.U_up[conf_up, :])
    U_downinvs = inv(orb.U_down[conf_down, :])
    new_conf_up = BitVector([0, 1, 1, 0])
    new_conf_down = BitVector([1, 0, 0, 1])
    @test FastFermionSampling.fast_update(orb.U_up, U_upinvs, new_conf_up, conf_up) ≈ det(orb.U_up[new_conf_up, :]) / det(orb.U_up[conf_up, :])
    @test FastFermionSampling.fast_update(orb.U_down, U_downinvs, new_conf_down, conf_down) ≈ det(orb.U_down[new_conf_down, :]) / det(orb.U_down[conf_down, :])
end