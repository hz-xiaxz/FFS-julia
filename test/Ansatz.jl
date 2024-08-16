using Test
using FastFermionSampling
using LinearAlgebra
using BitBasis

@testset "Gutzwiller" begin
    g = 1.0
    lat = LatticeRectangular(2, 2, Periodic())
    orb = AHmodel(lat, 1.0, 1.0, 1.0, 2, 2)
    conf_up = BitVector([1, 0, 1, 0])
    conf_down = BitVector([0, 1, 0, 1])
    ansatz = FastFermionSampling.Gutzwiller(g)
    @test ansatz.g == g
    Og = FastFermionSampling.getOg(orb, conf_up, conf_down)
    @test Og == -1 / 2 * sum(@. (conf_up + conf_down - (orb.N_down + orb.N_up) / lat.ns)^2)
end

@testset "fast_update" begin
    conf_up = BitVector([1, 0, 1, 0])
    conf_down = BitVector([0, 1, 0, 1])
    orb = AHmodel(LatticeRectangular(2, 2, Periodic()), 1.0, 1.0, 1.0, 2, 2)

    U_upinvs = orb.U_up[conf_up, :] \ I
    U_downinvs = orb.U_down[conf_down, :] \ I
    @test inv(orb.U_up[conf_up, :]) ≈ orb.U_up[conf_up, :] \ I
    @test inv(orb.U_down[conf_down, :]) ≈ orb.U_down[conf_down, :] \ I

    conf_upstr = LongBitStr(conf_up)
    conf_downstr = LongBitStr(conf_down)
    new_conf_up = BitVector([0, 1, 1, 0])
    new_conf_down = BitVector([1, 0, 0, 1])
    new_conf_upstr = LongBitStr([0, 1, 1, 0])
    new_conf_downstr = LongBitStr([1, 0, 0, 1])

    @test FastFermionSampling.fast_update(orb.U_up, U_upinvs, new_conf_upstr, conf_upstr) ≈
          det(orb.U_up[new_conf_up, :]) / det(orb.U_up[conf_up, :])
    @test FastFermionSampling.fast_update(
        orb.U_down, U_downinvs, new_conf_downstr, conf_downstr) ≈
          det(orb.U_down[new_conf_down, :]) / det(orb.U_down[conf_down, :])

    # LongBitStr testset
    conf_up = BitVector([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    conf_down = BitVector([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    conf_upstr = LongBitStr(conf_up)
    conf_downstr = LongBitStr(conf_down)
    lat = LatticeRectangular(4, 4, Periodic())
    orb = AHmodel(lat, 1.0, 1.0, 1.0, 8, 8)

    U_upinvs = inv(orb.U_up[conf_up, :])
    U_downinvs = inv(orb.U_down[conf_down, :])
    conf_upstr = LongBitStr(conf_up)
    conf_downstr = LongBitStr(conf_down)
    new_conf_up = BitVector([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]) # hop from 4 to 8
    new_conf_down = BitVector([1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]) # hop from 5 to 1
    new_conf_upstr = LongBitStr(new_conf_up)
    new_conf_downstr = LongBitStr(new_conf_down)

    @test FastFermionSampling.fast_update(orb.U_up, U_upinvs, new_conf_upstr, conf_upstr) ≈
          det(orb.U_up[new_conf_up, :]) / det(orb.U_up[conf_up, :])
    @test FastFermionSampling.fast_update(
        orb.U_down, U_downinvs, new_conf_downstr, conf_downstr) ≈
          det(orb.U_down[new_conf_down, :]) / det(orb.U_down[conf_down, :])
    @test FastFermionSampling.fast_update(
        orb.U_up, U_upinvs, SubDitStr(new_conf_upstr, 1, 16), conf_upstr) ≈
          det(orb.U_up[new_conf_up, :]) / det(orb.U_up[conf_up, :])
    @test FastFermionSampling.fast_update(
        orb.U_down, U_downinvs, SubDitStr(new_conf_downstr, 1, 16), conf_downstr) ≈
          det(orb.U_down[new_conf_down, :]) / det(orb.U_down[conf_down, :])
    @test FastFermionSampling.fast_update(
        orb.U_up, U_upinvs, SubDitStr(conf_upstr, 1, 16), conf_upstr) == 1.0
end

@testset "fast_G_update" begin
    conf_up = BitVector([1, 0, 1, 0])
    conf_down = BitVector([0, 1, 0, 1])
    orb = AHmodel(LatticeRectangular(2, 2, Periodic()), 1.0, 1.0, 1.0, 2, 2)
    n_mean = (orb.N_up + orb.N_down) / orb.lattice.ns
    conf = LongBitStr(vcat(conf_up, conf_down))
    g = 1.0
    @test FastFermionSampling.fast_G_update(conf, conf, g, n_mean) == 1.0
    @test FastFermionSampling.fast_G_update(
        LongBitStr([0, 1, 1, 0, 0, 1, 0, 1]), conf, g, n_mean) == exp(-g / 2 *
              ((0 - n_mean)^2 + (2 - n_mean)^2 + 2 * (1 - n_mean)^2 - 4 * (1 - n_mean)^2))
end


function OLbm(n::Int)
    lat = LatticeRectangular(n, n, Periodic())
    orb = AHmodel(lat, 1.0, 1.0, 1.0, n^2 ÷ 2, n^2 ÷ 2)
    conf_up = FFS(orb.U_up)
    conf_down = FFS(orb.U_down)
    OL = FastFermionSampling.getOL(orb, conf_up, conf_down, 1.0)
end
