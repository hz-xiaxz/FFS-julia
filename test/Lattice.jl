using Test
using FastFermionSampling

@testset "Lattice" begin
    # tests for periodical boundary condition lattice
    sp_lat = LatticeRectangular(4, 4, Periodic())
    @test sp_lat.ns == 4 * 4

    @test Set(sp_lat.neigh[1]) == Set([2, 5, 13, 4])
    @test Set(sp_lat.neigh[2]) == Set([1, 3, 14, 6])
    @test Set(sp_lat.neigh[16]) == Set([15, 12, 13, 4])
    @test length(sp_lat.neigh) == sp_lat.ns

    rp_lat = LatticeRectangular(4, 6, Periodic())
    @test rp_lat.ns == 4 * 6

    @test Set(rp_lat.neigh[1]) == Set([2, 5, 21, 4])
    @test Set(rp_lat.neigh[2]) == Set([1, 3, 22, 6])
    @test Set(rp_lat.neigh[24]) == Set([23, 21, 4, 20])
    @test length(rp_lat.neigh) == rp_lat.ns

    # tests for open boundary condition lattice
    so_lat = LatticeRectangular(4, 4, Open())
    @test so_lat.ns == 4 * 4

    @test Set(so_lat.neigh[1]) == Set([2, 5])
    @test Set(so_lat.neigh[2]) == Set([1, 3, 6])
    @test Set(so_lat.neigh[4]) == Set([3, 8])
    @test Set(so_lat.neigh[13]) == Set([9, 14])
    @test Set(so_lat.neigh[16]) == Set([15, 12])
    @test Set(so_lat.neigh[10]) == Set([9, 11, 6, 14])

    ro_lat = LatticeRectangular(4, 6, Open())
    @test ro_lat.ns == 4 * 6

    @test Set(ro_lat.neigh[1]) == Set([2, 5])
    @test Set(ro_lat.neigh[2]) == Set([1, 3, 6])
    @test Set(ro_lat.neigh[4]) == Set([3, 8])
    @test Set(ro_lat.neigh[10]) == Set([9, 11, 14, 6])
    @test Set(ro_lat.neigh[12]) == Set([11, 8, 16])
    @test Set(ro_lat.neigh[24]) == Set([23, 20])

end
