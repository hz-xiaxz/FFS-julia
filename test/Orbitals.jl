using FastFermionSampling, BitBasis
using Test

@testset "AHmodel" begin
    lattice = LatticeRectangular(4, 4, Periodic())
    model = AHmodel(lattice, 1.0, 1.0, 1.0, 8, 8)
    @test model.N_up == 8
    @test model.N_down == 8
    @test length(model.omega) == 16
    # how to test Hmat?
end

# @testset "getxprime" begin
#     lat = LatticeRectangular(2, 2, Periodic())
#     orb = AHmodel(lat, 1.0, 1.0, 1.0, 2, 2)
#     x = LongBitStr([true, true, false, false, false, true, true, false])
#     xprime = getxprime(orb, x)
#     occp = x[1:4] + x[5:end]
#     @test length(keys(xprime)) == 7
#     @test xprime[x] ≈ sum(occp .* orb.omega) + 1.0
#     @test xprime[LongBitStr([false, true, true, false, false, true, true, false])] == -1
#     @test xprime[LongBitStr([true, false, false, true, false, true, true, false])] == -1
#     @test xprime[LongBitStr([true, true, false, false, false, true, false, true])] == -1
#     @test xprime[LongBitStr([true, true, false, false, false, false, true, true])] == -1
#     @test xprime[LongBitStr([true, true, false, false, true, true, false, false])] == -1
#     @test xprime[LongBitStr([true, true, false, false, true, false, true, false])] == -1
#     for conf in keys(xprime)
#         @test sum([conf...]) == 4
#     end
# end

@testset "fixedAHmodel" begin
    lat = LatticeRectangular(4, 4, Periodic())
    orb = FastFermionSampling.fixedAHmodel(lat, 1.0, 1.0, 1.0, 8, 8)
    fixedmodel = FastFermionSampling.fixedAHmodel(lat, 1.0, 1.0, 1.0, 8, 8)
    @test fixedmodel.N_up == 8
    @test fixedmodel.N_down == 8
    @test length(fixedmodel.omega) == 16
    @test allequal(fixedmodel.omega) && fixedmodel.omega[1] == 0.0
end


@testset "Orbital Hamiltonian" begin
    @testset "Basic functionality" begin
        lat = LatticeRectangular(2, 2, Periodic())
        orb = AHmodel(lat, 1.0, 1.0, 1.0, 2, 2)

        # Initial state: 2 up electrons and 2 down electrons
        κup = [1, 2, 0, 0]    # First two sites occupied with up spins
        κdown = [0, 1, 2, 0]  # Second and third sites occupied with down spins

        xprime = getxprime(orb, κup, κdown)

        # Test number of terms
        @test length(keys(xprime)) > 0  # Should have diagonal and hopping terms

        # Test diagonal terms
        diagonal_key = (-1, -1, -1, -1)
        @test haskey(xprime, diagonal_key)

        # Expected diagonal energy:
        # - On-site energy for 4 electrons
        # - Hubbard interaction for 1 doubly occupied site
        expected_diagonal = orb.omega[1] + 2 * orb.omega[2] + orb.omega[3] + orb.U
        @test xprime[diagonal_key] ≈ expected_diagonal
    end

    @testset "Hopping terms" begin
        lat = LatticeRectangular(2, 2, Periodic())
        orb = AHmodel(lat, 1.0, 1.0, 1.0, 2, 2)

        # Test up spin hopping
        κup = [1, 0, 0, 0]    # One up electron at first site
        κdown = [0, 0, 0, 0]  # No down electrons

        xprime = getxprime(orb, κup, κdown)

        # Should have hopping terms to neighboring sites
        @test any(k[1] != -1 && k[2] == 1 for k in keys(xprime))  # Up spin hops
        @test all(k[3] == -1 && k[4] == -1 for k in keys(xprime))  # No down spin terms

        # Test down spin hopping
        κup = [0, 0, 0, 0]    # No up electrons
        κdown = [0, 1, 0, 0]  # One down electron at second site

        xprime = getxprime(orb, κup, κdown)

        # Should have hopping terms to neighboring sites
        @test any(k[3] != -1 && k[4] == 1 for k in keys(xprime))  # Down spin hops
        @test all(k[1] == -1 && k[2] == -1 for k in keys(xprime))  # No up spin terms
    end


    @testset "Boundary conditions" begin
        lat = LatticeRectangular(4, 4, Periodic())
        orb = AHmodel(lat, 1.0, 1.0, 1.0, 8, 8)

        κup = vcat(ones(Int, 1), zeros(Int, 15))
        κdown = zeros(Int, 16)

        xprime = getxprime(orb, κup, κdown)

        # Should have hopping terms across periodic boundaries
        boundary_hops = [k for k in keys(xprime) if k[1] == 4 && k[2] == 1]
        @test !isempty(boundary_hops)  # Should have periodic boundary hopping
    end
end
