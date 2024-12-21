using Test
using FastFermionSampling
using FastFermionSampling: MC, tilde_U, reevaluateW!

@testset "tilde_U tests" begin
    @testset "Basic functionality" begin
        # Create a simple test matrix
        U = [1.0 2.0 3.0
             4.0 5.0 6.0
             7.0 8.0 9.0]
        kappa = [2, 3, 1]

        result = tilde_U(U, kappa)

        # Test output dimensions
        @test size(result) == (3, 3)

        # Test correct row placement
        @test result[1, :] == U[3, :]  # kappa[3] = 1
        @test result[2, :] == U[1, :]  # kappa[1] = 2
        @test result[3, :] == U[2, :]  # kappa[2] = 3
    end

    @testset "Zero kappa entries" begin
        U = [1.0 2.0
             3.0 4.0]
        kappa = [0, 0]

        @test_throws ArgumentError tilde_U(U, kappa)
    end

    @testset "Different matrix shapes" begin
        # Test with rectangular matrix
        U = [1.0 2.0
             3.0 4.0
             5.0 6.0]
        kappa = [1, 0, 2]

        result = tilde_U(U, kappa)

        @test size(result) == (2, 2)
        @test result[1, :] == U[1, :]
        @test result[2, :] == U[3, :]
    end

    @testset "Complex numbers" begin
        U = [1.0+im 2.0+2im
             3.0+3im 4.0+4im]
        kappa = [2, 1]

        result = tilde_U(U, kappa)

        @test result[1, :] == U[2, :]
        @test result[2, :] == U[1, :]
    end

    @testset "Edge cases" begin
        # Empty matrix
        U = Matrix{Float64}(undef, 0, 0)
        kappa = Int[]
        result = tilde_U(U, kappa)
        @test size(result) == (0, 0)

        # Single element
        U = reshape([1.0], 1, 1)
        kappa = [1]
        result = tilde_U(U, kappa)
        @test result == U
    end

    @testset "Input preservation" begin
        U = [1.0 2.0
             3.0 4.0]
        U_original = copy(U)
        kappa = [1, 2]
        kappa_original = copy(kappa)

        result = tilde_U(U, kappa)

        # Test that inputs weren't modified
        @test U == U_original
        @test kappa == kappa_original
    end

    @testset "Invalid inputs" begin
        U = [1.0 2.0
             3.0 4.0]

        # Test kappa with invalid indices
        @test_throws BoundsError tilde_U(U, [3, 1])  # Index 3 is out of bounds

        # Test mismatched dimensions
        @test_throws DimensionMismatch tilde_U(U, [1, 2, 3])
    end

    @testset "Type stability" begin
        U = [1.0 2.0
             3.0 4.0]
        kappa = [1, 2]

        # Test that output type matches input type
        result = tilde_U(U, kappa)
        @test eltype(result) == eltype(U)

        # Test with different types
        U_int = [1 2; 3 4]
        result_int = tilde_U(U_int, kappa)
        @test eltype(result_int) == eltype(U_int)
    end
end

@testset "reevaluateW! tests" begin
    @testset "Basic functionality" begin
        # Create a simple 2x2 test case
        kappa_up = [1, 0, 2, 0]
        kappa_down = [2, 0, 1, 0]
        lat = LatticeRectangular(2, 2, Periodic())
        model = AHmodel(lat, 1.0, 1.0, 1.0, 2, 2)
        mc = MC(model, kappa_up, kappa_down, zeros(2, 2),
            zeros(2, 2), 1.0, [2 * pi / 2, 2 * pi / 2])

        reevaluateW!(mc)

        tilde_U_up = tilde_U(model.U_up, kappa_up)
        tilde_U_down = tilde_U(model.U_down, kappa_down)
        U_upinvs = tilde_U_up \ I
        U_downinvs = tilde_U_down \ I
        W_up_expected = model.U_up * U_upinvs
        W_down_expected = model.U_down * U_downinvs

        @test isapprox(mc.W_up, W_up_expected, atol = 1e-10)
        @test isapprox(mc.W_down, W_down_expected, atol = 1e-10)
    end


end
