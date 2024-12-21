using Test
using FastFermionSampling
using LinearAlgebra
using FastFermionSampling: Spin, Up, Down, getOL, compute_contribution, classify_term,
                           site_occupation

@testset "Gutzwiller" begin
    g = 1.0
    lat = LatticeRectangular(2, 2, Periodic())
    orb = AHmodel(lat, 1.0, 1.0, 1.0, 2, 2)
    κup = [1, 0, 2, 0]
    κdown = [0, 1, 0, 2]
    ansatz = FastFermionSampling.Gutzwiller(g)
    params = Dict(:nx => 2, :ny => 2, :B => "Periodic", :t => 1.0,
        :W => 1.0, :U => 1.0, :N_up => 2, :N_down => 2, :g => 1.0)
    mc = MC(params)
    mc.κup = κup
    mc.κdown = κdown
    @test ansatz.g == g
    Og = FastFermionSampling.getOg(mc)
    n_mean = (mc.model.N_up + mc.model.N_down) / mc.model.lattice.ns
    @test Og ==
          -1 / 2 * sum(@. (site_occupation(mc.κup) + site_occupation(mc.κdown) - n_mean)^2)
end

@testset "fast_G_update" begin
    @testset "Basic functionality" begin
        # Setup a 2x2 lattice with 2 up and 2 down electrons
        κup = [1, 0, 2, 0]    # electrons at sites 1 and 3
        κdown = [0, 1, 0, 2]  # electrons at sites 2 and 4
        g = 1.0

        # Test no movement case (should return 1.0)
        @test fast_G_update(κup, κdown, g; K = 1, l = 1, spin = Up) ≈ 1.0

        # Test single hop cases
        @testset "Up spin hops" begin
            # Hop from site 1 to 2 (empty → single occupation)
            ratio = fast_G_update(κup, κdown, g; K = 2, l = 1, spin = Up)
            @test ratio ≈ exp(-g * (1 - 1 + 1)) ≈ exp(-g)

            # Hop from site 1 to 4 (single → double occupation)
            ratio = fast_G_update(κup, κdown, g; K = 4, l = 1, spin = Up)
            @test ratio ≈ exp(-g * (1 - 1 + 1)) ≈ exp(-g)
        end

        @testset "Down spin hops" begin
            # Hop from site 2 to 1 (single → double occupation)
            ratio = fast_G_update(κup, κdown, g; K = 1, l = 1, spin = Down)
            @test ratio ≈ exp(-g * (1 - 1 + 1)) ≈ exp(-g)

            # Hop from site 2 to 3 (single → double occupation)
            ratio = fast_G_update(κup, κdown, g; K = 3, l = 1, spin = Down)
            @test ratio ≈ exp(-g * (1 - 1 + 1)) ≈ exp(-g)
        end
    end

    @testset "Edge cases" begin
        κup = [1, 0, 0, 0]
        κdown = [0, 1, 0, 0]
        g = 1.0

        # Test g = 0 case (should always return 1.0)
        @test fast_G_update(κup, κdown, 0.0; K = 2, l = 1, spin = Up) ≈ 1.0

        # Test large g case
        large_g = 10.0
        ratio = fast_G_update(κup, κdown, large_g; K = 2, l = 1, spin = Up)
        @test ratio ≈ exp(-large_g)
    end

    @testset "Error handling" begin
        κup = [1, 0, 0, 0]
        κdown = [0, 0, 0, 0]
        g = 1.0

        # Test invalid site index
        @test_throws AssertionError fast_G_update(κup, κdown, g; K = 5, l = 1, spin = Up)

        # Test invalid state label
        @test_throws ArgumentError fast_G_update(κup, κdown, g; K = 2, l = 2, spin = Up)

        # Test mismatched configuration lengths
        @test_throws AssertionError fast_G_update(
            κup[1:3], κdown, g; K = 2, l = 1, spin = Up)

    end
end

function OLbm(n::Int)
    lat = LatticeRectangular(n, n, Periodic())
    orb = AHmodel(lat, 1.0, 1.0, 1.0, n^2 ÷ 2, n^2 ÷ 2)
    conf_up = FFS(orb.U_up)
    conf_down = FFS(orb.U_down)
    OL = FastFermionSampling.getOL(orb, conf_up, conf_down, 1.0)
end

@testset "Local Energy and Parameter Gradient" begin
    # Setup common test data
    params = Dict(:nx => 2, :ny => 2, :B => "Periodic", :t => 1.0,
        :W => 1.0, :U => 1.0, :N_up => 2, :N_down => 2, :g => 1.0)
    test_mc = MC(params)

    @testset "classify_term" begin
        # Test diagonal term
        @test classify_term((-1, -1, -1, -1)) == FastFermionSampling.Diagonal

        # Test down hopping term
        @test classify_term((-1, -1, 1, 2)) == FastFermionSampling.DownHop

        # Test up hopping term
        @test classify_term((1, 2, -1, -1)) == FastFermionSampling.UpHop
    end

    @testset "compute_contribution" begin
        mc1 = MC(params)
        mc1.κup = [1, 2, 0, 0]    # Two up electrons at sites 1 and 2
        mc1.κdown = [0, 0, 1, 2]  # Two down electrons at sites 3 and 4
        mc1.W_up = ones(4, 4)      # Simple test matrix
        mc1.W_down = ones(4, 4)    # Simple test matrix

        # Test diagonal term
        diag_key = (-1, -1, -1, -1)
        @test compute_contribution(
            FastFermionSampling.Diagonal, 2.0, diag_key, mc1) ≈ 2.0

        # Test down hopping term
        down_key = (-1, -1, 1, 2)
        down_contrib = compute_contribution(
            FastFermionSampling.DownHop, 1.0, down_key, mc1)
        @test typeof(down_contrib) == Float64
        @test !isnan(down_contrib)

        # Test up hopping term
        up_key = (1, 2, -1, -1)
        up_contrib = compute_contribution(
            FastFermionSampling.UpHop, 1.0, up_key, mc1)
        @test typeof(up_contrib) == Float64
        @test !isnan(up_contrib)
    end

    @testset "getOL" begin
        mc2 = MC(params)
        mc2.κup = [1, 2, 0, 0]
        mc2.κdown = [0, 0, 1, 2]
        mc2.W_up = ones(4, 4)
        mc2.W_down = ones(4, 4)

        # Calculate OL
        OL = getOL(mc2)

        # Basic sanity checks
        @test typeof(OL) == Float64
        @test !isnan(OL)
        @test !isinf(OL)
    end

end
