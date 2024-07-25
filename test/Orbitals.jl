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

@testset "getxprime" begin
    lat = LatticeRectangular(2, 2, Periodic())
    orb = AHmodel(lat, 1.0, 1.0, 1.0, 2, 2)
    x = LongBitStr([true, true, false, false, false, true, true, false])
    xprime = getxprime(orb, x)
    occp = x[1:4] + x[5:end]
    @test length(keys(xprime)) == 7
    @test xprime[x] ≈ sum(occp .* orb.omega) + 1.0
    @test xprime[LongBitStr([false, true, true, false, false, true, true, false])] == -1
    @test xprime[LongBitStr([true, false, false, true, false, true, true, false])] == -1
    @test xprime[LongBitStr([true, true, false, false, false, true, false, true])] == -1
    @test xprime[LongBitStr([true, true, false, false, false, false, true, true])] == -1
    @test xprime[LongBitStr([true, true, false, false, true, true, false, false])] == -1
    @test xprime[LongBitStr([true, true, false, false, true, false, true, false])] == -1
    for conf in keys(xprime)
        @test sum([conf...]) == 4
    end
end

@testset "fixedAHmodel" begin
    lat = LatticeRectangular(4, 4, Periodic())
    orb = FastFermionSampling.fixedAHmodel(lat, 1.0, 1.0, 1.0, 8, 8)
    fixedmodel = FastFermionSampling.fixedAHmodel(lat, 1.0, 1.0, 1.0, 8, 8)
    @test fixedmodel.N_up == 8
    @test fixedmodel.N_down == 8
    @test length(fixedmodel.omega) == 16
    @test allequal(fixedmodel.omega) && fixedmodel.omega[1] == 0.0
end

using BenchmarkTools
function bm(n::Int)
    @assert n % 2 == 0
    lat = LatticeRectangular(n, n, Periodic())
    orb = AHmodel(lat, 1.0, 1.0, 1.0, n^2 ÷ 2, n^2 ÷ 2)
    #x = rand(Bool, 2*n^2)
    x = LongBitStr(rand(0:1, 2n^2))
    @btime getxprime($orb, $x)
end

# bm(50): 152.479 ms (4965 allocations: 25.09 MiB) - before opt
# after: 460.041 μs (12 allocations: 211.33 KiB)
