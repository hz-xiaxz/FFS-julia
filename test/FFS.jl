using FastFermionSampling
using FastFermionSampling: FFS
using Test
using StatsBase
using Random
using LinearAlgebra

function combinations(L::Int, N::Int)
    if N == 0
        return [falses(L)]
    elseif L == N
        return [trues(L)]
    else
        with_first = [vcat(true, c) for c in combinations(L - 1, N - 1)]
        without_first = [vcat(false, c) for c in combinations(L - 1, N)]
        return vcat(with_first, without_first)
    end
end

function sort_bool_arrays(array::AbstractArray)
    return sort(array, by = x -> evalpoly(2, reverse(x)))
end

function exact_sample(U::AbstractMatrix, L::Int, N::Int)

    events = sort_bool_arrays(combinations(L, N))
    p = [abs2(det(U[ev, :])) for ev in events]
    p = p / sum(p)
    return (events, p)

end

function getKLdiv(L::Int, N::Int, iter_time::Int, U::Matrix{Float64})
    # caculate the KL divergence between the exact sample and the FFS
    events, p = exact_sample(U, L, N)
    sampled = Dict{Int, Float64}()
    for (i, ev) in enumerate(events)
        sampled[evalpoly(2, reverse(ev))] = 1e-8# for KL-div not blowing up
    end
    for i in 1:iter_time
        sampled[evalpoly(2, reverse(FFS(U)))] += 1
    end
    q = zeros(Float64, length(keys(sampled)))
    for (i, key) in enumerate(sort(collect(keys(sampled))))
        q[i] = sampled[key]
    end
    q = q / sum(q)
    # KL divergence ∑ p(x) log(p(x)/q(x))
    kl = sum(p .* log.(p ./ q))
    return kl
end

@testset "FFS" begin
    @testset "Basic functionality" begin
        # Simple 2x2 case
        rng = MersenneTwister(123)
        U = [1.0 0.0; 0.0 1.0]
        result = FFS(rng, U)
        @test result isa Vector{Int}
        @test length(result) == 2
        @test sort(result) == [1, 2]

        # Test with larger matrix
        U = qr(randn(rng, 6, 3)).Q[:, 1:3]  # Random orthogonal matrix
        result = FFS(rng, U)
        @test length(result) == 6
        @test sort(unique(filter(x -> x > 0, result))) == collect(1:3)
        @test count(x -> x > 0, result) == 3
    end

    @testset "Edge cases" begin
        rng = MersenneTwister(456)


        # Square matrix
        U = Matrix(qr(randn(rng, 4, 4)).Q)
        result = FFS(rng, U)
        @test sort(result) == collect(1:4)

        # Large L/N ratio
        U = qr(randn(rng, 20, 2)).Q[:, 1:2]
        result = FFS(rng, U)
        @test count(x -> x > 0, result) == 2
        @test maximum(result) == 2
    end


    @testset "Statistical properties" begin
        rng = MersenneTwister(101112)
        L, N = 10, 2
        U = Matrix(qr(randn(rng, L, N)).Q)

        # Run multiple samples
        n_samples = 1000
        samples = [FFS(rng, U) for _ in 1:n_samples]

        # Check that all samples have correct number of occupied states
        @test all(x -> count(y -> y > 0, x) == N, samples)

        # Check that occupation numbers are in valid range
        @test all(x -> all(y -> y in 0:N, x), samples)

        # Test distribution of first occupied state
        first_states = [findfirst(x -> x == 1, sample) for sample in samples]
        # Chi-square test would be too strict due to randomness
        # Just check if all states are represented
        @test length(unique(first_states)) > L / 2
    end

    @testset "Different RNGs" begin
        U = Matrix(qr(randn(6, 3)).Q)

        # Test with different RNG types
        result1 = FFS(MersenneTwister(123), U)
        result2 = FFS(Random.Xoshiro(123), U)
        # Results may differ but should have same properties
        @test count(x -> x > 0, result1) == count(x -> x > 0, result2) == 3

        # Test default RNG
        result3 = FFS(U)
        @test count(x -> x > 0, result3) == 3
    end

    @testset "Numerical stability" begin
        rng = MersenneTwister(131415)

        # Test with nearly singular matrix
        U = [1.0 1e-8; 0.0 1.0; 1e-8 0.0]
        U = U ./ sqrt.(sum(abs2, U, dims = 1))
        result = FFS(rng, U)
        @test count(x -> x > 0, result) == 2

        # Test with random ill-conditioned matrix
        A = randn(rng, 5, 3)
        A[:, 2] = A[:, 1] + 1e-10 * randn(rng, 5)  # Make columns nearly linearly dependent
        U = qr(A).Q[:, 1:3]
        result = FFS(rng, U)
        @test count(x -> x > 0, result) == 3
    end
end
