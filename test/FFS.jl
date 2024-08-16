using FastFermionSampling
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
    rng = MersenneTwister(123)
    U = randn(Float64, 10, 2)
    @test FastFermionSampling.FFS(rng, U) isa BitVector
    kl1 = getKLdiv(10, 2, 500, U)
    @test kl1≈0 atol=5 # high tolerance for now
    kl2 = getKLdiv(10, 2, 5000, U)
    @test kl2≈0 atol=1
    @test kl2 < kl1
end
