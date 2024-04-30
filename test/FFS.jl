using FastFermionSampling
using StatsBase
using LinearAlgebra
using CairoMakie

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
    return sort(array, by=x -> evalpoly(2, reverse(x)))
end

function exact_sample(U::AbstractMatrix, L::Int, N::Int)

    events = sort_bool_arrays(combinations(L, N))
    p = [abs2(det(U[ev, :])) for ev in events]
    normalize!(p)
    return (events, p)

end

function visualize(U::AbstractMatrix, L::Int, N::Int, itertime::Int)
    fig = Figure()

    ax = Axis(fig[1, 1], title="Exact_sample", xlabel="Event", ylabel="Probability")

    events, p = exact_sample(U, L, N)

    events_int = zeros(Int, length(events))
    for (i, ev) in enumerate(events)
        events_int[i] = evalpoly(2, reverse(ev))
    end

    CairoMakie.scatter!(ax, events_int, p)

    sampled = zeros(Int, itertime)

    for i in 1:itertime
        sampled[i] = evalpoly(2, reverse(FastFermionSampling.FFS(U, L, N)))
    end

    ax2 = Axis(fig[1, 2], title="FFS", xlabel="Event", ylabel="Frequency")
    CairoMakie.hist!(ax2, sampled, bins=400, normalization=:pdf, color=:red)

    fig
    save("./tmp/benchmark.png", fig)
end