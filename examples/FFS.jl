using FastFermionSampling
using StatsBase
using LinearAlgebra
using CairoMakie
using Random
using Combinatorics

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

@inline function FFSwithv(r::AbstractRNG, u::AbstractMatrix)
    L, N = size(u)
    v = randperm(r, N)
    U = u[:, v]
    sampled = falses(L)
    avail = trues(L)
    groud_set = collect(1:L)
    # For x1 case, P(x1;m) = |U_{x1, m1}|^2
    p = abs2.(U[:, 1])
    x_new = sample(r, 1:L, Weights(p))
    sampled[x_new] = true
    avail[x_new] = false
    n_vec = normalize([-U[x_new, 2] / U[x_new, 1], 1])
    @inbounds for i in 2:(N - 1)
        prob = abs2.((view(U, :, 1:i) * n_vec)[avail])
        x_new = sample(r, groud_set[avail], Weights(prob))
        sampled[x_new] = true
        avail[x_new] = false
        # now compute next n_vec
        # I suggest not using the gaussian elimination
        U_x = U[sampled, 1:i]
        B = -U[1:i, i + 1]
        n_vec = normalize([U_x \ B; 1])
    end
    prob = abs2.((view(U, :, 1:N) * n_vec)[avail])
    x_new = sample(r, groud_set[avail], Weights(prob))
    sampled[x_new] = true
    avail[x_new] = false
    return sampled, v
end

FFSwithv(u::AbstractMatrix) = FFSwithv(Random.default_rng(), u)

function get_samples(U::AbstractMatrix, itertime::Int)
    L, N = size(U)
    events, p = exact_sample(U, L, N)

    events_len = length(events)
    events_int = zeros(Int, events_len)
    for (i, ev) in enumerate(events)
        events_int[i] = evalpoly(2, reverse(ev))
    end

    # sort according to the magnitude of exact probability
    indices = sortperm(p, rev = true)

    # give a one-to-one map from int representation to numbers in 1:length(event_int)
    mapping = Dict{Int, Int}()
    for (i, ev) in enumerate(events_int[indices])
        mapping[ev] = i
    end


    sampled = zeros(Int, itertime)
    sampledv = zeros(Int, itertime)
    perms = collect(Combinatorics.permutations(1:N))

    for i in 1:itertime
        conf, v = FFSwithv(U)
        sampled[i] = evalpoly(2, reverse(conf))
        sampledv[i] = findfirst(x -> x == v, perms)
    end

    for (i, ev) in enumerate(sampled)
        sampled[i] = mapping[ev] # sort according to the magnitude of exact probability
    end
    p_sorted = p[indices]
    return p_sorted, sampled, sampledv

end

function plot_dist(U::AbstractMatrix, itertime::Int)
    p_sorted, sampled, sampledv = get_samples(U, itertime)
    fig = Figure()

    ax = Axis(
        fig[1, 1], title = "Exact_sample vs FFS", xlabel = "Event", ylabel = "Probability")
    events_len = length(p_sorted)
    CairoMakie.scatter!(
        ax, collect(1:events_len), p_sorted, label = "Exact_sample")
    h = StatsBase.fit(Histogram, sampled, 0:1:events_len, closed = :right)
    stats_weight = h.weights / sum(h.weights)
    CairoMakie.scatter!(
        ax, collect(1:events_len), stats_weight, color = :red, label = "FFS")
    CairoMakie.axislegend(ax)
    # ax2 = Axis(fig[2, 1], title = "Sampled v", xlabel = "Event", ylabel = "Probability")
    # CairoMakie.hist!(ax2, sampledv, 0:1:length(sampledv))

    display(fig)
    dir = @__DIR__
    save(dir * "/../tmp/benchmark.png", fig)
end

function plot_kldiv(U::AbstractMatrix, rg::AbstractArray)
    fig = Figure()
    ax = Axis(fig[1, 1], title = "KL divergence",
        xlabel = "Iteration time", ylabel = "KL divergence")
    kl_list = zeros(Float64, length(rg))
    for (i, itertime) in enumerate(rg)
        p_sorted, sampled, sampledv = get_samples(U, itertime)
        h = StatsBase.fit(Histogram, sampled, 0:1:length(p_sorted), closed = :right)
        stats_weight = h.weights / sum(h.weights)
        stats_weight .+= 1e-16
        kl = sum(p_sorted .* log.(p_sorted ./ stats_weight))
        kl_list[i] = kl
    end
    CairoMakie.scatter!(ax, rg, kl_list)
    display(fig)
    dir = @__DIR__
    save(dir * "/../tmp/kldiv.png", fig)
end


U = [-0.47504745632183526 2.23947392133299 0.4190104799416631 1.8820429499802853;
     1.647575359352309 0.24764632503319334 -1.0528915129751741 -0.33332029137330343;
     0.7508437890055393 -1.1309312593011076 1.902302092461544 -1.0136818954113225;
     -0.13586133354519203 -0.49111844419196826 -1.0344505355906626 0.3353846930939702;
     0.5938815426408376 -0.80105142051938 0.6979534466386051 -1.4585065885760446;
     -1.017459720726235 -0.41981150212424656 -0.054374656648999854 0.9726909811845045;
     0.3775650241606924 -0.10886801546615807 0.7334751804198373 -0.5039965163462246;
     0.3530127019022432 1.4905629511002982 -0.3742556953304839 0.8717745800110613;
     0.4732450750273541 -0.6547178130978334 -1.0025877479776981 1.860367010307612;
     -0.10791156099645734 0.49540118381450726 1.1797749148252705 -0.044210142753559944]
plot_dist(U, 1000)
plot_kldiv(U, 1000:100:10000)
