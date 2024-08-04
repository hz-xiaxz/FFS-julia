using Random
using LinearAlgebra
using StatsBase
using BitBasis

"""
    FFS(U::AbstractMatrix)
    FFS([rng=default_rng()], U::AbstractMatrix)
    
Employing Fast Fermion Sampling Algorithm to sample free Fermions

`U` : the sampling ensemble, a matrix of size `L` x `N`, where `L` is the number of energy states and `N` is the number of Fermions 
"""
function FFS(r::AbstractRNG, u::AbstractMatrix)
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
    return sampled
end

FFS(u::AbstractMatrix) = FFS(Random.default_rng(), u)

# add wrapper structs to define Bool-vector like to_index function for BitStr
struct AvailConf{D, N, T <: Integer}
    ref::Base.RefValue{DitStr{D, N, T}}
end

struct SampledConf{D, N, T <: Integer}
    ref::Base.RefValue{DitStr{D, N, T}}
end

function Base.to_index(d::SampledConf{D, N, T}) where {D, N, T}
    return Base.LogicalIndex(map(i -> readbit(d.ref[], i) == 1, 1:N))
end


function Base.to_index(d::AvailConf{D, N, T}) where {D, N, T}
    return Base.LogicalIndex(map(i -> readbit(d.ref[], i) == 0, 1:N))
end


function newFFS(r::AbstractRNG, u::AbstractMatrix)
    L, N = size(u)
    v = randperm(r, N)
    U = u[:, v]
    bstr_type = L <= 64 ? Int64 : BitBasis.LongLongUInt{L ÷ 64 + 1}
    conf = BitStr{L}(0)
    groud_set = collect(1:L)
    # For x1 case, P(x1;m) = |U_{x1, m1}|^2 
    p = abs2.(U[:, 1])
    x_new = sample(r, 1:L, Weights(p))
    conf |= indicator(bstr_type, x_new)
    n_vec = normalize([-U[x_new, 2] / U[x_new, 1], 1])
    @inbounds for i in 2:(N - 1)
        avail = AvailConf(Ref(conf))
        prob = abs2.((view(U, :, 1:i) * n_vec)[avail])
        x_new = sample(r, groud_set[avail], Weights(prob))
        conf |= indicator(bstr_type, x_new)
        # now compute next n_vec
        # I suggest not using the gaussian elimination
        U_x = U[SampledConf(Ref(conf)), 1:i]
        B = -U[1:i, i + 1]
        n_vec = normalize([U_x \ B; 1])
    end
    avail = AvailConf(Ref(conf))
    prob = abs2.((view(U, :, 1:N) * n_vec)[avail])
    x_new = sample(r, groud_set[avail], Weights(prob))
    conf |= indicator(bstr_type, x_new)
    return conf
end

newFFS(u::AbstractMatrix) = newFFS(Random.default_rng(), u)