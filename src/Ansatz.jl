abstract type AbstractAnsatz end

@doc raw"""
Gutzwiller Ansatz
-----------------
* `g` : `Float64` Gutzwiller factor

Store the Gutzwiller factor parameter `g`.

The Gutzwiller factor is defined as ``G = \exp(-g/2 \sum_i (n_i - n_{mean})^2)``.
"""
struct Gutzwiller <: AbstractAnsatz
    g::Float64
end

@doc raw"""
    fast_G_update(κup::Vector{Int}, κdown::Vector{Int}, g::Float64; K::Int, l::Int, spin::Spin) -> Float64

Compute the Gutzwiller factor ratio for a single electron hop.

The Gutzwiller factor ratio for an electron hopping from site `R_l` to site `K(=R_l'` is:
```math
\begin{aligned}
\frac{G_{new}}{G_{old}} &= \exp(-\frac{g}{2}[(n_K+1 - n_{mean})^2 - (n_K-n_{mean})^2 + (n_{R_l}-1-n_{mean})^2 - (n_{R_l}-n_{mean})^2]) \\
 &= \exp(-\frac{g}{2}[(2n_K + 1 - 2n_{mean}) - (2n_{R_l}-1-2n_{mean})]) \\
 &= \exp(-g[n_K - n_{R_l} + 1])
\end{aligned}
```

# Arguments
- `κup::Vector{Int}`: Current configuration of up spins
- `κdown::Vector{Int}`: Current configuration of down spins
- `g::Float64`: Gutzwiller parameter
- `n_mean::Float64`: Mean occupation number
- `K::Int`: Target site index for the hop
- `l::Int`: State label to be moved
- `spin::Spin`: Spin species of the hopping electron

# Returns
- `ratio::Float64`: The ratio of new to old Gutzwiller factors

# Throws
- `ArgumentError`: If indices are invalid or state not found
"""
@inline function fast_G_update(
        κup::Vector{Int},
        κdown::Vector{Int},
        g::Float64;
        K::Int,
        l::Int,
        spin::Spin
)
    # Input validation
    num_sites = length(κup)
    @assert length(κdown)==num_sites "Spin configurations must have same length"
    @assert 1<=K<=num_sites "Target site index out of bounds"
    # Find source site
    Rl = if spin == Up
        findfirst(==(l), κup)
    else
        findfirst(==(l), κdown)
    end
    Rl == K && return 1.0
    isnothing(Rl) && throw(ArgumentError("State $l not found in configuration"))

    # Calculate current occupations
    n_K = site_occupation(κup[K]) + site_occupation(κdown[K])
    n_Rl = site_occupation(κup[Rl]) + site_occupation(κdown[Rl])

    # Final simplified exponent calculation
    return exp(-g * (n_K - n_Rl + 1))
end


"""Calculate electron occupation (0 or 1) at a site."""
@inline function site_occupation(κ::Int)::Int
    return κ != 0 ? 1 : 0
end

# """
#     fast_update(U::AbstractMatrix, Uinvs::AbstractMatrix, newconf::BitStr{N,T}, oldconf::BitStr{N,T}) where {N,T}

# Fast computing technique from Becca and Sorella 2017
# """
# @inline function fast_update(
#         U::AbstractMatrix,
#         Uinvs::AbstractMatrix,
#         newconf::Union{SubDitStr{D, N1, T1}, DitStr{D, N1, T1}},
#         oldconf::BitStr{N, T}
# ) where {D, N1, N, T, T1}
#     @assert length(newconf)==N "The length of the new configuration should be the same as the old configuration, got: $(length(newconf))(old) and $N(new)"
#     Rl = -1 # if not found should return error
#     k = -1
#     flag = 0
#     @inbounds for i in 1:N
#         if getindex(oldconf, i) == 1 && getindex(newconf, i) == 0
#             Rl = i # the old position of the l-th electron
#             flag += 1
#         end
#         if getindex(newconf, i) == 1 && getindex(oldconf, i) == 0
#             k = i # the new position of the l-th electron, K = R_l'
#             flag += 1
#         end
#         if flag == 2
#             break
#         end
#     end
#     if flag == 0
#         return 1.0
#     end
#     l = sum(oldconf[1:Rl]) # l-th electron
#     ratio = sum(U[k, :] .* Uinvs[:, l])
#     return ratio
# end

@enum TermType Diagonal UpHop DownHop

@doc raw"""
    getOL(orb::AHmodel, κup::Vector{Int}, κdown::Vector{Int}, g::Float64,
          Wup::AbstractMatrix{Float64}, Wdown::AbstractMatrix{Float64}) -> Float64

Compute the local energy ``O_L = \frac{⟨x|H|\psi_G⟩}{⟨x|\psi_G⟩}``.

This includes:
1. Diagonal terms (on-site energy and Hubbard interaction)
2. Up-spin hopping terms
3. Down-spin hopping terms

# Arguments
- `orb::AHmodel`: The Anderson-Hubbard model
- `κup::Vector{Int}`: Current up-spin configuration
- `κdown::Vector{Int}`: Current down-spin configuration
- `g::Float64`: Gutzwiller parameter
- `Wup::AbstractMatrix{Float64}`: Up-spin Slater determinant ratios
- `Wdown::AbstractMatrix{Float64}`: Down-spin Slater determinant ratios

# Returns
- `OL::Float64`: The local energy
"""
@inline function getOL(
        orb::AHmodel{B},
        κup::Vector{Int},
        κdown::Vector{Int},
        g::Float64,
        Wup::AbstractMatrix{Float64},
        Wdown::AbstractMatrix{Float64}
) where {B}
    # Get matrix elements of H
    xprime = getxprime(orb, κup, κdown)

    # Initialize local energy
    OL = 0.0

    # Process each term in the Hamiltonian
    @inbounds for (key, coeff) in pairs(xprime)
        term_type = classify_term(key)

        OL += compute_contribution(
            term_type,
            coeff,
            key,
            κup,
            κdown,
            g,
            Wup,
            Wdown
        )
    end

    return OL
end

"""Classify the type of Hamiltonian term based on its key."""
@inline function classify_term(key::Tuple{Int, Int, Int, Int})::TermType
    if all(==(DIAGONAL_INDEX), key)
        return Diagonal
    elseif key[1] == DIAGONAL_INDEX && key[2] == DIAGONAL_INDEX
        return DownHop
    else
        return UpHop
    end
end

"""Compute contribution to local energy from a specific term."""
@inline function compute_contribution(
        term_type::TermType,
        coeff::Float64,
        key::Tuple{Int, Int, Int, Int},
        κup::Vector{Int},
        κdown::Vector{Int},
        g::Float64,
        Wup::AbstractMatrix{Float64},
        Wdown::AbstractMatrix{Float64}
)::Float64
    if term_type == Diagonal
        return coeff
    elseif term_type == DownHop
        K, l = key[3], key[4]
        return coeff * Wup[K, l] * fast_G_update(κup, κdown, g; K = K, l = l, spin = Down)
    else  # UpHop
        K, l = key[1], key[2]
        return coeff * Wdown[K, l] * fast_G_update(κup, κdown, g; K = K, l = l, spin = Up)
    end
end

# Constants
const DIAGONAL_INDEX = -1

@doc raw"""
    getOg(orbitals::AHmodel{B}, conf_up::BitVector, conf_down::BitVector)

The local operator to update the variational parameter `g`
``mathcal{O}_k(x)=\frac{\partial \ln \Psi_\alpha(x)}{\partial \alpha_k}``
"""
@inline function getOg(
        orbitals::AHmodel{B},
        conf_up::BitVector,
        conf_down::BitVector
) where {B}
    occupation = conf_up + conf_down
    n_mean = (orbitals.N_up + orbitals.N_down) / orbitals.lattice.ns
    Og = -1 / 2 * sum(@. (occupation - n_mean)^2)
    return Og
end
