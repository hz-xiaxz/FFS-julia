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

"""
    fast_G_update(newwholeconf::BitStr{N,T}, oldwholeconf::BitStr{N,T}, g::Float64, n_mean::Float64) where {N,T}

Fast Gutzwiller Factor update technique from Becca and Sorella 2017

Should input whole configuration 
"""
function fast_G_update(newwholeconf::BitStr{N, T}, oldwholeconf::BitStr{N, T},
        g::Float64, n_mean::Float64) where {N, T}
    # search for the electron that moves
    exponent = 0
    L = N ÷ 2
    flag = 0
    @inbounds for i in 1:L
        if readbit(newwholeconf, i) ⊻ readbit(oldwholeconf, i) == 1
            occup_new = Float64(readbit(newwholeconf, i) + readbit(newwholeconf, i + L))
            occup_old = Float64(readbit(oldwholeconf, i) + readbit(oldwholeconf, i + L))
            exponent += (occup_new - n_mean)^2 - (occup_old - n_mean)^2
            flag += 1
        end
        if flag == 2
            break
        end
    end
    ratio = exp(-g / 2 * exponent)
    return ratio
end

"""
    fast_update(U::AbstractMatrix, Uinvs::AbstractMatrix, newconf::BitStr{N,T}, oldconf::BitStr{N,T}) where {N,T}

Fast computing technique from Becca and Sorella 2017
"""
function fast_update(
        U::AbstractMatrix,
        Uinvs::AbstractMatrix,
        newconf::Union{SubDitStr{D, N1, T1}, DitStr{D, N1, T1}},
        oldconf::BitStr{N, T}
) where {D, N1, N, T, T1}
    @assert length(newconf)==N "The length of the new configuration should be the same as the old configuration, got: $(length(newconf))(old) and $N(new)"
    Rl = -1 # if not found should return error
    k = -1
    flag = 0
    @inbounds for i in 1:N
        if getindex(oldconf, i) == 1 && getindex(newconf, i) == 0
            Rl = i # the old position of the l-th electron
            flag += 1
        end
        if getindex(newconf, i) == 1 && getindex(oldconf, i) == 0
            k = i # the new position of the l-th electron, K = R_l'
            flag += 1
        end
        if flag == 2
            break
        end
    end
    l = sum(oldconf[1:Rl]) # l-th electron 
    ratio = sum(U[k, :] .* Uinvs[:, l])
    return ratio
end

@doc raw"""
    getOL(orb::AHmodel, conf_up::BitVector, conf_down::BitVector, g::Float64)

The observable ``O_L = \frac{<x|H|\psi_G>}{<x|\psi_G>}``
"""
function getOL(orb::AHmodel, conf_up::BitVector, conf_down::BitVector, g::Float64)
    conf = LongBitStr(vcat(conf_up, conf_down))
    L = length(conf) ÷ 2
    OL = 0.0
    n_mean = (orb.N_up + orb.N_down) / orb.lattice.ns
    U_upinvs = orb.U_up[conf_up, :] \ I # do invs more efficiently
    U_downinvs = orb.U_down[conf_down, :] \ I
    xprime = getxprime(orb, conf)
    @inbounds for (confstr, coff) in pairs(xprime)
        if confstr == conf
            OL += coff
        elseif confstr[1:L] == conf_up
            OL += coff *
                  fast_update(orb.U_down, U_downinvs, SubDitStr(confstr, L + 1, 2 * L),
                      LongBitStr(conf_down)) * fast_G_update(confstr, conf, g, n_mean)
        elseif confstr[(L + 1):end] == conf_down
            OL += coff *
                  fast_update(
                      orb.U_up, U_upinvs, SubDitStr(confstr, 1, L), LongBitStr(conf_up)) *
                  fast_G_update(confstr, conf, g, n_mean)
        else
            OL += coff *
                  fast_update(
                      orb.U_up, U_upinvs, SubDitStr(confstr, 1, L), LongBitStr(conf_up)) *
                  fast_update(orb.U_down, U_downinvs, SubDitStr(confstr, L + 1, 2 * L),
                      LongBitStr(conf_down)) * fast_G_update(confstr, conf, g, n_mean)
        end
    end
    return OL
end

@doc raw"""
    getOg(orbitals::AHmodel{B}, conf_up::BitVector, conf_down::BitVector)

The local operator to update the variational parameter `g`
``mathcal{O}_k(x)=\frac{\partial \ln \Psi_\alpha(x)}{\partial \alpha_k}``
"""
function getOg(
        orbitals::AHmodel{B},
        conf_up::BitVector,
        conf_down::BitVector
) where {B}
    occupation = conf_up + conf_down
    n_mean = (orbitals.N_up + orbitals.N_down) / orbitals.lattice.ns
    Og = -1 / 2 * sum(@. (occupation - n_mean)^2)
    return Og
end
