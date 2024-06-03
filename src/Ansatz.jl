abstract type AbstractAnsatz end
struct Gutzwiller <: AbstractAnsatz
    conf_up::BitVector
    conf_down::BitVector
    g::Float64
    G::Float64 # may have precision problem!
    Og::Float64
    OL::Float64
end

"""
Fast Gutzwiller Factor update technique from Becca and Sorella 2017
This is not needed for one-g Gutzwiller Factor
"""
function fast_G_update(newconf::BitStr{N,T}, oldconf::BitStr{N,T}, g::Float64, n_mean::Float64) where {N,T}
    L = length(newconf) ÷ 2
    diff = newconf - oldconf
    @assert sum(abs.(diff)) == 2 "Only one electron move"
    ratio = exp(-g / 2 * sum(@. diff * ((1 - n_mean)^2) - n_mean^2))
    return ratio
end

"""
Fast computing technique from Becca and Sorella 2017
"""
function fast_update(
    U::AbstractMatrix,
    Uinvs::AbstractMatrix,
    newconf::BitStr{N,T},
    oldconf::BitStr{N,T}
) where {N,T}
    diff = newconf .⊻ oldconf
    Rl = -1
    k = -1
    flag = 0
    @inbounds for i in 1:N
        if readbit(diff, i) == 1
            if readbit(oldconf, i) == 1
                Rl = i # the old position of the l-th electron
                flag += 1
            end
            if readbit(newconf, i) == 1
                k = i # the new position of the l-th electron, K = R_l'
                flag += 1
            end
        end
        if flag == 2
            break
        end
    end
    l = sum(oldconf[1:Rl]) # l-th electron 
    ratio = sum(U[k, :] .* Uinvs[:, l])
    return ratio
end

"""
The observable OL = <x|H|Ψ_G>/<x|Ψ_G> 
"""
function getOL(orb::AHmodel, conf_up::BitVector, conf_down::BitVector)
    conf = LongBitStr(vcat(conf_up, conf_down))
    L = length(conf) ÷ 2
    OL = 0.0
    U_upinvs = inv(orb.U_up[conf_up, :]) # might be optimized by column slicing
    U_downinvs = inv(orb.U_down[conf_down, :])
    xprime = getxprime(orb, conf)
    @inbounds for confstr in keys(xprime)
        coff = xprime[confstr]
        if confstr == conf
            OL += coff
        elseif confstr[1:L] == conf_up
            OL +=
                coff *
                fast_update(orb.U_down, U_downinvs, LongBitStr(confstr[L+1:end]), LongBitStr(conf_down))
        elseif confstr[L+1:end] == conf_down
            OL +=
                coff *
                fast_update(orb.U_up, U_upinvs, LongBitStr(confstr[1:L]), LongBitStr(conf_up))
        else
            OL +=
                coff *
                fast_update(orb.U_up, U_upinvs, LongBitStr(confstr[1:L]), LongBitStr(conf_up)) *
                fast_update(orb.U_down, U_downinvs, LongBitStr(confstr[L+1:end]), LongBitStr(conf_down))
        end

    end
    return OL
end


"""
add Gutzwiller Ansatz where G  = exp(-g/2 ∑_i (n_i - n_mean)^2), Ψ_G = G Ψ_0
"""
function Gutzwiller(
    orbitals::AHmodel{B},
    conf_up::BitVector,
    conf_down::BitVector,
    g::Float64
) where {B}
    occupation = conf_up + conf_down
    n_mean = (orbitals.N_up + orbitals.N_down) / orbitals.lattice.ns
    Og = -1 / 2 * sum(@. (occupation - n_mean)^2)
    G = exp(g * Og)
    OL = getOL(orbitals, conf_up, conf_down)
    return Gutzwiller(conf_up, conf_down, g, G, Og, OL)
end
