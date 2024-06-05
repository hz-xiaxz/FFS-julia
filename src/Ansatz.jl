abstract type AbstractAnsatz end
struct Gutzwiller <: AbstractAnsatz
    g::Float64
    Og::Float64
    OL::Float64
end

"""
Fast Gutzwiller Factor update technique from Becca and Sorella 2017
Should input whole conf
"""
function fast_G_update(newwholeconf::BitStr{N,T}, oldwholeconf::BitStr{N,T}, g::Float64, n_mean::Float64) where {N,T}
    # search for the electron that moves
    diff = newwholeconf .⊻ oldwholeconf # this is more efficient
    exponent = 0
    L = N ÷ 2
    flag = 0
    @inbounds for i in 1:L
        if readbit(diff, i) == 1
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
Fast computing technique from Becca and Sorella 2017
"""
function fast_update(
    U::AbstractMatrix,
    Uinvs::AbstractMatrix,
    newconf::BitStr{N,T},
    oldconf::BitStr{N,T}
) where {N,T}
    diff = newconf .⊻ oldconf
    Rl = -1 # if not found should return error
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
function getOL(orb::AHmodel, conf_up::BitVector, conf_down::BitVector, g::Float64)
    conf = LongBitStr(vcat(conf_up, conf_down))
    L = length(conf) ÷ 2
    OL = 0.0
    n_mean = (orb.N_up + orb.N_down) / orb.lattice.ns
    U_upinvs = inv(orb.U_up[conf_up, :]) # might be optimized by column slicing
    U_downinvs = inv(orb.U_down[conf_down, :])
    xprime = getxprime(orb, conf)
    @inbounds for (confstr, coff) in pairs(xprime)
        if confstr == conf
            OL += coff
        elseif confstr[1:L] == conf_up
            OL +=
                coff *
                fast_update(orb.U_down, U_downinvs, LongBitStr(confstr[L+1:end]), LongBitStr(conf_down)) * fast_G_update(confstr, conf, g, n_mean)
        elseif confstr[L+1:end] == conf_down
            OL +=
                coff *
                fast_update(orb.U_up, U_upinvs, LongBitStr(confstr[1:L]), LongBitStr(conf_up)) * fast_G_update(confstr, conf, g, n_mean)
        else
            OL +=
                coff *
                fast_update(orb.U_up, U_upinvs, LongBitStr(confstr[1:L]), LongBitStr(conf_up)) *
                fast_update(orb.U_down, U_downinvs, LongBitStr(confstr[L+1:end]), LongBitStr(conf_down)) * fast_G_update(confstr, conf, g, n_mean)
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
    OL = getOL(orbitals, conf_up, conf_down, g)
    return Gutzwiller(g, Og, OL)
end
