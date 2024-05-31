abstract type AbstractAnsatz end
struct Gutzwiller <: AbstractAnsatz
    conf_up::BitVector
    conf_down::BitVector
    g::Float64
    G::Float64
    Og::Float64
    OL::Float64
end

"""
Fast Gutzwiller Factor update technique from Becca and Sorella 2017
"""
function fast_G_update(newconf::BitVector, oldconf::BitVector, g::Float64, n_mean::Float64)
    L = length(newconf) ÷ 2
    occup = newconf[1:L] + newconf[L+1:end]
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
    newconf::BitVector,
    oldconf::BitVector,
)
    diff = newconf - oldconf
    Rl = findfirst(==(-1), diff) # the old position of the l-th electron
    l = sum(oldconf[1:Rl]) # l-th electron 
    k = findfirst(==(1), diff) # the new position of the l-th electron K = R_l'
    ratio = sum(U[k, :] .* Uinvs[:, l])
    return ratio
end

"""
The observable OL = <x|H|Ψ_G>/<x|Ψ_G> 
"""
function getOL(orb::AHmodel, conf_up::BitVector, conf_down::BitVector, g)
    conf = vcat(conf_up, conf_down)
    L = length(conf) ÷ 2
    n_mean = (orb.N_up + orb.N_down) / orb.lattice.ns
    OL = 0.0
    U_upinvs = inv(orb.U_up[conf_up, :]) # might be optimized by column slicing
    U_downinvs = inv(orb.U_down[conf_down, :])
    xprime = getxprime(orb, LongBitStr(conf))
    for confstr in keys(xprime)
        new_conf = BitVector([confstr...])
        coff = xprime[confstr]
        if new_conf == conf
            OL += coff
        elseif new_conf[1:L] == conf_up
            OL +=
                coff *
                fast_update(orb.U_down, U_downinvs, new_conf[L+1:end], conf_down) *
                fast_G_update(new_conf[L+1:end], conf_down, g, n_mean)
        elseif new_conf[L+1:end] == conf_down
            OL +=
                coff *
                fast_update(orb.U_up, U_upinvs, new_conf[1:L], conf_up) *
                fast_G_update(new_conf[1:L], conf_up, g, n_mean)
        else
            OL +=
                coff *
                fast_update(orb.U_up, U_upinvs, new_conf[1:L], conf_up) *
                fast_update(orb.U_down, U_downinvs, new_conf[L+1:end], conf_down) *
                fast_G_update(new_conf[1:L], conf_up, g, n_mean) *
                fast_G_update(new_conf[L+1:end], conf_down, g, n_mean)
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
    g::Float64,
) where {B}
    occupation = conf_up + conf_down
    n_mean = (orbitals.N_up + orbitals.N_down) / orbitals.lattice.ns
    Og = -1 / 2 * sum(@. (occupation - n_mean)^2)
    G = exp(g * Og)
    OL = getOL(orbitals, conf_up, conf_down, g)
    return Gutzwiller(conf_up, conf_down, g, G, Og, OL)
end
