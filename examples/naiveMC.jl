# Perform MC without Carlo.jl Framework
using Statistics
using FastFermionSampling

include("../src/ED.jl")

function run(MCsteps::Int)
    lat = LatticeRectangular(3, 2, Periodic())
    orb = FastFermionSampling.fixedAHmodel(lat, 1.0, 1.0, 1.0, 3, 3)
    sm = zeros(Float64, MCsteps)
    for i in 1:MCsteps
        conf_up = FFS(orb.U_up)
        conf_down = FFS(orb.U_down)
        OL = getOL(orb, conf_up, conf_down, 1.0)
        # HACK!!!
        if i > 1
            if abs(OL) > abs(10 * mean(sm[1:i]))
                sm[i] = mean(sm[1:i])
            else
                sm[i] = OL
            end
        else
            sm[i] = OL
        end
    end
    result_ED = doED(3, 2, 1.0, 1.0, zeros(3 * 2), 'P')
    return mean(sm), (std(sm) / √(MCsteps - 1)), result_ED
end
