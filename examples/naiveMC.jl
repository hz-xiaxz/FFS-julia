# Perform MC without Carlo.jl Framework
using Statistics
using FastFermionSampling

function run(MCsteps::Int)
    lat = LatticeRectangular(4, 4, Periodic())
    orb = FastFermionSampling.fixedAHmodel(lat, 1.0, 1.0, 1.0, 8, 8)
    sm = zeros(Float64, MCsteps)
    for i in 1:MCsteps
        conf_up = FFS(orb.U_up)
        conf_down = FFS(orb.U_down)
        gut = Gutzwiller(orb, conf_up, conf_down, 1.0)
        # HACK!!!
        if i > 1 
            if abs(gut.OL) > abs(10 * mean(sm[1:i]))
                sm[i] = mean(sm[1:i])
            else
                sm[i] = gut.OL
            end
        else
            sm[i] = gut.OL
        end
    end
    return mean(sm), (std(sm) / √(MCsteps - 1))
end