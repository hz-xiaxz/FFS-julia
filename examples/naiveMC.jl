# Perform MC without Carlo.jl Framework
using Statistics
using FastFermionSampling

function run(MCsteps::Int)
    lat = LatticeRectangular(4, 4, Periodic())
    orb = AHmodel(lat, 1.0, 1.0, 1.0, 8, 8)
    sm = Vector{Float64}(undef, MCsteps)
    for i in 1:MCsteps
        conf_up = FFS(orb.U_up)
        conf_down = FFS(orb.U_down)
        gut = Gutzwiller(orb, conf_up, conf_down, 1.0)
        sm[i] = gut.OL
    end
    return mean(sm), (std(sm) / √(MCsteps - 1))
end