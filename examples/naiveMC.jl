# Perform MC without Carlo.jl Framework
using Statistics
using FastFermionSampling
using CairoMakie

include("../src/ED.jl")

function run(MCsteps::Int, Lx::Int, Ly::Int)
    lat = LatticeRectangular(Lx, Ly, Open())
    N = Lx * Ly ÷ 2 # number of particles
    orb = FastFermionSampling.fixedAHmodel(lat, 1.0, 1.0, 1.0, N, N)
    sm = zeros(Float64, MCsteps)
    for i in 1:MCsteps
        conf_up = FFS(orb.U_up)
        conf_down = FFS(orb.U_down)
        OL = FastFermionSampling.getOL(orb, conf_up, conf_down, 0.3)
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
    fig = Figure()
    ax = Axis(fig[1, 1], title = "sampled local energy")
    CairoMakie.scatter!(ax, sm)
    display(fig)
    result_ED = doED(Lx, Ly, 1.0, 1.0, zeros(Lx * Ly), 'O')
    return mean(sm), (std(sm) / √(MCsteps - 1)), result_ED
end
