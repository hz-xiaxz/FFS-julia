"""
Use Carlo.jl to perform high efficiency
"""

mutable struct MC <: AbstractMC
    model::AHmodel
    conf::BitVector
end

seed = 42
rng = MersenneTwister(seed)

function Carlo.init!(mc::MC, ctx::MCContext)
    mc.conf = hcat(FFS(ctx.rng, mc.model.U_up), FFS(ctx.rng, mc.model.U_down))
    return nothing
end

function Carlo.sweep!(mc::MC, ctx::MCContext)
    mc.conf = hcat(FFS(ctx.rng, mc.model.U_up), FFS(ctx.rng, mc.model.U_down))
    return nothing
end

function Carlo.measure!(mc::MC, ctx::MCContext)
    conf_up = FFS(ctx.rng, mc.model.U_up)
    conf_down = FFS(ctx.rng, mc.model.U_down)
    g = mc.model.g
    Gutz = Gutzwiller(mc.model, conf_up, conf_down, g)
    measure!(ctx, :OL, Gutz.OL)
    return nothing
end
