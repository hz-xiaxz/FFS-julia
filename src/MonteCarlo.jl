"""
Use Carlo.jl to perform high efficiency
"""

mutable struct MC{B} <: AbstractMC where {B}
    model::AHmodel{B}
    conf::BitVector
end

seed = 42
rng = MersenneTwister(seed)

"""
    MC(params::AbstractDict)
------------
Create a Monte Carlo object
"""
function MC(params::AbstractDict)
    nx = params[:nx]
    ny = params[:ny]
    if params[:B] == "Periodic"
        B = Periodic()
    elseif params[:B] == "Open"
        B = Open()
    else
        throw(ArgumentError("Boundary condition not recognized"))
    end
    lat = LatticeRectangular(params[:nx], params[:ny], B)
    model = AHmodel(lat, params[:t], params[:W], params[:U], params[:N_up], params[:N_down])
    return MC{B}(model, BitVector(zeros(2 * nx * ny)))
end


"""
    Carlo.init!(mc::MC, ctx::MCContext, params::AbstractDict)
------------
Initialize the Monte Carlo object
`params`
* `nx` : `Int` number of sites in x direction
* `ny` : `Int` number of sites in y direction
* `B` : `AbstractBoundary` boundary condition, `Periodic` or `Open`
* `t` : `Float64` hopping parameter
* `W` : `Float64` disorder strength
* `U` : `Float64` on-site interaction strength
* `N_up` : `Int` number of up spins
* `N_down` : `Int` number of down spins
"""
function Carlo.init!(mc::MC{B}, ctx::MCContext, params::AbstractDict) where {B}
    lat = LatticeRectangular(params[:nx], params[:ny], B)
    orb = AHmodel(lat, params[:t], params[:W], params[:U], params[:N_up], params[:N_down])
    conf = vcat(FFS(ctx.rng, orb.U_up), FFS(ctx.rng, orb.U_down))
    mc.model = orb
    mc.conf = conf
end

function Carlo.sweep!(mc::MC{B}, ctx::MCContext) where {B}
    mc.conf = vcat(FFS(ctx.rng, mc.model.U_up), FFS(ctx.rng, mc.model.U_down))
    return nothing
end

function Carlo.measure!(mc::MC{B}, ctx::MCContext) where {B}
    conf_up = FFS(ctx.rng, mc.model.U_up)
    conf_down = FFS(ctx.rng, mc.model.U_down)
    g = 1.0 # temporarily fixed
    Gutz = Gutzwiller(mc.model, conf_up, conf_down, g)
    measure!(ctx, :OL, Gutz.OL)
    return nothing
end

function Carlo.register_evaluables(::Type{MC}, eval::Evaluator, params::AbstractDict)
    return nothing
end


function Carlo.write_checkpoint(mc::MC{B}, out::HDF5.Group) where {B}
    out["conf"] = Vector{Bool}(mc.conf)
    return nothing
end

function Carlo.read_checkpoint!(mc::MC{B}, in::HDF5.Group) where {B}
    mc.conf = BitVector(read(in, "conf"))
    return nothing
end
