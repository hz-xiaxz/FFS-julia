"""
Use Carlo.jl to perform high efficiency
"""

mutable struct MC{B} <: AbstractMC where {B}
    model::AHmodel{B}
    conf::BitVector
    g::Float64
    OLbench::Float64
end

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
    g = params[:g]
    OLbench = getOL(model, FFS(model.U_up), FFS(model.U_down), g)
    return MC{B}(model, BitVector(zeros(2 * nx * ny)), g, OLbench)
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
    conf_up = FFS(ctx.rng, orb.U_up)
    conf_down = FFS(ctx.rng, orb.U_down)
    conf = vcat(conf_up, conf_down)
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
    OL = getOL(mc.model, conf_up, conf_down, mc.g)
    Og = getOg(mc.model, conf_up, conf_down)
    if abs(OL) > 10*abs(mc.OLbench)
        OL = mc.OLbench
    end
    measure!(ctx, :OL, OL)
    measure!(ctx, :Og, Og)
    measure!(ctx, :OLOg, OL * Og)
    measure!(ctx, :Og2, Og^2)
    return nothing
end

@doc raw"""
     **fg** the gradient of ⟨E_g⟩
       

Get the gradient of the observable, ``f_g = - ∂ ⟨E_g⟩/ ∂ g``.
``f_k = -2 ℜ[⟨O_L(x)^* × (O_g(x)- ⟨O_g⟩) ⟩ ] = -2 ℜ[⟨O_L(x)^* × O_g(x) ⟩ - ⟨O_L(x)^* ⟩ × ⟨O_g⟩  ]``

    **fisherScalar**

Get the Fisher Matrix of the observable, ``S_{k,k'}  = ℜ⟨⟨O_k O_{k'}⟩⟩ = ℜ( ⟨O_k O_{k'}⟩ -⟨O_k⟩ ⟨O_{k'}⟩ ) `` where ``k`` and ``k'`` are labels of the parameters of the model.

When ansatz has only one parameter, the Fisher Matrix is a scalar, and the Fisher Information is the inverse of the Fisher Matrix.
"""
function Carlo.register_evaluables(::Type{MC}, eval::Evaluator, params::AbstractDict)

    evaluate!(eval, :fg, (:OL, :Og, :OLOg)) do OL, Og, OLOg
        @assert isa(OL, Real) "OL should be a real number, got $OL"
        return -2 * real(OLOg - OL * Og)
    end
    evaluate!(eval, :fisherScalar, (:Og, :Og2)) do Og, Og2
        return Og2 - Og^2
    end
    return nothing
end

function Carlo.write_checkpoint(mc::MC{B}, out::HDF5.Group) where {B}
    out["conf"] = Vector{Bool}(mc.conf)
    out["OLbench"] = mc.OLbench
    return nothing
end

function Carlo.read_checkpoint!(mc::MC{B}, in::HDF5.Group) where {B}
    mc.conf = BitVector(read(in, "conf"))
    out["OLbench"] = read(in, "OLbench")
    return nothing
end
