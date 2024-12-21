"""
Use Carlo.jl to perform high efficiency
"""

mutable struct MC{B} <: AbstractMC where {B}
    model::AHmodel{B}
    κup::Vector{Int}
    κdown::Vector{Int}
    W_up::Matrix{Float64}
    W_down::Matrix{Float64}
    g::Float64
    q::Vector{Float64}
end

"""
    tilde_U(U::AbstractMatrix, kappa::Vector{Int})
------------
Creates a tilde matrix by rearranging rows of U according to kappa indices.

Parameters:
- `U`: Source matrix of size (n × m)
- `kappa`: Vector of indices where each non-zero value l indicates that row Rl of U
          should be placed at row l of the output

Returns:
- A matrix of size (m × m) with same element type as U
"""
function tilde_U(U::AbstractMatrix, κ::Vector{Int})
    m = size(U, 2)
    n = size(U, 1)
    # check if kappa is valid
    length(κ) == n || throw(
        DimensionMismatch(
        "Length of kappa ($(length(κ))) must match number of rows in U ($n)",
    ),
    )
    length(filter(x -> x != 0, κ)) == m ||
        throw(ArgumentError("kappa ($κ) is not valid"))

    # Create output matrix with same element type as U and requested size
    tilde_U = zeros(eltype(U), m, m)

    @inbounds for (Rl, l) in enumerate(κ)
        if l != 0
            (1 ≤ l ≤ m) || throw(BoundsError(tilde_U, (l, :)))
            tilde_U[l, :] = U[Rl, :]
        end
    end

    return tilde_U
end

"""
    is_occupied(kappa::Vector{Int}, l::Int) -> Bool

Check if site `l` is occupied in the kappa configuration vector.

Throws:
    BoundsError: if l is outside the valid range of kappa
"""
@inline function is_occupied(kappa::Vector{Int}, l::Int)
    @boundscheck 1 ≤ l ≤ length(kappa) || throw(BoundsError(kappa, l))
    @inbounds return !iszero(kappa[l])
end

function reevaluateW!(mc::MC{B}) where {B}
    tilde_U_up = tilde_U(mc.model.U_up, mc.κup)
    tilde_U_down = tilde_U(mc.model.U_down, mc.κdown)
    U_upinvs = tilde_U_up \ I
    U_downinvs = tilde_U_down \ I
    mc.W_up = mc.model.U_up * U_upinvs
    mc.W_down = mc.model.U_down * U_downinvs
    return nothing
end

# Add max retries constant
const MAX_INVERSION_RETRIES = 10
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
    q = [2 * pi / nx, 2 * pi / ny]
    κup = zeros(Int, lat.ns)
    κdown = zeros(Int, lat.ns)
    W_up = zeros(Float64, lat.ns, lat.ns)
    W_down = zeros(Float64, lat.ns, lat.ns)
    return MC{B}(model, κup, κdown, W_up, W_down, g, q)
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
@inline function Carlo.init!(mc::MC{B}, ctx::MCContext, params::AbstractDict) where {B}
    lat = LatticeRectangular(params[:nx], params[:ny], B)
    mc.model = AHmodel(
        lat, params[:t], params[:W], params[:U], params[:N_up], params[:N_down])
    mc.g = params[:g]
    mc.q = [2 * pi / nx, 2 * pi / ny]
    mc.κup = FFS(mc.model.U_up)
    mc.κdown = FFS(mc.model.U_down)
    tilde_U_up = tilde_U(mc.model.U_up, mc.κup)
    tilde_U_down = tilde_U(mc.model.U_down, mc.κdown)
    for attempt in 1:MAX_INVERSION_RETRIES
        try
            U_upinvs = tilde_U_up \ I
            U_downinvs = tilde_U_down \ I
            break  # Success - continue with these values
        catch e
            if e isa SingularException || e isa LinearAlgebra.LAPACKException
                if attempt == MAX_INVERSION_RETRIES
                    error("Matrix inversion failed after $MAX_INVERSION_RETRIES attempts. Please check configuration stability.")
                end
                # Regenerate configurations and update MC state
                κup, κdown = FFS(model.U_up), FFS(model.U_down)
                tilde_U_up = tilde_U(model.U_up, κup)
                tilde_U_down = tilde_U(model.U_down, κdown)
                continue
            else
                rethrow(e)  # If it's not a matrix inversion error,
            end
        end
    end
    mc.W_up = mc.model.U_up * U_upinvs
    mc.W_down = mc.model.U_down * U_downinvs
    return nothing
end

@inline function Carlo.sweep!(mc::MC{B}, ctx::MCContext) where {B}
    mc.κup, mc.κdown = FFS(mc.model.U_up), FFS(mc.model.U_down)
    reevaluateW!(mc)
    return nothing
end

@inline function Carlo.measure!(mc::MC{B}, ctx::MCContext) where {B}
    OL = getOL(mc.model, mc.κup, mc.κdown, mc.g, mc.W_up, mc.W_down)
    Og = getOg(mc.model, mc.κup, mc.κdown)
    G = exp(mc.g * Og)
    if abs(OL) > 10 * abs(mc.OLbench)
        OL = mc.OLbench
    end
    measure!(ctx, :OL, OL)
    measure!(ctx, :Og, Og)
    measure!(ctx, :OLOg, OL * Og)
    measure!(ctx, :Og2, Og^2)

    measure!(ctx, :G2, G^2)

    # sampling N_q
    nx = mc.model.lattice.nx
    ny = mc.model.lattice.ny
    occ_2d = reshape(site_occupation(mc.κup) + site_occupation(mc.κdown), nx, ny)
    nq = zero(ComplexF64)
    @inbounds for i in 1:nx
        @inbounds for j in 1:ny
            nq += occ_2d[i, j] * exp(im * (mc.q[1] * i + mc.q[2] * j))
        end
    end

    measure!(ctx, :G2nqnmq, G^2 * abs2(nq))
    measure!(ctx, :G2nq, G^2 * nq)

    return nothing
end

@doc raw"""

``fg`` the gradient of ⟨E_g⟩
------------

Get the gradient of the observable, ``f_g = - ∂ ⟨E_g⟩/ ∂ g``.

``math
\begin{align}
f_k &= -2 ℜ[⟨O_L(x)^* × (O_g(x)- ⟨O_g⟩) ⟩ ]\\
 &= -2 ℜ[⟨O_L(x)^* × O_g(x) ⟩ - ⟨O_L(x)^* ⟩ × ⟨O_g⟩  ]\\
\end{align}
``

Fisher Scalar
-----------------
Get the Fisher Matrix of the observable, ``S_{k,k'}  = ℜ⟨⟨O_k O_{k'}⟩⟩ = ℜ( ⟨O_k O_{k'}⟩ -⟨O_k⟩ ⟨O_{k'}⟩ ) `` where ``k`` and ``k'`` are labels of the parameters of the model.

When ansatz has only one parameter, the Fisher Matrix is a scalar, and the Fisher Information is the inverse of the Fisher Matrix.

Structure Factor in low momentum
-------------

``N_q = ⟨⟨n_qn_{-q}⟩⟩_{disorder}- ⟨⟨n_q⟩⟨n_{-q}⟩⟩_{disorder}``
"""
@inline function Carlo.register_evaluables(
        ::Type{MC}, eval::Evaluator, params::AbstractDict)

    evaluate!(eval, :fg, (:OL, :Og, :OLOg)) do OL, Og, OLOg
        @assert isa(OL, Real) "OL should be a real number, got $OL"
        return -2 * real(OLOg - OL * Og)
    end

    evaluate!(eval, :fisherScalar, (:Og, :Og2)) do Og, Og2
        return Og2 - Og^2
    end

    evaluate!(eval, :Nq, (:G2nqnmq, :G2nq, :G2)) do G2nqnmq, G2nq, G2
        return G2nqnmq / G2 - abs2(G2nq) / G2^2
    end
    return nothing
end

@inline function Carlo.write_checkpoint(mc::MC{B}, out::HDF5.Group) where {B}
    out["kappa_up"] = mc.κup
    out["kappa_down"] = mc.κdown
    return nothing
end

@inline function Carlo.read_checkpoint!(mc::MC{B}, in::HDF5.Group) where {B}
    mc.κup = read(in, "kappa_up")
    mc.κdown = read(in, "kappa_down")
    return nothing
end
