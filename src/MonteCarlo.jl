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
    update_W_matrices(mc::MC; K_up::Int, K_down::Int, l_up::Int, l_down::Int)
------------
Update the W matrices
"""
function update_W_matrices!(mc::MC; K_up::Int, K_down::Int, l_up::Int, l_down::Int)
    update_W!(mc.W_up; l = l_up, K = K_up)
    update_W!(mc.W_down; l = l_down, K = K_down)
end

"""
    update_W!(W::AbstractMatrix; l::Int, K::Int)
------------
Update the W matrix
``W'_{I,j} = W_{I,j} - W_{I,l} / W_{K,l} * (W_{K,j} - \\delta_{l,j})``
"""
function update_W!(W::AbstractMatrix; l::Int, K::Int)
    factors = W[:, l] ./ W[K, l]
    Krow = copy(W[K, :])
    Krow[l] -= 1.0
    @views for j in axes(W, 2)
        W[:, j] .-= factors .* Krow[j]
    end
    return nothing
end

function update_configurations!(
        mc, i_up::Int, i_down::Int, K_up::Int, K_down::Int, l_up::Int, l_down::Int)
    # Update W matrices
    update_W_matrices!(mc; K_up = K_up, K_down = K_down, l_up = l_up, l_down = l_down)
    # Update kappa configurations
    mc.κup[i_up], mc.κup[K_up] = 0, l_up
    mc.κdown[i_down], mc.κdown[K_down] = 0, l_down
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
    init_conf(rng::AbstractRNG, ns::Int, N_up::Int)

initialize a half-filled configuration for this system
"""
function init_conf(rng::AbstractRNG, ns::Int, N_up::Int)
    # dealing with conf_up
    # Initialize array with zeros
    κup = zeros(Int, ns)

    # Generate random positions for 1:N_up
    positions = randperm(rng, ns)[1:N_up]

    # Place 1:N_up at the random positions
    for (i, pos) in enumerate(positions)
        κup[pos] = i
    end
    κdown = zeros(Int, ns)
    # κdown should occupy the rest of the sites
    res_pos = randperm(rng, ns - N_up)
    for i in 1:ns
        if κup[i] == 0
            κdown[i] = pop!(res_pos)
        end
    end

    return κup, κdown
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
    mc.q = [2 * pi / params[:nx], 2 * pi / params[:ny]]
    mc.κup, mc.κdown = init_conf(ctx.rng, lat.ns, params[:N_up])
    tilde_U_up = tilde_U(mc.model.U_up, mc.κup)
    tilde_U_down = tilde_U(mc.model.U_down, mc.κdown)
    N_up = params[:N_up]
    N_down = params[:N_down]
    U_upinvs = zeros(eltype(mc.model.U_up), N_up, N_up)
    U_downinvs = zeros(eltype(mc.model.U_down), N_down, N_down)
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
                mc.κup, mc.κdown = init_conf(ctx.rng, lat.ns, params[:N_up])
                tilde_U_up = tilde_U(model.U_up, κup)
                tilde_U_down = tilde_U(model.U_down, κdown)
                continue
            else
                rethrow(e)  # If it's not a matrix inversion error,
            end
        end
    end
    # calculate W_up and W_down
    mc.W_up = mc.model.U_up * U_upinvs
    mc.W_down = mc.model.U_down * U_downinvs
    return nothing
end

function hop!(mc::MC{B}, ctx::MCContext, neigh, ns, conf) where {B}
    occupied_sites = findall(x -> (x != 0), conf)
    i = rand(ctx.rng, occupied_sites)
    site = sample(ctx.rng, neigh[i])
    # i and site have same spin thus can't hop
    block_cond = (is_occupied(mc.κup, i) && is_occupied(mc.κup, site)) ||
                 (is_occupied(mc.κdown, i) && is_occupied(mc.κdown, site))
    block_cond, i, site
end

@inline function Carlo.sweep!(mc::MC{B}, ctx::MCContext) where {B}
    neigh = mc.model.lattice.neigh
    ns = mc.model.lattice.ns
    r = rand(ctx.rng)
    block_cond_up, i_up, site_up = hop!(mc, ctx, neigh, ns, mc.κup)
    block_cond_down, i_down, site_down = hop!(mc, ctx, neigh, ns, mc.κdown)
    # Check if both hops are blocked
    if block_cond_up || block_cond_down
        measure!(ctx, :acc, 0.0)
        return nothing
    end
    # Calculate ratio based on selected move
    l_up = mc.κup[i_up]
    l_down = mc.κdown[i_down]

    # Calculate acceptance ratio
    ratio = mc.W_up[site_up, l_up] * mc.W_down[site_down, l_down]
    if ratio^2 >= 1
        update_configurations!(mc, i_up, i_down, site_up, site_down, l_up, l_down)
        measure!(ctx, :acc, 1.0)
    elseif ratio^2 < 1 && r < ratio^2
        update_configurations!(mc, i_up, i_down, site_up, site_down, l_up, l_down)
        measure!(ctx, :acc, 1.0)
    else
        measure!(ctx, :acc, 0.0)
    end

    # Re-evaluate W matrices periodically
    n_occupied = min(count(!iszero, mc.κup), count(!iszero, mc.κdown))
    if ctx.sweeps % n_occupied == 0
        try
            reevaluateW!(mc)
        catch e
            if e isa LinearAlgebra.SingularException
                @warn "lu factorization failed, aborting re-evaluation..."
                throw(e)
            end
        end
    end
    return nothing
end

@inline function Carlo.measure!(mc::MC{B}, ctx::MCContext) where {B}
    OL = getOL(mc)
    Og = getOg(mc)
    G = exp(mc.g * Og)
    measure!(ctx, :OL, OL)
    measure!(ctx, :Og, Og)
    measure!(ctx, :OLOg, OL * Og)
    measure!(ctx, :Og2, Og^2)

    measure!(ctx, :G2, G^2)

    # sampling N_q
    nx = mc.model.lattice.nx
    ny = mc.model.lattice.ny
    occ_2d = reshape(site_occupation.(mc.κup) + site_occupation.(mc.κdown), nx, ny)
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
    out["κup"] = mc.κup
    out["κdown"] = mc.κdown
    return nothing
end

@inline function Carlo.read_checkpoint!(mc::MC{B}, in::HDF5.Group) where {B}
    mc.κup = read(in, "κup")
    mc.κdown = read(in, "κdown")
    return nothing
end
