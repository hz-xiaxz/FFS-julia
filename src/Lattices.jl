abstract type AbstractBoundary end
struct Periodic <: AbstractBoundary end
struct Open <: AbstractBoundary end

struct LatticeRectangular{B}
    nx::Int
    ny::Int
    ns::Int
    neigh::AbstractVector
end

"""
    LatticeRectangular(nx::Int, ny::Int, B::Periodic)
    LatticeRectangular(nx::Int, ny::Int, B::Open)

Generate a rectangular lattice with periodic or open boundary conditions

* `nx` : number of sites in x-direction
* `ny` : number of sites in y-direction
* `B` : boundary condition, either `Periodic` or `Open`

"""
function LatticeRectangular(nx::Int, ny::Int, B::Periodic)
    ns = nx * ny
    # Create site index array
    neigh = Vector(undef, ns)
    rectl = reshape(1:ns, nx, ny)

    # Define neighbor indices for each direction
    up = circshift(rectl, (0, 1))
    right = circshift(rectl, (1, 0))
    down = circshift(rectl, (0, -1))
    left = circshift(rectl, (-1, 0))

    # Store neighbor information in a nested vector
    @inbounds for i in 1:ns
        neigh[i] = Set([up[i], right[i], down[i], left[i]])
    end
    return LatticeRectangular{B}(nx, ny, ns, neigh)
end

function LatticeRectangular(nx::Int, ny::Int, B::Open)
    ns = nx * ny
    neigh = Vector{Vector{Int}}(undef, ns)
    # Loop through each site
    @inbounds for i in 1:ns
        neighbors = Int[]

        # Identify valid neighbors for each direction
        if i % nx != 1
            push!(neighbors, i - 1)
        end
        if i % nx != 0
            push!(neighbors, i + 1)
        end
        if (i - 1) ÷ nx != 0
            push!(neighbors, i - nx)
        end
        if (i - 1) ÷ nx != ny - 1
            push!(neighbors, i + nx)
        end

        # Store neighbor list for the current site
        neigh[i] = neighbors
    end
    return LatticeRectangular{B}(nx, ny, ns, neigh)
end
