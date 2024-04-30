abstract type AbstractBoundary end
struct Periodic <: AbstractBoundary end
struct Open <: AbstractBoundary end

struct LatticeRectangular{B}
    nx::Int
    ny::Int
    neigh::Vector{Vector{Int}}
end

"""
    LatticeRectangular(nx::Int, ny::Int, B::Periodic)
    LatticeRectangular(nx::Int, ny::Int, B::Open)
"""
function LatticeRectangular(nx::Int, ny::Int, B::Periodic)
    # Create site index array
    neigh = Vector{Vector{Int}}(undef, nx * ny)
    rectl = reshape(1:nx*ny, nx, ny)

    # Define neighbor indices for each direction
    up = circshift(rectl, (0, 1))
    right = circshift(rectl, (1, 0))
    down = circshift(rectl, (0, -1))
    left = circshift(rectl, (-1, 0))

    # Store neighbor information in a nested vector
    for i in 1:nx*ny
        neigh[i] = [up[i], right[i], down[i], left[i]]
    end
    return LatticeRectangular{B}(nx, ny, neigh)
end

function LatticeRectangular(nx::Int, ny::Int, B::Open)
    neigh = Vector{Vector{Int}}(undef, nx * ny)
    # Loop through each site
    for i in 1:nx*ny
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
    return LatticeRectangular{B}(nx, ny, neigh)

end



