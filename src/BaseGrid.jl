module BaseGrid

using LinearAlgebra
using NearestNeighbors

export AbstractGrid, Grid, integrate, getindex

abstract type AbstractGrid end

"""
Basic Grid struct for grid information storage.
"""
struct Grid <: AbstractGrid
    points::Union{Array{Float64,1},Array{Float64,2}}
    weights::Array{Float64}
    # kdtree::KDTree{Float64, Float64}  # Requires KDTree package

    function Grid(points::Union{Array{Float64,1},Array{Float64,2}}, weights::Array{Float64,1})
        if length(points) != length(weights)
            throw("""Number of points and weights does not match.
            Number of points: $(length(points)), 
            Number of weights: $(length(weights)).
            """)
        end
        if ndims(weights) != 1
            throw("Argument weights should be a 1-D array. 
            weights.ndim = $(ndims(weights))")
        end
        if ndims(points) ∉ [1, 2]
            throw("Argument points should be a 1D or 2D array. 
            points.ndim = $(ndims(points))")
        end
        new(points, weights)
    end

    function Grid(; points::Union{Array{Float64,1},Array{Float64,2}}, weights::Array{Float64,1})
        return Grid(points, weights)
    end
end

function Base.getindex(grid::Grid, index::Int)
    return grid.points[index], grid.weights[index]
end


function size(grid::Grid)
    return length(grid.weights)
end

function integrate(grid::Grid, value_arrays...)
    if length(value_arrays) < 1
        throw(ArgumentError("No array is given to integrate."))
    end

    for (i, array) in enumerate(value_arrays)
        if !(typeof(array) <: Array{Float64,1})
            throw(ArgumentError("Arg $i is not a Float64 Array."))
        end

        if length(array) != length(grid.points)
            throw(ArgumentError("Arg $i needs to be of size $(length(grid.points))."))
        end
    end

    integrand = grid.weights .* reduce(.*, value_arrays)
    result = sum(integrand)
    return result
end

function get_localgrid(grid::Grid, center::Union{Float64,Array{Float64,1}}, radius::Float64)
    # TODO
    return
end

function moments(grid::Grid, orders::Integer, centers::Array{Float64,2}, func_vals::Array{Float64,1})
    # TODO
    return
end

function save(grid::Grid, filename::AbstractString)
    # TODO
    return
end

end # end of moudle