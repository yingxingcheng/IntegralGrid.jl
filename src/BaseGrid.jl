module BaseGrid

using LinearAlgebra
using NearestNeighbors

export AbstractGrid, Grid, integrate, getindex

abstract type AbstractGrid end

"""
Basic Grid struct for grid information storage.
"""
struct Grid{T<:Real,U<:Number} <: AbstractGrid
    points::Union{Array{T,1},Array{T,2}}
    weights::Array{U,1}
    kdtree::Union{KDTree,Nothing}

    function Grid(points::Union{Array{T,1},Array{T,2}}, weights::Array{U,1}, kdtree::Union{KDTree,Nothing}=nothing) where {T<:Real,U<:Number}
        if length(points) != length(weights)
            error("Number of points and weights does not match.\nNumber of points: $(length(points)), Number of weights: $(length(weights)).")
        end
        new{T,U}(points, weights, kdtree)
    end
end

function Base.getindex(grid::Grid{T,U}, index::Integer) where {T<:Real,U<:Number}
    return grid.points[index], grid.weights[index]
end


function size(grid::Grid{T,U}) where {T<:Real,U<:Number}
    return length(grid.weights)
end

function integrate(grid::Grid{T,U}, value_arrays...) where {T<:Real,U<:Number}
    if length(value_arrays) < 1
        throw(ArgumentError("No array is given to integrate."))
    end
    grid_points_length = length(grid.points)
    for (i, array) in enumerate(value_arrays)
        if !(typeof(array) <: Array{U,1})
            throw(ArgumentError("Arg $i is not a Real Array."))
        end
        if length(array) != grid_points_length
            throw(ArgumentError("Arg $i needs to be of size $grid_points_length."))
        end
    end
    return sum(grid.weights .* reduce(.*, value_arrays))
end

function get_localgrid(grid::Grid{T,U}, center::Union{Real,Array{Real,1}}, radius::Real) where {T<:Real,U<:Number}
    # TODO
    return
end

function moments(grid::Grid{T,U}, orders::Integer, centers::Array{Real,2}, func_vals::Array{Number,1}) where {T<:Real,U<:Number}
    # TODO
    return
end

function save(grid::Grid, filename::AbstractString)
    # TODO
    return
end

end # end of moudle