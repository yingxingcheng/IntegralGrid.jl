module BaseGrid

using LinearAlgebra, NearestNeighbors, NPZ

export AbstractGrid, Grid, get_points, get_weights, getindex, integrate, moments, save

abstract type AbstractGrid end

"""
Basic Grid struct for grid information storage.
"""
struct Grid{T<:Real,U<:Number} <: AbstractGrid
    _points::Union{AbstractArray{T,1},AbstractArray{T,2}}
    _weights::AbstractArray{U,1}
    _kdtree::Union{KDTree,Nothing}

    function Grid{T,U}(
        points::Union{AbstractArray{T,1},AbstractArray{T,2}},
        weights::AbstractArray{U,1},
        kdtree::Union{KDTree,Nothing}=nothing
    ) where {T<:Real,U<:Number}
        length(points) == length(weights) || error("Number of points and weights does not match.\nNumber of points: $(length(points)), Number of weights: $(length(weights)).")
        new{T,U}(points, weights, kdtree)
    end
end

function Grid(
    points::Union{AbstractArray{T,1},AbstractArray{T,2}},
    weights::AbstractArray{U,1},
    kdtree::Union{KDTree,Nothing}=nothing
) where {T<:Real,U<:Number}
    Grid{T,U}(points, weights, kdtree)
end


get_points(grid::AbstractGrid) = grid._points
get_weights(grid::AbstractGrid) = grid._weights
Base.length(grid::AbstractGrid) = length(get_weights(grid))

function Base.getindex(grid::T, index::Union{Int,AbstractRange{Int},AbstractArray{Int}}) where {T<:AbstractGrid}
    if typeof(index) <: Int
        return T(get_points(grid)[index:index], get_weights(grid)[index:index])
    else
        return T(get_points(grid)[index], get_weights(grid)[index])
    end
end

function integrate(grid::T, value_arrays::AbstractArray{U,1}...) where {T<:AbstractGrid,U<:Number}
    if length(value_arrays) < 1
        throw(ArgumentError("No array is given to integrate."))
    end
    grid_points_length = length(get_points(grid))
    for (i, array) in enumerate(value_arrays)
        if length(array) != grid_points_length
            throw(ArgumentError("Arg $i needs to be of size $grid_points_length."))
        end
    end
    return sum(get_weights(grid) .* reduce(.*, value_arrays))
end

function get_localgrid(grid::Grid, center::Union{Real,Array{Real,1}}, radius::Real)
    # TODO
    return
end

function moments(
    grid::T,
    orders,
    centers,
    func_vals,
    type_mom::String="cartesian",
    return_orders::Bool=false
) where {T<:AbstractGrid}
    if ndims(func_vals) > 1
        throw(ArgumentError("func_vals should have dimension one."))
    end
    if ndims(centers) != 2
        throw(ArgumentError("centers should have dimension one or two."))
    end
    if length(get_points(grid)) != length(centers)
        throw(ArgumentError("The dimension of the grid should match the dimension of the centers."))
    end
    if length(func_vals) != legnth(get_points(grid))
        throw(ArgumentError("The length of function values should match the number of points in the grid."))
    end
    if type_mom == "pure-radial" && orders == 0
        throw(ArgumentError("The n/order parameter for pure-radial multipole moments should be positive."))
    end

    if typeof(orders) <: Int || typeof(orders) <: Int32 || typeof(orders) <: Int64
        orders = (0:orders) |> collect
        if type_mom == "pure-radial"
            orders = orders[2:end]
        end
    else
        error("Orders should be either an integer, list, or numpy array.")
    end

    dim = shape(get_points(grid))[1]
    all_orders = generate_orders_horton_order(orders[1], type_mom, dim)
    for l_ord in orders[2:end]
        all_orders = vcat(all_orders, generate_orders_horton_order(l_ord, type_mom, dim))
    end

    integrals = []
    for center in eachrow(centers)
        centered_pts = self.points .- center

        if type_mom == "cartesian"
            cent_pts_with_order = centered_pts .^ all_orders'
            cent_pts_with_order = prod(cent_pts_with_order, dims=2)
            integral = sum(cent_pts_with_order .* func_vals .* self.weights, dims=1)
        elseif type_mom == "radial" || type_mom == "pure" || type_mom == "pure-radial"
            cent_pts_with_order = vec(norm.(centered_pts, dims=2))

            if type_mom == "pure" || type_mom == "pure-radial"
                sph_pts = convert_cart_to_sph(centered_pts)
                solid_harm = solid_harmonics(orders[end], sph_pts)

                if type_mom == "pure"
                    integral = sum(solid_harm .* func_vals .* self.weights, dims=1)
                elseif type_mom == "pure-radial"
                    n_princ, l_degrees, m_orders = eachcol(all_orders)
                    indices = l_degrees .^ 2
                    indices[m_orders.>0] .+= 2 .* m_orders[m_orders.>0] .- 1
                    indices[m_orders.<=0] .+= 2 .* abs.(m_orders[m_orders.<=0])
                    cent_pts_with_order = cent_pts_with_order .^ n_princ'
                    integral = sum(cent_pts_with_order .* solid_harm[indices] .* func_vals .* self.weights, dims=1)
                end
            elseif type_mom == "radial"
                cent_pts_with_order = cent_pts_with_order .^ vec(all_orders)
                integral = sum(cent_pts_with_order .* func_vals .* self.weights, dims=1)
            end
        end

        push!(integrals, integral)
    end

    if return_orders
        return hcat(integrals...)', all_orders
    end

    return hcat(integrals...)'  # output has shape (L, Number of centers)
end


function save(grid::T, filename::AbstractString) where {T<:AbstractGrid}
    save_dict = Dict("points" => get_points(grid), "weights" => get_weights(grid))
    npzwrite(filename, save_dict)
end

end # end of moudle