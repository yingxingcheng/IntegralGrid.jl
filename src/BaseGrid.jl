module BaseGrid

using LinearAlgebra, NearestNeighbors, NPZ

export AbstractGrid
export Grid, LocalGrid, OneDGrid

export get_points, get_weights, get_kdtree, get_size, get_gird
export get_center, get_indices
export get_domain
export getindex, integrate, moments, save, get_localgrid
export _d_grid, _d_localgrid, _d_onedgrid

abstract type AbstractGrid end

"""
Basic Grid struct for grid information storage.
"""
mutable struct Grid <: AbstractGrid
    _points::VecOrMat{<:Real}
    _weights::Vector{<:Number}
    _kdtree::Union{KDTree,Nothing}

    function Grid(
        points::VecOrMat{<:Real},
        weights::Vector{<:Number},
        kdtree::Union{KDTree,Nothing}=nothing)
        check_input(points, weights)
        new(points, weights, kdtree)
    end
end

# the subinstance must have a field named '_grid'
get_grid(grid::AbstractGrid) = getfield(grid, :_grid)
get_grid(grid::Grid) = grid
get_points(grid::AbstractGrid) = get_grid(grid)._points
get_weights(grid::AbstractGrid) = get_grid(grid)._weights
get_kdtree(grid::AbstractGrid) = get_grid(grid)._kdtree
get_size(grid::AbstractGrid) = size(grid.weights, 1)

_d_grid = Dict(:points => get_points, :weights => get_weights, :kdtree => get_kdtree, :size => get_size)
function Base.getproperty(grid::Grid, key::Symbol)
    return key in keys(_d_grid) ? _d_grid[key](grid) : getfield(grid, key)
end

function set_kdtree(grid::AbstractGrid, kdtree::Union{KDTree,Nothing})
    get_grid(grid)._kdtree = kdtree
end


mutable struct LocalGrid <: AbstractGrid
    _grid::Grid
    _center::Union{<:Real,Vector{<:Real}}
    _indices::Union{Vector{<:Real},Nothing}

    function LocalGrid(
        points::VecOrMat{<:Real},
        weights::Vector{<:Number},
        center::Union{<:Real,Vector{<:Real}},
        indices::Union{Vector{<:Real},Nothing}=nothing
    )
        new(Grid(points, weights, nothing), center, indices)
    end
end

get_center(grid::LocalGrid) = grid._center
get_indices(grid::LocalGrid) = grid._indices
_d_localgrid = Dict(:center => get_center, :indices => get_indices)
function Base.getproperty(grid::LocalGrid, key::Symbol)
    if key in keys(_d_grid)
        return _d_grid[key](grid)
    elseif key in keys(_d_localgrid)
        return _d_localgrid[key](grid)
    else
        getfield(grid, key)
    end
end


mutable struct OneDGrid <: AbstractGrid
    _grid::Grid
    _domain::Union{Nothing,Tuple{<:Real,<:Real}}

    function OneDGrid(
        points::Vector{<:Real},
        weights::Vector{<:Real},
        domain::Union{Nothing,Tuple{<:Real,<:Real}}=nothing)
        check_input(points, weights, domain=domain)
        new(Grid(points, weights, nothing), domain)
    end
end

get_domain(grid::OneDGrid) = grid._domain
_d_onedgrid = Dict(:domain => get_domain)
function Base.getproperty(grid::OneDGrid, key::Symbol)
    if key in keys(_d_grid)
        return _d_grid[key](grid)
    elseif key in keys(_d_onedgrid)
        return _d_onedgrid[key](grid)
    else
        getfield(grid, key)
    end
end


function check_input(points::VecOrMat{<:Real}, weights::Vector{<:Number}; domain::Union{Nothing,Tuple{<:Real,<:Real}}=nothing)
    sizep, sizew = size(points, 1), size(weights, 1)
    if sizep != sizew
        throw(ArgumentError("Number of points and weights does not match.\n"))
    end

    if !isnothing(domain)
        if length(domain) != 2 || domain[1] > domain[2]
            throw(ArgumentError("domain should be an ascending tuple of length 2. domain=$domain"))
        end
        min_p = minimum(points)
        if domain[1] - 1e-7 > min_p
            throw(ArgumentError("point coordinates should not be below domain! $min_p < $(domain[1])"))
        end
        max_p = maximum(points)
        if domain[2] + 1e-7 < max_p
            throw(ArgumentError("point coordinates should not be above domain! $(domain[2]) < $max_p"))
        end
    end
end

# which one is better?
# Method 1
function Base.getindex(grid::T, index) where {T<:AbstractGrid}
    if typeof(index) <: Int
        return T(grid.points[index:index], grid.weights[index:index])
    else
        return T(grid.points[index], grid.weights[index])
    end
end

# Method 2
# function Base.getindex(grid::AbstractGrid, index)
#     if typeof(index) <: Int
#         return typeof(grid)(grid.points[index:index],grid.weights[index:index])
#     else
#         return typeof(grid)(grid.points[index],grid.weights[index])
#     end
# end

function Base.getindex(grid::LocalGrid, index)
    if typeof(index) <: Int
        OneDGrid([grid.points[index]], [grid.weights[index]], get_domain(grid))
    else
        OneDGrid(grid.points[index], grid.weights[index], get_domain(grid))
    end
end

#-------------------------------------------------------------
# Common functions
#-------------------------------------------------------------

# Here uses Vector{<:Number} instead of Vector{<:Number}, 
# becuase ! (Vector{Float64} <: Vector{Number}), i.e., invariant
"""
Calculate integrate on the grid.
"""
function integrate(grid::AbstractGrid, value_arrays::Vector{<:Number}...)
    if length(value_arrays) < 1
        throw(ArgumentError("No array is given to integrate."))
    end
    for (i, array) in enumerate(value_arrays)
        if length(array) != grid.size
            throw(ArgumentError("Arg $i needs to be of size $(grid.size)."))
        end
    end
    return sum(grid.weights .* reduce(.*, value_arrays))
end

"""
    get_localgrid(center::Union{Float64, Vector{Float64}}, radius::Float64) -> LocalGrid

Create a grid containing points within the given `radius` of `center`.

# Arguments
- `center`: Cartesian coordinates of the center of the local grid.
- `radius`: Radius of sphere around the center. When equal to `Inf`, the
  local grid coincides with the whole grid, which can be useful for debugging.

# Returns
- `LocalGrid`: Instance of `LocalGrid`.
"""
function get_localgrid(grid::AbstractGrid, center::Union{<:Real,Vector{<:Real}}, radius::T) where {T<:Real}
    if typeof(center) <: Real
        center = [center]
    end
    sizep = ndims(grid._points) == 2 ? 1 : size(grid._points, 2)
    if size(center, 1) != sizep
        throw(ArgumentError(
            "Argument center has the wrong shape \n" *
            "center.shape: $(size(center)), points.shape: $(sizep)"
        ))
    end
    if radius < 0
        throw(ArgumentError("Negative radius: $radius"))
    end
    if !(isfinite(radius) || radius == Inf)
        throw(ArgumentError("Invalid radius: $radius"))
    end
    if radius == Inf
        return LocalGrid(grid._points, grid.weights, center, collect(1:get_size(grid)))
    else
        # When points.ndim == 1, we have to reshape a few things to make the input compatible with KDTree
        _points = ndims(grid._points) == 1 ? reshape(grid._points, (:, 1)) : grid._points
        if get_kdtree(grid) === nothing
            # grid._kdtree = KDTree(_points')
            set_kdtree(grid, KDTree(_points'))
        end
        indices = inrange(get_kdtree(grid), reshape(center, (1, :)), radius, false)[1]
        return LocalGrid(grid._points[indices], grid.weights[indices], center, indices)
    end
end


function moments(
    grid::AbstractGrid,
    orders,
    centers,
    func_vals,
    type_mom::String="cartesian",
    return_orders::Bool=false
)
    if ndims(func_vals) > 1
        throw(ArgumentError("func_vals should have dimension one."))
    end
    if ndims(centers) != 2
        throw(ArgumentError("centers should have dimension one or two."))
    end
    if length(grid.points) != length(centers)
        throw(ArgumentError("The dimension of the grid should match the dimension of the centers."))
    end
    if length(func_vals) != legnth(grid.points)
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

    dim = shape(grid.points)[1]
    all_orders = generate_orders_horton_order(orders[1], type_mom, dim)
    for l_ord in orders[2:end]
        all_orders = vcat(all_orders, generate_orders_horton_order(l_ord, type_mom, dim))
    end

    integrals = []
    for center in eachrow(centers)
        centered_pts = grid.points .- center

        if type_mom == "cartesian"
            cent_pts_with_order = centered_pts .^ all_orders'
            cent_pts_with_order = prod(cent_pts_with_order, dims=2)
            integral = sum(cent_pts_with_order .* func_vals .* grid.weights, dims=1)
        elseif type_mom == "radial" || type_mom == "pure" || type_mom == "pure-radial"
            cent_pts_with_order = vec(norm.(centered_pts, dims=2))

            if type_mom == "pure" || type_mom == "pure-radial"
                sph_pts = convert_cart_to_sph(centered_pts)
                solid_harm = solid_harmonics(orders[end], sph_pts)

                if type_mom == "pure"
                    integral = sum(solid_harm .* func_vals .* grid.weights, dims=1)
                elseif type_mom == "pure-radial"
                    n_princ, l_degrees, m_orders = eachcol(all_orders)
                    indices = l_degrees .^ 2
                    indices[m_orders.>0] .+= 2 .* m_orders[m_orders.>0] .- 1
                    indices[m_orders.<=0] .+= 2 .* abs.(m_orders[m_orders.<=0])
                    cent_pts_with_order = cent_pts_with_order .^ n_princ'
                    integral = sum(cent_pts_with_order .* solid_harm[indices] .* func_vals .* grid.weights, dims=1)
                end
            elseif type_mom == "radial"
                cent_pts_with_order = cent_pts_with_order .^ vec(all_orders)
                integral = sum(cent_pts_with_order .* func_vals .* grid.weights, dims=1)
            end
        end

        push!(integrals, integral)
    end

    if return_orders
        return hcat(integrals...)', all_orders
    end

    return hcat(integrals...)'  # output has shape (L, Number of centers)
end


function save(grid::AbstractGrid, filename::String)
    save_dict = Dict("points" => grid.points, "weights" => grid.weights)
    npzwrite(filename, save_dict)
end

function save(grid::LocalGrid, filename::String)
    save_dict = Dict("points" => grid.points, "weights" => grid.weights,
        "center" => grid.center, "indices" => get_indices(grid))
    npzwrite(filename, save_dict)
end


end # end of moudle