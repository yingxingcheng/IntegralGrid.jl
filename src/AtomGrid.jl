module AtomicGrid

using NPZ
using IntegralGrid.BaseGrid: AbstractGrid, AbstractExtendedGrid, Grid, OneDGrid, _getproperty, get_grid
using IntegralGrid.Angular: AngularGrid, convert_angular_sizes_to_degrees, _get_size_and_degree
using IntegralGrid.Utils: convert_cart_to_sph, convert_derivative_from_spherical_to_cartesian, generate_real_spherical_harmonics, generate_derivative_real_spherical_harmonics, newaxis
using PyCall
import IntegralGrid.BaseGrid.get_points


export AtomGrid, from_preset, from_pruned, convert_cartesian_to_spherical, integrate_angular_coordinates, spherical_average, radial_component_splines, interpolate, interpolate_low
export _generate_degree_from_radius, _find_l_for_rad_list, _generate_atomic_grid
export get_shell_grid

mutable struct AtomGrid <: AbstractExtendedGrid
    _grid::Grid
    rgrid::OneDGrid
    degrees::Union{AbstractVector{<:Real},Nothing}
    # sizes::Union{AbstractVector{<:Int},Nothing}
    center::Union{AbstractVector{<:Real},Nothing}
    rotate::Int
    use_spherical::Bool
    indices
    basis


    function AtomGrid(
        rgrid::OneDGrid;
        degrees::Union{AbstractVector{<:Real},Nothing}=nothing,
        sizes::AbstractVector{<:Int}=Int[],
        center::Union{AbstractVector{<:Real},Nothing}=[0.0, 0.0, 0.0],
        rotate::Int=0,
        use_spherical::Bool=false
    )
        center = isnothing(center) ? [0.0, 0.0, 0.0] : center
        _input_type_check(rgrid, center)

        if rotate != 0 && !(0 <= rotate < 2^32 - length(rgrid.points))
            throw(ArgumentError("rotate need to be an integer [0, 2^32 - length(rgrid)] rotate is not within [0, 2^32 - length(rgrid)], got $rotate"))
        end
        if isnothing(degrees)
            degrees = convert_angular_sizes_to_degrees(sizes)
        end
        if length(degrees) == 1
            degrees = fill(degrees[1], length(rgrid.points))
        end
        (
            _points,
            _weights,
            _indices,
            _degrees
        ) = _generate_atomic_grid(rgrid, degrees, rotate=rotate, use_spherical=use_spherical)

        return new(
            Grid(_points, _weights),
            rgrid,
            _degrees,
            center,
            rotate,
            use_spherical,
            _indices,
            nothing
        )
    end
end


function from_preset(
    rgrid::OneDGrid;
    atnum::Int,
    preset::String,
    center::AbstractVector{<:Real}=[0.0, 0.0, 0.0],
    rotate::Int=0,
    use_spherical::Bool=false
)
    if !isa(use_spherical, Bool)
        throw(ArgumentError("use_spherical $use_spherical should be of type Bool."))
    end
    if rgrid === nothing
        # TODO: generate a default rgrid, currently raise an error instead
        throw(ArgumentError("TODO: A default OneDGrid will be generated"))
    end
    center = isnothing(center) ? [0.0, 0.0, 0.0] : center
    _input_type_check(rgrid, center)

    fn_path = joinpath(@__DIR__, "data", "prune_grid", "prune_grid_$(preset).npz")
    _data = npzread(fn_path)
    rad, npt = _data["$(atnum)_rad"], _data["$(atnum)_npt"]
    degs = convert_angular_sizes_to_degrees(npt, use_spherical=use_spherical)
    rad_degs = _find_l_for_rad_list(rgrid.points, rad, degs)
    return AtomGrid(rgrid, degrees=rad_degs, center=center, rotate=rotate, use_spherical=use_spherical)
end

function from_pruned(
    rgrid::OneDGrid;
    radius::Real,
    sectors_r::AbstractVector,
    sectors_degree::Union{AbstractVector{<:Real},Nothing}=nothing,
    sectors_size::Union{AbstractVector{Int},Nothing}=nothing,
    center::AbstractVector{<:Real}=[0.0, 0.0, 0.0],
    rotate::Int=0,
    use_spherical::Bool=false
)
    if isnothing(sectors_degree)
        sectors_degree = convert_angular_sizes_to_degrees(sectors_size, use_spherical=use_spherical)
    end
    center = isnothing(center) ? [0.0, 0.0, 0.0] : center
    _input_type_check(rgrid, center)
    degrees = _generate_degree_from_radius(rgrid, radius, sectors_r, sectors_degree, use_spherical)
    return AtomGrid(rgrid, degrees=degrees, center=center, rotate=rotate, use_spherical=use_spherical)
end

function get_points(grid::AtomGrid)
    _grid = get_grid(grid)
    _points = _grid._points
    if ndims(_points) == 1
        return _points .+ getfield(grid, :center)
    else
        return _points .+ getfield(grid, :center)[newaxis, :]
    end
end

get_l_max(grid::AtomGrid) = maximum(grid.degrees)
get_n_shells(grid::AtomGrid) = length(grid.degrees)
Base.getproperty(grid::AtomGrid, key::Symbol) = (
    _getproperty(grid, key, extended_dict=Dict(:l_max => get_l_max, :n_shells => get_n_shells))
)



function get_shell_grid(grid::AtomGrid, index::Int; r_sq::Bool=true)
    if !(1 ≤ index ≤ length(grid.degrees))
        throw(ArgumentError("Index $index should be between 0 and less than number of radial points $(length(grid.degrees))."))
    end
    degree = grid.degrees[index]
    sphere_grid = AngularGrid(degree=degree, use_spherical=grid.use_spherical)

    pts = copy(sphere_grid.points)
    wts = copy(sphere_grid.weights)
    # Rotate the points
    # TODO:
    # if grid.rotate != 0
    #     rot_mt = R.random(random_state=grid.rotate + index).as_matrix()
    #     pts = pts * rot_mt
    # end

    pts = pts .* grid.rgrid[index].points
    wts = wts .* grid.rgrid[index].weights
    if r_sq == true
        wts = wts .* grid.rgrid[index].points .^ 2
    end
    return AngularGrid(pts, wts)
end


function convert_cartesian_to_spherical(
    grid::AtomGrid,
    points::Union{AbstractVecOrMat{<:Real},Nothing}=nothing,
    center::Union{AbstractVector{<:Real},Nothing}=nothing
)
    is_atomic = false
    if isnothing(points)
        points = grid.points
        is_atomic = true
    end

    if ndims(points) == 1
        points = reshape(points, :, 3)
    end

    center = isnothing(center) ? grid.center : center
    spherical_points = convert_cart_to_sph(points, center)

    # If atomic grid points are being converted, then choose canonical angles (when r=0)
    # to be from the degree specified of that point. The reasoning is to ensure that
    # the integration of spherical harmonic when l=l, m=0, is zero even when r=0.
    if is_atomic == true
        r_index = findall(x -> x == 0.0, grid.rgrid.points)
        for i in r_index
            # build angular grid for the degree at r=0
            agrid = AngularGrid(degree=grid.degrees[i], use_spherical=grid.use_spherical)
            start_index = grid.indices[i]
            final_index = grid.indices[i+1]
            spherical_points[start_index:final_index-1, 2:end] .= convert_cart_to_sph(agrid.points)[:, 2:end]
        end
    end

    return spherical_points
end


function reshape_for_broadcast(array_dim::Int, weights::AbstractVector)
    # Create a shape compatible for broadcasting with func_vals
    reshape_dims = ones(Int, array_dim)
    reshape_dims[end] = length(weights)

    # Reshape weights to the appropriate size for broadcasting
    weights_reshaped = reshape(weights, tuple(reshape_dims...))
    return weights_reshaped
end

function integrate_angular_coordinates(grid::AtomGrid, func_vals::AbstractArray{<:Real})
    if ndims(func_vals) == 1
        func_vals = reshape(func_vals, 1, :)
    end
    _ndim = ndims(func_vals)
    weights_reshaped = reshape_for_broadcast(_ndim, grid.weights)
    prod_value = func_vals .* weights_reshaped

    radial_coefficients = []
    for i in 1:grid.n_shells
        # Get a slice of the function values for the current shell
        res = sum(prod_value[[Colon() for _ in 1:_ndim-1]..., grid.indices[i]:grid.indices[i+1]-1], dims=_ndim)
        @assert ndims(res) == _ndim && size(res, _ndim) == 1
        push!(radial_coefficients, res)
    end

    # fixme: StackOverFlowError when the radial_coefficients is too long
    # Convert to array and adjust the size
    radial_coefficients = cat(radial_coefficients..., dims=_ndim)

    # Divide by r^2 and weights
    r_weights = grid.rgrid.points .^ 2 .* grid.rgrid.weights
    r_weights_reshaped = reshape_for_broadcast(_ndim, r_weights)
    radial_coefficients = radial_coefficients ./ r_weights_reshaped

    # For small radii
    r_index = findall(x -> x < 1e-8, grid.rgrid.points)
    for i in r_index
        agrid = AngularGrid(degree=grid.degrees[i], use_spherical=grid.use_spherical)
        _weights_reshaped = reshape_for_broadcast(_ndim, agrid.weights)
        values = func_vals[[Colon() for _ in 1:_ndim-1]..., grid.indices[i]:grid.indices[i+1]-1] .* _weights_reshaped
        radial_coefficients[[Colon() for _ in 1:_ndim-1]..., i] .= sum(values, dims=_ndim)
    end
    return radial_coefficients
end

function spherical_average(grid, func_vals::AbstractVector{<:Number})
    f_radial = integrate_angular_coordinates(grid, func_vals)
    f_radial /= 4.0 * π
    x = grid.rgrid.points
    y = vec(f_radial)
    SCIPY_LIB = pyimport("scipy.interpolate")
    return SCIPY_LIB.CubicSpline(x=x, y=y)
end

function radial_component_splines(grid, func_vals::AbstractVector{<:Number})
    if length(func_vals) != grid.size
        throw(ArgumentError("The size of values does not match with the size of grid. 
        The size of value array: $(length(func_vals)). 
        The size of grid: $(grid.size)"))
    end

    if isnothing(grid.basis)
        sph_pts = convert_cartesian_to_spherical(grid)
        theta, phi = sph_pts[:, 2], sph_pts[:, 3]
        grid.basis = generate_real_spherical_harmonics(grid.l_max ÷ 2, theta, phi)
    end

    values = grid.basis .* reshape(func_vals, 1, :)
    radial_components = integrate_angular_coordinates(grid, values)

    for i in 1:grid.n_shells
        if grid.degrees[i] != grid.l_max
            num_nonzero_sph = (grid.degrees[i] ÷ 2 + 1)^2
            radial_components[num_nonzero_sph+1:end, i] .= 0.0
        end
    end

    SCIPY_LIB = pyimport("scipy.interpolate")
    res = []
    for y in eachrow(radial_components)
        x = grid.rgrid.points
        push!(res, SCIPY_LIB.CubicSpline(x=x, y=vec(y)))
    end
    # return [SCIPY_LIB.CubicSpline(x=grid.rgrid.points, y=sph_val) for sph_val in radial_components]
    return res
end

function interpolate(grid::AbstractGrid, func_vals::AbstractVector{<:Number})
    throw(ArgumentError("Not implemented!"))
end

function interpolate(grid::AtomGrid, func_vals::AbstractVector{<:Number})
    splines = radial_component_splines(grid, func_vals)
    NUMPY = pyimport("numpy")


    function interpolate_low(
        points::Union{AbstractVecOrMat{<:Real}};
        deriv::Int=0,
        deriv_spherical::Bool=false,
        only_radial_deriv::Bool=false)
        if deriv_spherical && only_radial_deriv
            warn("Since `only_radial_derivs` is true, then only the derivative wrt to" *
                 "radius is returned and `deriv_spherical` value is ignored.")
        end

        sph_pts = convert_cartesian_to_spherical(grid, points)
        r_pts, theta, phi = sph_pts[:, 1], sph_pts[:, 2], sph_pts[:, 3]

        r_values = [spline(r_pts, deriv) for spline in splines]
        r_sph_harm = generate_real_spherical_harmonics(grid.l_max ÷ 2, theta, phi)

        if !only_radial_deriv && deriv == 1
            radial_components = [spline(r_pts, 0) for spline in splines]
            deriv_sph_harm = generate_derivative_real_spherical_harmonics(grid.l_max ÷ 2, theta, phi)

            deriv_r = NUMPY.einsum("ij,ij->j", r_values, r_sph_harm)
            deriv_theta = NUMPY.einsum("ij,ij->j", radial_components, deriv_sph_harm[1, :, :])
            deriv_phi = NUMPY.einsum("ij,ij->j", radial_components, deriv_sph_harm[2, :, :])

            if deriv_spherical
                return hcat(deriv_r, deriv_theta, deriv_phi)
            end

            derivs = zeros(eltype(deriv_r), length(r_pts), 3)
            for i in 1:length(r_pts)
                # radial_i, theta_i, phi_i = r_pts[i_pt], theta[i_pt], phi[i_pt]
                res = convert_derivative_from_spherical_to_cartesian(
                    deriv_r[i], deriv_theta[i], deriv_phi[i], r_pts[i], theta[i], phi[i]
                )
                derivs[i, :] .= res
            end
            return derivs
        elseif !only_radial_deriv && deriv != 0
            throw(ArgumentError("Higher order derivatives are only supported for derivatives" *
                                "with respect to the radius. Deriv is $(deriv)."))
        end

        return NUMPY.einsum("ij,ij->j", r_values, r_sph_harm)
    end

    return interpolate_low
end

function _input_type_check(rgrid::OneDGrid, center::AbstractVector{<:Real})
    if !(rgrid isa OneDGrid)
        throw(ArgumentError("Argument rgrid is not an instance of OneDGrid, got $(typeof(rgrid))."))
    end
    if !isnothing(rgrid.domain) && rgrid.domain[1] < 0
        throw(ArgumentError("Argument rgrid should have a positive domain, got $(rgrid.domain)"))
    elseif minimum(rgrid.points) < 0.0
        throw(ArgumentError("Smallest rgrid.points is negative, got $(minimum(rgrid.points))"))
    end
    if size(center) != (3,)
        throw(ArgumentError("Center should be of shape (3,), got $(size(center))."))
    end
end

function _generate_degree_from_radius(
    rgrid::OneDGrid,
    radius::Real,
    r_sectors::AbstractVector,
    deg_sectors::AbstractVector{Int},
    use_spherical::Bool=false
)
    if isempty(deg_sectors)
        throw(ArgumentError("deg_sectors can't be empty."))
    end
    if length(deg_sectors) - length(r_sectors) != 1
        throw(ArgumentError("degs should have only one more element than r_sectors."))
    end
    matched_deg = [
        _get_size_and_degree(degree=d, size=nothing, use_spherical=use_spherical)[1]
        for d in deg_sectors
    ]
    rad_degs = _find_l_for_rad_list(rgrid.points, radius .* r_sectors, matched_deg)
    return rad_degs
end

function _find_l_for_rad_list(radial_arrays, radius_sectors, deg_sectors)
    if length(radius_sectors) == 0
        position = ones(Int, length(radial_arrays))
    else
        position = vec(sum(broadcast(>, reshape(radial_arrays, :, 1), reshape(radius_sectors, 1, :)), dims=2))
        position .+= 1 # for julia index
    end
    return deg_sectors[position]
end

function _generate_atomic_grid(
    rgrid::OneDGrid,
    degrees::Union{AbstractVector{<:Real},Nothing};
    rotate::Int=0,
    use_spherical::Bool=false
)
    if length(degrees) != rgrid.size
        throw(ArgumentError("The shape of radial grid does not match given degs."))
    end

    all_points, all_weights = [], []
    shell_pt_indices = ones(Int, length(degrees) + 1)
    actual_degrees = zeros(Int, length(degrees))

    for (i, deg_i) in enumerate(degrees)
        sphere_grid = AngularGrid(degree=deg_i, use_spherical=use_spherical)
        points, weights = copy(sphere_grid.points), copy(sphere_grid.weights)
        # push!(actual_degrees, sphere_grid.degree)
        actual_degrees[i] = sphere_grid.degree

        if rotate == 0
            # do nothing
        else
            throw(ArgumentError("rotate != 0 is not implemented!"))
            # @assert rotate isa Int
            # rot_mt = R.random(random_state=rotate + i).as_matrix()
            # points = points * rot_mt
        end

        points = points .* rgrid[i].points
        weights = weights .* rgrid[i].weights .* rgrid[i].points .^ 2
        shell_pt_indices[i+1] = shell_pt_indices[i] + size(points, 1)
        push!(all_points, points)
        push!(all_weights, weights)
    end

    indices = shell_pt_indices
    points = vcat(all_points...)
    weights = vcat(all_weights...)
    return points, weights, indices, actual_degrees
end





end # end module AtomGrid