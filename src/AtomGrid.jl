module AtomicGrid

using IntegralGrid.BaseGrid: AbstractExtendedGrid, Grid, OneDGrid
using IntegralGrid.Angular: AngularGrid, convert_angular_sizes_to_degrees
using IntegralGrid.Utils: convert_cart_to_sph, convert_derivative_from_spherical_to_cartesian, generate_real_spherical_harmonics, generate_derivative_real_spherical_harmonics

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
end

function AtomGrid(
    rgrid::OneDGrid;
    degrees::Union{AbstractVector{<:Real},Nothing}=nothing,
    # sizes::Union{AbstractVector{<:Int},Nothing}=nothing,
    center::Union{AbstractVector{<:Real},Nothing}=[0.0, 0.0, 0.0],
    rotate::Int=0,
    use_spherical::Bool=false
)
    center = isnothing(center) ? [0.0, 0.0, 0.0] : center
    _input_type_check(rgrid, center)

    if rotate != 0 && !(0 <= rotate < 2^32 - length(rgrid.points))
        throw(
            ArgumentError(
                "rotate need to be an integer [0, 2^32 - length(rgrid)] rotate is not within [0, 2^32 - length(rgrid)], got $rotate"
            )
        )
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
    ) = _generate_atomic_grid(rgrid, degrees, rotate=_rot, use_spherical=use_spherical)

    return AtomGrid(
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

function from_preset(
    rgrid::OneDGrid;
    atnum::Int,
    preset::String,
    center::Vector{Float64}=[0.0, 0.0, 0.0],
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
    degs = convert_angular_sizes_to_degrees(npt, use_spherical)
    rad_degs = _find_l_for_rad_list(rgrid.points, rad, degs)
    AtomGrid(
        rgrid,
        degrees=rad_degs,
        center=center,
        rotate=rotate,
        use_spherical=use_spherical
    )
end

function from_pruned(
    rgrid::OneDGrid,
    radius::Real;
    sectors_r::Vector{<:Real},
    sectors_degree::Union{AbstractVector{<:Real},Nothing}=nothing,
    sectors_size::Union{AbstractVector{Int},Nothing}=nothing,
    center::Vector{<:Real}=[0.0, 0.0, 0.0],
    rotate::Int=0,
    use_spherical::Bool=false
)
    if isnothing(sectors_degree)
        sectors_degree = convert_angular_sizes_to_degrees(sectors_size, use_spherical)
    end
    center = isnothing(center) ? [0.0, 0.0, 0.0] : center
    _input_type_check(rgrid, center)
    degrees = _generate_degree_from_radius(
        rgrid,
        radius,
        sectors_r,
        sectors_degree,
        use_spherical
    )
    AtomGrid(
        rgrid,
        degrees=degrees,
        center=center,
        rotate=rotate,
        use_spherical=use_spherical
    )
end


function get_shell_grid(grid::AtomGrid, index::Int, r_sq::Bool=true)
    raw"""Get the spherical integral grid at radial point from specified index.

    The spherical integration grid has points scaled with the ith radial point
    and weights multipled by the ith weight of the radial grid.

    Note that when r=0 then the Cartesian points are all zeros.

    Parameters
    ----------
    index : int
        Index of radial points.
    r_sq : bool, default True
        If true, multiplies the angular grid weights with r^2.

    Returns
    -------
    AngularGrid
        AngularGrid at given radial index position and weights.

    """
    if !(0 ≤ index < length(grid.degrees))
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


function convert_cartesian_to_spherical(grid::AtomGrid, points::Array{Float64,2}=nothing, center::Array{Float64,1}=nothing)
    raw"""
    Convert a set of points from Cartesian to spherical coordinates.

    The conversion is defined as

    .. math::
        \begin{align}
            r &= \sqrt{x^2 + y^2 + z^2}\\
            \theta &= \text{arc}\tan \left(\frac{y}{x}\right)\\
            \phi &= \text{arc}\cos\left(\frac{z}{r}\right)
        \end{align}

    with the canonical choice :math:`r=0`, then :math:`\theta,\phi = 0`.
    If the `points` attribute is not specified, then atomic grid points are used
    and the canonical choice when :math:`r=0`, is the points
    :math:`(r=0, \theta_j, \phi_j)` where :math:`(\theta_j, \phi_j)` come
    from the Angular grid with the degree at :math:`r=0`.

    Parameters
    ----------
    points : Array{Float64,2}, optional
        Points in three-dimensions. Atomic grid points will be used if `points` is not given.
    center : Array{Float64,1}, optional
        Center of the atomic grid points. If `center` is not provided, then the atomic
        center of this class is used.

    Returns
    -------
    Array{Float64,2}
        Spherical coordinates of atoms respect to the center
        (radius :math:`r`, azimuthal :math:`\theta`, polar :math:`\phi`).
    """

    is_atomic = false
    if isnothing(points)
        points = AtomGrid.points
        is_atomic = true
    end

    if ndims(points) == 1
        points = reshape(points, :, 3)
    end

    center = isnothing(center) ? grid.center : convert(Array{Float64,1}, center)
    spherical_points = convert_cart_to_sph(points, center)

    # If atomic grid points are being converted, then choose canonical angles (when r=0)
    # to be from the degree specified of that point. The reasoning is to ensure that
    # the integration of spherical harmonic when l=l, m=0, is zero even when r=0.
    if is_atomic == true
        r_index = findall(x -> x == 0.0, grid.rgrid.points)
        for i in r_index
            # build angular grid for the degree at r=0
            agrid = AngularGrid(degree=grid._degs[i], use_spherical=grid.use_spherical)
            start_index = grid._indices[i]
            final_index = grid._indices[i+1]
            spherical_points[start_index:final_index, 2:end] = convert_cart_to_sph(agrid.points[:, 2:end])[:, 2:end]
        end
    end

    return spherical_points
end






end # end module AtomGrid