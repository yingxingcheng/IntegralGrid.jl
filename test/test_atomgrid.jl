using LinearAlgebra
using Test
using IntegralGrid.BaseGrid
using IntegralGrid.OnedGrid
using IntegralGrid.RTransform
using IntegralGrid.AtomicGrid
using IntegralGrid.Angular
using IntegralGrid.Utils
using Distributions
using PyCall

function test_total_atomic_grid()
    # Normal initialization test.
    radial_pts = collect(0.1:0.1:1.0)
    radial_wts = fill(0.1, 10)
    rgrid = OneDGrid(radial_pts, radial_wts)
    rad = 0.5
    r_sectors = [0.5, 1, 1.5]
    degs = [6, 14, 14, 6]
    # generate a proper instance without failing.
    ag_ob = from_pruned(rgrid, radius=rad, sectors_r=r_sectors, sectors_degree=degs)
    @test isa(ag_ob, AtomGrid)
    @test length(ag_ob.indices) == 11
    @test ag_ob.l_max == 15
    ag_ob = from_pruned(
        rgrid, radius=rad, sectors_r=Int[], sectors_degree=[6]
    )
    @test isa(ag_ob, AtomGrid)
    @test length(ag_ob.indices) == 11
    ag_ob = AtomGrid(rgrid, sizes=[110])
    @test ag_ob.l_max == 17
    @test isapprox(ag_ob.degrees, fill(17, 10))
    @test ag_ob.size == 110 * 10
    # new init AtomGrid
    ag_ob2 = AtomGrid(rgrid, degrees=[17])
    @test ag_ob2.l_max == 17
    @test isapprox(ag_ob2.degrees, fill(17, 10))
    @test ag_ob2.size == 110 * 10
    @test isa(ag_ob.rgrid, OneDGrid)
    @test isapprox(ag_ob.rgrid.points, rgrid.points)
    @test isapprox(ag_ob.rgrid.weights, rgrid.weights)
end

function test_from_predefined()
    # Test grid construction with predefined grid.
    # test coarse grid
    pts = UniformInteger(20)
    tf = PowerRTransform(7.0879993828935345e-06, 16.05937640019924)
    rad_grid = transform_1d_grid(tf, pts)
    atgrid = from_preset(rad_grid, atnum=1, preset="coarse")
    # 604 points for coarse H atom
    @test atgrid.size == 616
    @test isapprox(
        sum(exp.(-sum(atgrid.points .^ 2, dims=2)) .* atgrid.weights),
        5.56840953,
    )

    # test medium grid
    pts = UniformInteger(24)
    tf = PowerRTransform(3.69705074304963e-06, 19.279558946793685)
    rad_grid = transform_1d_grid(tf, pts)
    atgrid = from_preset(rad_grid, atnum=1, preset="medium")
    # 928 points for coarse H atom
    @test atgrid.size == 940
    @test isapprox(
        sum(exp.(-sum(atgrid.points .^ 2, dims=2)) .* atgrid.weights),
        5.56834559,
    )
    # test fine grid
    pts = UniformInteger(34)
    tf = PowerRTransform(2.577533167224667e-07, 16.276983371222354)
    rad_grid = transform_1d_grid(tf, pts)
    atgrid = from_preset(rad_grid, atnum=1, preset="fine")
    # 1984 points for coarse H atom
    @test atgrid.size == 1984 + 4 * 4
    @test isapprox(
        sum(exp.(-sum(atgrid.points .^ 2, dims=2)) .* atgrid.weights),
        5.56832800,
    )
    # test veryfine grid
    pts = UniformInteger(41)
    tf = PowerRTransform(1.1774580743206259e-07, 20.140888089596444)
    rad_grid = transform_1d_grid(tf, pts)
    atgrid = from_preset(rad_grid, atnum=1, preset="veryfine")
    # 3154 points for coarse H atom
    @test atgrid.size == 3154 + 4 * 6
    @test isapprox(
        sum(exp.(-sum(atgrid.points .^ 2, dims=2)) .* atgrid.weights),
        5.56832800,
    )
    # test ultrafine grid
    pts = UniformInteger(49)
    tf = PowerRTransform(4.883104847991021e-08, 21.05456999309752)
    rad_grid = transform_1d_grid(tf, pts)
    atgrid = from_preset(rad_grid, atnum=1, preset="ultrafine")
    # 4546 points for coarse H atom
    @test atgrid.size == 4546 + 4 * 6
    @test isapprox(
        sum(exp.(-sum(atgrid.points .^ 2, dims=2)) .* atgrid.weights),
        5.56832800,
    )
    # test insane grid
    pts = UniformInteger(59)
    tf = PowerRTransform(1.9221827244049134e-08, 21.413278983919113)
    rad_grid = transform_1d_grid(tf, pts)
    atgrid = from_preset(rad_grid, atnum=1, preset="insane")
    # 6622 points for coarse H atom
    @test atgrid.size == 6622 + 4 * 7
    @test isapprox(
        sum(exp.(-sum(atgrid.points .^ 2, dims=2)) .* atgrid.weights),
        5.56832800,
    )
end

function test_from_pruned_with_degs_and_size()
    # Test different initialize method.
    radial_pts = collect(0.1:0.1:1.0)
    radial_wts = fill(0.1, 10)
    rgrid = OneDGrid(radial_pts, radial_wts)
    rad = 0.5
    r_sectors = [0.5, 1, 1.5]
    degs = [3, 5, 7, 5]
    size = [6, 14, 26, 14]
    # construct atomic grid with degs
    atgrid1 = from_pruned(rgrid, radius=rad, sectors_r=r_sectors, sectors_degree=degs)
    # construct atomic grid with size
    atgrid2 = from_pruned(rgrid, radius=rad, sectors_r=r_sectors, sectors_size=size)
    # test two grids are the same
    @test atgrid1.size == atgrid2.size
    @test isapprox(atgrid1.points, atgrid2.points)
    @test isapprox(atgrid1.weights, atgrid2.weights)
end

function test_find_l_for_rad_list()
    # Test private method find_l_for_rad_list.
    radial_pts = collect(0.1:0.1:1.0)
    radial_wts = fill(0.1, 10)
    rgrid = OneDGrid(radial_pts, radial_wts)
    rad = 1
    r_sectors = [0.2, 0.4, 0.8]
    degs = [3, 5, 7, 3]
    atomic_grid_degree = _find_l_for_rad_list(rgrid.points, rad .* r_sectors, degs)
    @test atomic_grid_degree == [3, 3, 5, 5, 7, 7, 7, 7, 3, 3]
end

function test_generate_atomic_grid()
    # Test for generating atomic grid.
    # setup testing class
    rad_pts = [0.1, 0.5, 1]
    rad_wts = [0.3, 0.4, 0.3]
    rad_grid = OneDGrid(rad_pts, rad_wts)
    degs = [2, 5, 7]
    pts, wts, ind, degrees = _generate_atomic_grid(rad_grid, degs)
    @test size(pts, 1) == 50
    @test ind == [1, 7, 25, 51]
    # set tests for slicing grid from atomic grid
    for i in 1:3
        # set each layer of points
        ref_grid = AngularGrid(degree=degs[i])
        # check for each point
        @test isapprox(pts[ind[i]:ind[i+1]-1, :], ref_grid.points .* rad_pts[i])
        # check for each weight
        @test isapprox(
            wts[ind[i]:ind[i+1]-1],
            ref_grid.weights .* rad_wts[i] .* rad_pts[i] .^ 2,
        )
    end
end

function test_atomic_grid()
    # Test atomic grid center translation.
    rad_pts = [0.1, 0.5, 1]
    rad_wts = [0.3, 0.4, 0.3]
    rad_grid = OneDGrid(rad_pts, rad_wts)
    degs = [2, 5, 7]
    # origin center
    # randome center
    pts, wts, ind, _ = _generate_atomic_grid(rad_grid, degs)
    ref_pts, ref_wts, ref_ind, _ = _generate_atomic_grid(rad_grid, degs)
    # diff grid points diff by center and same weights
    @test isapprox(pts, ref_pts)
    @test isapprox(wts, ref_wts)
    # @test isapprox(target_grid.center + ref_center, ref_grid.center)
end


function test_get_shell_grid(use_spherical)
    rad_pts = [0.1, 0.5, 1]
    rad_wts = [0.3, 0.4, 0.3]
    rad_grid = OneDGrid(rad_pts, rad_wts)
    degs = [3, 5, 7]
    atgrid = AtomGrid(rad_grid, degrees=degs, use_spherical=use_spherical)
    @test atgrid.n_shells == 3
    # grep shell with r^2
    for i in 1:atgrid.n_shells
        sh_grid = get_shell_grid(atgrid, i)
        @test sh_grid isa AngularGrid
        ref_grid = AngularGrid(degree=degs[i], use_spherical=use_spherical)
        @test isapprox(sh_grid.points, ref_grid.points .* rad_pts[i])
        @test isapprox(sh_grid.weights, ref_grid.weights .* rad_wts[i] .* rad_pts[i] .^ 2)
    end
    # grep shell without r^2
    for i in 1:atgrid.n_shells
        sh_grid = get_shell_grid(atgrid, i, r_sq=false)
        @test sh_grid isa AngularGrid
        ref_grid = AngularGrid(degree=degs[i], use_spherical=use_spherical)
        @test isapprox(sh_grid.points, ref_grid.points .* rad_pts[i])
        @test isapprox(sh_grid.weights, ref_grid.weights .* rad_wts[i])
    end
end

function test_convert_points_to_sph()
    rad_pts = [0.1, 0.5, 1]
    rad_wts = [0.3, 0.4, 0.3]
    rad_grid = OneDGrid(rad_pts, rad_wts)
    center = rand(3)
    atgrid = AtomGrid(rad_grid, degrees=[7], center=center)
    points = rand(100, 3)
    calc_sph = convert_cartesian_to_spherical(atgrid, points)
    # compute ref result
    ref_coor = broadcast(-, points, reshape(center, 1, 3))
    r = vec(sqrt.(sum(abs2, ref_coor, dims=2)))
    phi = acos.(ref_coor[:, 3] ./ r)
    theta = atan.(ref_coor[:, 2], ref_coor[:, 1])
    @test isapprox(hcat(r, theta, phi), calc_sph)
    @test size(calc_sph) == (100, 3)

    # test single point
    point = rand(3)
    calc_sph = convert_cartesian_to_spherical(atgrid, point)
    ref_coor = point - center
    r = sqrt(sum(ref_coor .^ 2))
    theta = atan(ref_coor[2], ref_coor[1])
    phi = acos(ref_coor[3] / r)
    @test isapprox(reshape([r, theta, phi], (1, 3)), calc_sph)
end

function test_spherical_complete(use_spherical)
    num_pts = length(LEBEDEV_DEGREES)
    pts = UniformInteger(num_pts)
    for _ in 1:10
        start = rand() * 1e-5
        end_ = rand() * 10 + 10
        tf = PowerRTransform(start, end_)
        rad_grid = transform_1d_grid(tf, pts)
        atgrid = AtomGrid(
            rad_grid,
            degrees=collect(keys(LEBEDEV_DEGREES)),
            use_spherical=use_spherical,
        )
        values = rand(length(LEBEDEV_DEGREES))
        pt_val = zeros(atgrid.size)
        for (index, value) in enumerate(values)
            pt_val[atgrid.indices[index]:atgrid.indices[index+1]-1] .= value
            rad_int_val = (
                value * rad_grid.weights[index] * 4 * π * rad_grid.points[index]^2
            )
            atgrid_int_val = sum(
                pt_val[atgrid.indices[index]:atgrid.indices[index+1]-1]
                .*
                atgrid.weights[atgrid.indices[index]:atgrid.indices[index+1]-1]
            )
            @test isapprox(rad_int_val, atgrid_int_val)
        end
        ref_int_at = integrate(atgrid, pt_val)
        ref_int_rad = integrate(rad_grid, 4 * π * rad_grid.points .^ 2 .* values)
        @test isapprox(ref_int_at, ref_int_rad)
    end
end


function helper_func_gauss(points; center=nothing)
    """Compute gauss function value for test interpolation."""
    if isnothing(center)
        center = [0.0, 0.0, 0.0]
    end
    pts = points .- reshape(center, :, 3)
    x, y, z = pts[:, 1], pts[:, 2], pts[:, 3]

    return exp.(-(x .^ 2)) .* exp.(-(y .^ 2)) .* exp.(-(z .^ 2))
end

function helper_func_power(points; deriv=false)
    """Compute function value for test interpolation."""
    if deriv == true
        deriv = zeros(size(points, 1), 3)
        deriv[:, 1] = 4.0 * points[:, 1]
        deriv[:, 2] = 6.0 * points[:, 2]
        deriv[:, 3] = 8.0 * points[:, 3]
        return deriv
    end
    return 2 * points[:, 1] .^ 2 + 3 * points[:, 2] .^ 2 + 4 * points[:, 3] .^ 2
end

function helper_func_power_deriv(points)
    """Compute function derivative for test derivation."""
    # r = vecnorm(points, dims=2)
    r = vec(sqrt.(sum(abs2, points, dims=2)))
    dxf = 4 * points[:, 1] .* points[:, 1] ./ r
    dyf = 6 * points[:, 2] .* points[:, 2] ./ r
    dzf = 8 * points[:, 3] .* points[:, 3] ./ r
    return dxf + dyf + dzf
end

function test_integrating_angular_components_spherical(use_spherical)
    """Test integrating angular components of a spherical harmonics of maximum degree 3."""
    odg = OneDGrid([0.0, 1e-16, 1e-8, 1e-4, 1e-2], ones(5), (0, Inf))
    atom_grid = AtomGrid(odg, degrees=[3], use_spherical=use_spherical)
    spherical = convert_cartesian_to_spherical(atom_grid)
    # Evaluate all spherical harmonics on the atomic grid points (r_i, theta_j, phi_j).
    spherical_harmonics = generate_real_spherical_harmonics(
        3, spherical[:, 2], spherical[:, 3]  # theta, phi points
    )
    # Convert to three-dimensional array (Degrees, Order, Points)
    spherical_array = zeros((3, 2 * 3 + 1, size(atom_grid.points, 1)))
    spherical_array[1, 1, :] = spherical_harmonics[1, :]  # (l,m) = (0,0)
    spherical_array[2, 1, :] = spherical_harmonics[2, :]  # = (1, 0)
    spherical_array[2, 2, :] = spherical_harmonics[3, :]  # = (1, 1)
    spherical_array[2, 3, :] = spherical_harmonics[4, :]  # = (1, -1)
    spherical_array[3, 1, :] = spherical_harmonics[5, :]  # = (2, 0)
    spherical_array[3, 2, :] = spherical_harmonics[6, :]  # = (2, 2)
    spherical_array[3, 3, :] = spherical_harmonics[7, :]  # = (2, 1)
    spherical_array[3, 4, :] = spherical_harmonics[8, :]  # = (2, -2)
    spherical_array[3, 5, :] = spherical_harmonics[9, :]  # = (2, -1)

    integral = integrate_angular_coordinates(atom_grid, spherical_array)
    @test size(integral) == (3, 2 * 3 + 1, 5)
    # Assert that all spherical harmonics except when l=0,m=0 are all zero.
    @test all(isapprox.(integral[1, 1, :], sqrt(4.0 * pi)))
    @test all(isapprox.(integral[1, 2:end, :], 0.0))
    @test all(isapprox.(integral[2:end, :, :], 0.0, atol=1e-7))
end


function test_integrating_angular_components_with_gaussian_projected_to_spherical_harmonic(numb_rads, degs)
    odg = OneDGrid(collect(range(0.0, stop=1.0, length=numb_rads)), ones(numb_rads), (0, Inf))
    atom_grid = AtomGrid(odg, degrees=[degs])
    func_vals = helper_func_gauss(atom_grid.points)

    # Generate spherical harmonic basis
    sph_coords = convert_cartesian_to_spherical(atom_grid)
    theta, phi = sph_coords[:, 2], sph_coords[:, 3]
    degrees = div(degs, 2)
    basis = generate_real_spherical_harmonics(degrees, theta, phi)

    # Multiply spherical harmonic basis with the Gaussian function values to project.
    values = basis .* reshape(func_vals, 1, :)

    # Take the integral of the projection of the Gaussian onto spherical harmonic basis.
    integrals = integrate_angular_coordinates(atom_grid, values)

    # Since the Gaussian is spherical, then this is just the integral of a spherical harmonic
    # thus whenever l != 0, we have it is zero everywhere.
    @test all(isapprox.(integrals[2:end, :], 0.0, atol=1e-6))

    # Integral of e^(-r^2) * int sin(theta) dtheta dphi / (sqrt(4 pi))
    @test all(isapprox.(integrals[1, :], exp.(-atom_grid.rgrid.points .^ 2.0) .* (4.0 * π)^0.5))
end

function test_integrating_angular_components_with_offcentered_gaussian()
    numb_rads, degs = 50, 10

    # Go from 1e-3, since it is zero
    odg = OneDGrid(collect(range(1e-3, stop=1.0, length=numb_rads)), ones(numb_rads), (0, Inf))
    atom_grid = AtomGrid(odg, degrees=[degs])

    # Since it is off-centered, one of the l=1, (p-z orbital) would be non-zero
    center = [0.0, 0.0, 0.1]
    func_vals = helper_func_gauss(atom_grid.points, center=center)

    # Generate spherical harmonic basis
    sph_coords = convert_cartesian_to_spherical(atom_grid)
    theta, phi = sph_coords[:, 2], sph_coords[:, 3]
    degrees = div(degs, 2)
    basis = generate_real_spherical_harmonics(degrees, theta, phi)

    # Multiply spherical harmonic basis with the Gaussian function values to project.
    values = basis .* reshape(func_vals, 1, :)

    # Take the integral of the projection of the Gaussian onto spherical harmonic basis.
    integrals = integrate_angular_coordinates(atom_grid, values)

    # Integral of e^(-r^2) * int sin(θ) dθ dφ / (sqrt(4π))
    @test all(integrals[2, :] .> 1e-5)  # pz-orbital should be non-zero
    @test all(integrals[3, :] .< 1e-5)  # px, py-orbital should be zero
    @test all(integrals[4, :] .< 1e-5)
end


function test_spherical_average_of_gaussian(use_spherical)
    """Test spherical average of a Gaussian (radial) function is itself and its integral."""

    # construct helper function
    function func(sph_points)
        return exp.(-sph_points[:, 1] .^ 2.0)
    end

    # Construct Radial Grid and atomic grid with spherical harmonics of degree 10
    #   for all points.
    oned_grid = collect(0.0:0.001:5.0)
    rad = OneDGrid(oned_grid, ones(length(oned_grid)), (0, Inf))
    atgrid = AtomGrid(rad, degrees=[5], use_spherical=use_spherical)
    spherical_pts = convert_cartesian_to_spherical(atgrid, atgrid.points)
    func_values = func(spherical_pts)
    spherical_avg = spherical_average(atgrid, func_values)
    # Test that the spherical average of a Gaussian is itself
    numb_pts = 1000
    random_rad_pts = rand(Uniform(0, 1.5), numb_pts, 3)
    # random_rad_pts = rand(Uniform(0.0, 1.5), numb_pts, 3)
    spherical_avg2 = spherical_avg(random_rad_pts[:, 1])
    func_vals = func(random_rad_pts)
    @test all(isapprox.(spherical_avg2, func_vals, atol=1e-4))

    # Test the integral of spherical average is the integral of Gaussian e^(-x^2)e^(-y^2)...
    #   from -infinity to infinity which is equal to pi^(3/2)
    NUMPY = pyimport("numpy")
    integral = 4.0 * π * NUMPY.trapz(spherical_avg(oned_grid) .* oned_grid .^ 2.0, x=oned_grid)
    actual_integral = sqrt(π)^3.0
    @test isapprox(actual_integral, integral)
end

function test_spherical_average_of_spherical_harmonic()
    """Test spherical average of spherical harmonic is zero."""

    # construct helper function
    function func(sph_points)
        # Spherical harmonic of order 6 and magnetic 0
        r, phi, theta = sph_points[:, 1], sph_points[:, 2], sph_points[:, 3]
        return sqrt(13) / (sqrt(π) * 32) * (231 * cos.(theta) .^ 6.0 .- 315 * cos.(theta) .^ 4.0 .+ 105 * cos.(theta) .^ 2.0 .- 5.0)
    end

    # Construct Radial Grid and atomic grid with spherical harmonics of degree 10
    #   for all points.
    oned_grid = collect(0.0:0.001:5.0)
    rad = OneDGrid(oned_grid, ones(length(oned_grid)), (0, Inf))
    atgrid = AtomGrid(rad, degrees=[10])
    spherical_pts = convert_cartesian_to_spherical(atgrid, atgrid.points)
    func_values = func(spherical_pts)

    spherical_avg = spherical_average(atgrid, func_values)

    # Test that the spherical average of a spherical harmonic is zero.
    numb_pts = 1000
    random_rad_pts = rand(Uniform(0.02, π), numb_pts, 3)
    spherical_avg2 = spherical_avg(random_rad_pts[:, 1])
    @test all(isapprox.(spherical_avg2, 0.0, atol=1e-4))
end


function test_fitting_spherical_harmonics(use_spherical)
    max_degree = 10  # Maximum degree
    rad = OneDGrid(collect(range(0.0, stop=1.0, length=10)), ones(10), (0, Inf))
    atom_grid = AtomGrid(rad, degrees=[max_degree], use_spherical=use_spherical)
    max_degree = atom_grid.l_max
    spherical = convert_cartesian_to_spherical(atom_grid)

    # Evaluate all spherical harmonics on the atomic grid points (r_i, theta_j, phi_j).
    spherical_harmonics = generate_real_spherical_harmonics(
        max_degree, spherical[:, 2], spherical[:, 3]  # theta, phi points
    )

    i = 1
    # Go through each spherical harmonic up to max_degree // 2 and check if projection
    # for its radial component is one and the rest are all zeros.
    for l_value in 0:max_degree÷2
        for _m in 1:2*l_value+1
            spherical_harm = spherical_harmonics[i, :]
            radial_components = radial_component_splines(atom_grid, spherical_harm)
            @test length(radial_components) == (atom_grid.l_max ÷ 2 + 1)^2.0

            radial_pts = collect(0.0:0.01:1.0)
            # Go through each (l, m)
            for (j, radial_comp) in enumerate(radial_components)
                # If the current index is j, then projection of spherical harmonic
                # onto itself should be all ones, else they are all zero.
                if i == j
                    @test all(isapprox.(radial_comp(radial_pts), 1.0))
                else
                    @test all(isapprox.(radial_comp(radial_pts), 0.0, atol=1e-8))
                end
            end
            i += 1
        end
    end
end

function test_interpolation_of_gaussian(center)
    oned = GaussLegendre(350)
    btf = BeckeRTransform(0.0001, 1.5)
    rad = transform_1d_grid(btf, oned)
    atgrid = AtomGrid(rad, degrees=[31], use_spherical=false)
    value_array = helper_func_gauss(atgrid.points)
    npt = 1000
    r = rand(Uniform(0.01, maximum(rad.points)), npt)
    theta = rand(Uniform(0, pi), npt)
    phi = rand(Uniform(0, 2.0 * pi), npt)
    x = r .* sin.(phi) .* cos.(theta)
    y = r .* sin.(phi) .* sin.(theta)
    z = r .* cos.(phi)
    input_points = hcat(x, y, z)
    interfunc = interpolate(atgrid, value_array)
    result = helper_func_gauss(input_points, center=center)
    expected = interfunc(input_points)
    @test all(isapprox.(result, expected, atol=1e-4))
end


function test_cubicspline_and_interp_mol(use_spherical)
    # "Test cubicspline interpolation values."
    odg = OneDGrid(collect(1:10), ones(10), (0, Inf))
    rad = transform_1d_grid(IdentityRTransform(), odg)
    atgrid = from_pruned(rad, radius=1, sectors_r=[], sectors_degree=[7], use_spherical=use_spherical)

    values = helper_func_power(atgrid.points)
    # spls = fit_values(atgrid, values)

    for i in 1:10
        inter_func = interpolate(atgrid, values)
        # same result from points and interpolation
        expected = values[atgrid.indices[i]:atgrid.indices[i+1]-1]
        result = inter_func(atgrid.points[atgrid.indices[i]:atgrid.indices[i+1]-1, :])
        @test all(isapprox.(result, expected, atol=1e-7))
    end
end

function test_cubicspline_and_interp(use_spherical)
    # "Test cubicspline interpolation values."
    odg = OneDGrid(collect(1:10) .+ 1, ones(10), (0, Inf))
    rad_grid = transform_1d_grid(IdentityRTransform(), odg)
    for _ in 1:10
        degree = rand(5:20)
        atgrid = from_pruned(rad_grid, radius=1, sectors_r=[], sectors_degree=[degree], use_spherical=use_spherical)
        values = helper_func_power(atgrid.points)
        # spls = fit_values(atgrid, values)

        for i in 1:10
            inter_func = interpolate(atgrid, values)
            interp = inter_func(atgrid.points[atgrid.indices[i]:atgrid.indices[i+1]-1, :])
            # same result from points and interpolation
            @test all(isapprox.(interp, values[atgrid.indices[i]:atgrid.indices[i+1]-1]))
        end

        # test random x, y, z
        for _ in 1:10
            xyz = rand(10, 3) .* rand(1:6)
            ref_value = helper_func_power(xyz)

            interp_func = interpolate(atgrid, values)
            @test all(isapprox.(interp_func(xyz), ref_value))
        end
    end
end

function test_interpolate_and_its_derivatives_on_polynomial(use_spherical)
    # "Test interpolation of derivative of polynomial function."
    odg = OneDGrid(collect(range(0.01, stop=10, length=50)), ones(50), (0, Inf))
    rad = transform_1d_grid(IdentityRTransform(), odg)
    for _ in 1:10
        degree = rand(5:20)
        atgrid = from_pruned(
            rad,
            radius=1,
            sectors_r=[],
            sectors_degree=[degree],
            use_spherical=use_spherical,
        )
        values = helper_func_power(atgrid.points)
        # spls = fit_values(atgrid, values)

        for i in 1:10
            points = atgrid.points[atgrid.indices[i]:atgrid.indices[i+1]-1, :]
            interp_func = interpolate(atgrid, values)
            result = interp_func(points, deriv=1)
            # same result from points and interpolation
            ref_deriv = helper_func_power(points, deriv=1)
            # display(ref_deriv)
            @test all(isapprox.(result, ref_deriv, atol=1e-7))
        end

        # test random x, y, z with fd
        for _ in 1:10
            xyz = rand(10, 3) .* rand(1:6)
            ref_value = helper_func_power(xyz, deriv=1)
            interp_func = interpolate(atgrid, values)
            interp = interp_func(xyz, deriv=1)
            @test all(isapprox.(interp, ref_value))
        end

        # test random x, y, z with fd
        for _ in 1:10
            xyz = rand(10, 3) .* rand(1:6)
            ref_value = helper_func_power_deriv(xyz)
            interp_func = interpolate(atgrid, values)
            interp = interp_func(xyz, deriv=1, only_radial_deriv=true)
            @test all(isapprox.(interp, ref_value))
        end

    end
end

function test_cartesian_moment_integral_with_gaussian_upto_order_1()
    # "Test Cartesian moment integral of Gaussian up to order 1."
    # The moment integral is computed analytically with wolframalpha in one-dimension.
    # Generate atomic grid.
    oned = GaussLegendre(10000)
    btf = BeckeRTransform(0.0001, 0.1)
    rad = transform_1d_grid(btf, oned)
    atgrid = from_pruned(rad, radius=1, sectors_r=[], sectors_degree=[7])

    # Create Gaussian function
    func_vals = helper_func_gauss(atgrid.points, center=[0.175, 0.25, 0.15])

    # Consider two centers.
    orders = 1
    centers = [0.0 0.0 0.0; 0.1 0.1 0.3]
    result = moments(atgrid, orders, centers, func_vals)
    # Test Cartesian order: (0, 0, 0), which is integral e^{-(x - c)^2} in x-dim
    @test isapprox(result[1, 1], π^1.5, atol=1e-4)
    @test isapprox(result[1, 2], π^1.5, atol=1e-4)

    # Test Cartesian order: (1, 0, 0), which is integral (x - X_c) e^{-(x-c)^2}
    #  Wolfram: integral (x - c) e^(-(x - d)^2)  = sqrt(pi) (d - c)
    @test isapprox(result[2, 1], π^1.5 * 0.175, atol=1e-5)
    @test isapprox(result[2, 2], π^1.5 * (0.175 - 0.1), atol=1e-5)

    # Test (0, 0, 1)
    @test isapprox(result[4, 1], (π^1.5) * 0.15, atol=1e-3)
    @test isapprox(result[4, 2], (π^1.5) * (0.15 - 0.3), atol=1e-3)

    # @test_throws TypeError begin
    #     # orders should be integer
    #     moments(atgrid, [1, 1], centers, func_vals)
    # end
    # @test_throws ValueError begin
    #     multidim_f = [func_vals, func_vals]
    #     moments(atgrid, 1, centers, multidim_f)  # func_vals should be ndim = 1
    #     # centers should be ndim =2
    #     moments(atgrid, 1, [1, 1, 1], func_vals)
    #     moments(atgrid, 1, [[1, 1]], func_vals)  # centers should be dim=3
    #     moments(atgrid, 0, centers, func_vals, type_mom="pure_radial")  # l>0
    #     # func_vals too little points
    #     moments(atgrid, 1, centers, [1, 2, 3])
    # end
end

function test_pure_moment_integral_with_identity_function()
    # "Test pure moment integral with identify function is mostly all zeros."
    center = [0.0 0.0 0.0;]
    oned = GaussLaguerre(15)
    atgrid = AtomGrid(oned, degrees=[50])
    func_vals = ones(size(atgrid.points, 1))
    result = moments(atgrid, orders=2, centers=center, func_vals=func_vals, type_mom="pure")
    @test all(isapprox.(result[2:end], 0.0, atol=1e-3))
end

using Test

function test_pure_moment_integrals_with_gaussian_upto_order_5()
    # "Test pure multipole moment integral with Gaussian upto order 5."
    center = [0.0 0.5 1.0;]
    # Obtained this from Horton on atomgrid
    horton_answer = [
        5.56832800,
        -5.56832800,
        8.75535482e-18,
        -2.78416400,
        4.87228700,
        1.96316697e-18,
        4.82231350,
        -1.20557838,
        -1.26575073e-17,
        -3.48020500,
        1.19286145e-16,
        -6.39354483,
        2.69575520,
        5.23442037e-17,
        -4.68564343e-17,
        5.50268726e-01,
        1.52258969,
        -3.26136843e-13,
        7.15349344,
        -4.47463560,
        -2.80536246e-13,
        -1.40807305e-12,
        -1.45587420,
        2.57364630e-01,
        -1.56214452e-13,
        7.39543562e-01,
        1.35462566e-12,
        -6.82363035,
        6.24076062,
        2.67670790e-12,
        5.05401764e-12,
        2.82075627,
        -7.72093891e-01,
        3.45622696e-12,
        -3.70252405e-13,
        -1.22078763e-01,
    ]
    # Generate atomic grid.
    oned = GaussLaguerre(50)
    atgrid = AtomGrid(oned, degrees=[50])
    order = 5
    r = vec(sqrt.(sum(abs2, atgrid.points, dims=2)))
    gaussian = exp.(-r .^ 2.0)
    result = moments(atgrid, orders=order, centers=center, func_vals=gaussian, type_mom="pure")
    @test all(isapprox.(result[:], horton_answer, atol=1e-6))
end


function test_pure_radial_moments_of_identity_function_against_pure_moments()
    # "Test pure-radial multipole moments with identity function against pure moments."
    center = [0.0 0.0 0.0;]
    # Generate atomic grid.
    oned = GaussLaguerre(100)
    atgrid = AtomGrid(oned, degrees=[50])
    order = 3
    ident_func = ones(size(atgrid.points, 1))
    result, orders = moments(atgrid, order, center, ident_func, "pure-radial", true)

    radial = convert_cartesian_to_spherical(atgrid, atgrid.points, center[1, :])[:, 1]
    # Go through each (n, deg, m)
    for (i, row) in enumerate(eachrow(orders))
        n, deg, ord = row
        index = deg^2 + 2 * ord - 1
        if ord > 0
            index = deg^2 + 2 * ord - 1
        else
            index = deg^2 - 2 * ord
        end
        index += 1 # for julia
        # Integrate Y_l^m r^l f(x) where f(x)=r^n
        desired, orders = moments(atgrid, deg, center, radial .^ n, "pure", true)
        @test isapprox(desired[index], result[i], atol=1e-4, rtol=1e-7)
    end
end

function test_radial_moments_of_gaussian_against_horton()
    # "Test radial moments of Gausian against theochem/horton."
    center = [0.0 0.5 0.0;]
    # Generate atomic grid.
    oned = GaussLaguerre(100)
    atgrid = AtomGrid(oned, degrees=[50])
    horton_answer = [5.568328, 6.79414003, 9.744574, 15.78558743]
    order = 3
    r = vec(sqrt.(sum(abs2, atgrid.points, dims=2)))
    gaussian = exp.(-r .^ 2.0)
    result = moments(atgrid, orders=order, centers=center, func_vals=gaussian, type_mom="radial")
    @test all(isapprox.(result[:, 1], horton_answer, atol=1e-4))
end

function test_pure_radial_moments_of_spherical_harmonics()
    # "Test pure-radial multipole moments with spherical harmonics."
    center = [0.0 0.0 0.0;]
    # Generate atomic grid.
    oned = GaussLaguerre(20)
    atgrid = AtomGrid(oned, degrees=[50])
    order = 5
    sph_pts = convert_cartesian_to_spherical(atgrid, atgrid.points)
    _, theta, phi = sph_pts[:, 1], sph_pts[:, 2], sph_pts[:, 3]
    spherical = generate_real_spherical_harmonics(order, theta, phi)

    i_sph = 1
    for (l_sph, m_sph) in [(0, 0), (1, 0), (1, 1), (1, -1), (2, 0), (2, 1), (2, -1), (2, 2), (2, -2)]
        result, orders = moments(atgrid, order ÷ 2, center, spherical[i_sph, :], "pure-radial", true)
        for (i_mom, (_n_mom, l_mom, m_mom)) in enumerate(eachrow(orders))
            # If the spherical harmonics match, then the integral over sph_coords is one
            # then we are left with a diverging integral.
            if l_mom == l_sph && m_sph == m_mom
                @test result[i_mom] > 1000
            else
                # If the spherical harmonics don't match, then the integral over sph coords
                # is zero.
                @test isapprox(result[i_mom], 0.0, atol=1e-3) # TODO: the is 1e-5 in qc-grid
            end
        end
        i_sph += 1
    end
end



@testset "AtomGrid.jl" begin
    test_total_atomic_grid()
    test_from_predefined()
    test_from_pruned_with_degs_and_size()
    test_find_l_for_rad_list()
    test_generate_atomic_grid()
    test_atomic_grid()
    test_get_shell_grid(false)
    test_get_shell_grid(true)
    test_convert_points_to_sph()
    test_spherical_complete(false)
    test_spherical_complete(true)
    test_integrating_angular_components_spherical(false)
    test_integrating_angular_components_spherical(true)
    test_integrating_angular_components_with_gaussian_projected_to_spherical_harmonic(10, 2)
    test_integrating_angular_components_with_gaussian_projected_to_spherical_harmonic(10, 5)
    test_integrating_angular_components_with_gaussian_projected_to_spherical_harmonic(10, 10)
    test_integrating_angular_components_with_offcentered_gaussian()
    test_spherical_average_of_gaussian(false)
    test_spherical_average_of_gaussian(true)
    test_spherical_average_of_spherical_harmonic()
end

@testset "Fitting Spherical Harmonics" begin
    test_fitting_spherical_harmonics(false)
    test_fitting_spherical_harmonics(true)
end

@testset "Interpolation" begin
    centers_values = [0.0 0.0 0.0; 1e-2 0.0 0.0; 1e-2 0.0 -1e-2]
    for center in eachrow(centers_values)
        test_interpolation_of_gaussian(center)
    end
end


@testset "test_cubicspline_and_interp_mol" begin
    test_cubicspline_and_interp_mol(false)
    test_cubicspline_and_interp_mol(true)
    test_cubicspline_and_interp(false)
    test_cubicspline_and_interp(true)
    test_interpolate_and_its_derivatives_on_polynomial(false)
end


@testset "moment" begin
    test_cartesian_moment_integral_with_gaussian_upto_order_1()
    test_pure_moment_integral_with_identity_function()
    test_pure_moment_integrals_with_gaussian_upto_order_5()
    test_pure_radial_moments_of_identity_function_against_pure_moments()
    test_radial_moments_of_gaussian_against_horton()
    test_pure_radial_moments_of_spherical_harmonics()
end