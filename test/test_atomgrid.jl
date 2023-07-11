using LinearAlgebra
using Test
using IntegralGrid.BaseGrid, IntegralGrid.OnedGrid, IntegralGrid.RTransform, IntegralGrid.AtomicGrid, IntegralGrid.Angular

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

@testset "Conversion Tests" begin
    # @testset "test_get_shell_grid" begin
    #     test_get_shell_grid(false)
    #     test_get_shell_grid(true)
    # end

    # @testset "test_convert_points_to_sph" begin
    #     test_convert_points_to_sph()
    # end

    @testset "test_spherical_complete" begin
        test_spherical_complete(false)
        test_spherical_complete(true)
    end
end


# @testset "Utils.jl" begin
#     test_total_atomic_grid()
#     test_from_predefined() 
#     test_from_pruned_with_degs_and_size() 
#     test_find_l_for_rad_list() 
#     test_generate_atomic_grid() 
#     test_atomic_grid() 
# end