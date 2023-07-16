using LinearAlgebra
using Test
using Distributions
using IntegralGrid.BaseGrid
using IntegralGrid.AtomicGrid
using IntegralGrid.OnedGrid
using IntegralGrid.RTransform
using IntegralGrid.Becke
using IntegralGrid.Hirshfeld
using IntegralGrid.MolecularGrid
using IntegralGrid.Utils

function test_integrate_hydrogen_single_1s(rgrid)
    # Test molecular integral in H atom
    coordinates = [0.0, 0.0, -0.5]
    atg1 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates
    )
    becke = BeckeWeights(nothing, 3)
    mg = MolGrid([1], [atg1], becke)
    dist0 = [norm(coordinates .- vec(point)) for point in eachrow(mg.points)]
    # dist0 = sqrt.(sum(abs2, coordinates[newaxis, :] .- mg.points, dims=2))[:, 1]
    fn = exp.(-2 * dist0) / π
    occupation = integrate(mg, fn)
    @test isapprox(occupation, 1.0, atol=1e-6)
end

function test_make_grid_integral()
    # Test molecular make_grid works as designed
    pts = UniformInteger(70)
    tf = ExpRTransform(1e-5, 2e1)
    rgrid = transform_1d_grid(tf, pts)
    numbers = [1, 1]
    coordinates = [0.0 0.0 -0.5; 0.0 0.0 0.5]
    becke = BeckeWeights(nothing, 3)

    for (grid_type, deci) in [
        ("coarse", 3),
        ("medium", 4),
        ("fine", 5),
        ("veryfine", 6),
        ("ultrafine", 6),
        ("insane", 6)
    ]
        mg = MolGrid(numbers, coordinates, rgrid, grid_type, becke)
        dist0 = [norm(coordinates[1, :] - vec(point)) for point in eachrow(mg.points)]
        dist1 = [norm(coordinates[2, :] - vec(point)) for point in eachrow(mg.points)]
        fn = exp.(-2 * dist0) / π + exp.(-2 * dist1) / π
        occupation = integrate(mg, fn)
        atol = 10.0^-deci
        @test isapprox(occupation, 2.0, atol=atol)
    end
end

function test_make_grid_different_grid_type()
    # Test different kind molgrid initizalize setting.
    # three different radial grid
    rad2 = GaussLaguerre(50)
    rad3 = GaussLaguerre(70)
    # construct grid
    numbers = [1, 8, 1]
    coordinates = [0.0 0.0 -0.5; 0.0 0.0 0.5; 0.0 0.5 0.0]
    becke = BeckeWeights(nothing, 3)

    # grid_type test with list
    mg = MolGrid(
        numbers,
        coordinates,
        rad2,
        ["fine", "veryfine", "medium"],
        becke,
        rotate=0,
        store=true,
    )
    dist0 = [norm(coordinates[1, :] - vec(point)) for point in eachrow(mg.points)]
    dist1 = [norm(coordinates[2, :] - vec(point)) for point in eachrow(mg.points)]
    dist2 = [norm(coordinates[3, :] - vec(point)) for point in eachrow(mg.points)]
    fn = (exp.(-2 * dist0) + exp.(-2 * dist1) + exp.(-2 * dist2)) / π
    occupation = integrate(mg, fn)
    @test isapprox(occupation, 3, rtol=1e-3)

    atgrid1 = from_preset(rad2, atnum=numbers[1], preset="fine", center=coordinates[1, :])
    atgrid2 = from_preset(rad2, atnum=numbers[2], preset="veryfine", center=coordinates[2, :])
    atgrid3 = from_preset(rad2, atnum=numbers[3], preset="medium", center=coordinates[3, :])
    @test all(isapprox.(mg.atgrids[1].points, atgrid1.points))
    @test all(isapprox.(mg.atgrids[2].points, atgrid2.points))
    @test all(isapprox.(mg.atgrids[3].points, atgrid3.points))

    # grid type test with dict
    mg = MolGrid(
        numbers,
        coordinates,
        rad3,
        Dict(1 => "fine", 8 => "veryfine"),
        becke,
        rotate=0,
        store=true,
    )
    dist0 = [norm(coordinates[1, :] - vec(point)) for point in eachrow(mg.points)]
    dist1 = [norm(coordinates[2, :] - vec(point)) for point in eachrow(mg.points)]
    dist2 = [norm(coordinates[3, :] - vec(point)) for point in eachrow(mg.points)]
    fn = (exp.(-2 * dist0) + exp.(-2 * dist1) + exp.(-2 * dist2)) / π
    occupation = integrate(mg, fn)
    @test isapprox(occupation, 3, rtol=1e-3)

    atgrid1 = from_preset(rad3, atnum=numbers[1], preset="fine", center=coordinates[1, :])
    atgrid2 = from_preset(rad3, atnum=numbers[2], preset="veryfine", center=coordinates[2, :])
    atgrid3 = from_preset(rad3, atnum=numbers[3], preset="fine", center=coordinates[3, :])
    @test all(isapprox.(mg.atgrids[1].points, atgrid1.points))
    @test all(isapprox.(mg.atgrids[2].points, atgrid2.points))
    @test all(isapprox.(mg.atgrids[3].points, atgrid3.points))
end

function test_make_grid_different_rad_type()
    # Test different radial grid input for make molgrid.
    # radial grid test with list
    rad1 = GaussLaguerre(30)
    rad2 = GaussLaguerre(50)
    rad3 = GaussLaguerre(70)
    # construct grid
    numbers = [1, 8, 1]
    coordinates = [0.0 0.0 -0.5; 0.0 0.0 0.5; 0.0 0.5 0.0]
    becke = BeckeWeights(nothing, 3)
    # construct molgrid
    mg = MolGrid(
        numbers,
        coordinates,
        [rad1, rad2, rad3],
        Dict(1 => "fine", 8 => "veryfine"),
        becke,
        rotate=0,
        store=true,
    )
    # dist0 = sqrt.(sum(abs2, coordinates[1] .- mg.points, dims=2))
    # dist1 = sqrt.(sum(abs2, coordinates[2] .- mg.points, dims=2))
    # dist2 = sqrt.(sum(abs2, coordinates[3] .- mg.points, dims=2))
    dist0 = [norm(coordinates[1, :] - vec(point)) for point in eachrow(mg.points)]
    dist1 = [norm(coordinates[2, :] - vec(point)) for point in eachrow(mg.points)]
    dist2 = [norm(coordinates[3, :] - vec(point)) for point in eachrow(mg.points)]
    fn = (exp.(-2 * dist0) + exp.(-2 * dist1) + exp.(-2 * dist2)) / π
    occupation = integrate(mg, fn)
    @test isapprox(occupation, 3, rtol=1e-3)

    atgrid1 = from_preset(rad1, atnum=numbers[1], preset="fine", center=coordinates[1, :])
    atgrid2 = from_preset(rad2, atnum=numbers[2], preset="veryfine", center=coordinates[2, :])
    atgrid3 = from_preset(rad3, atnum=numbers[3], preset="fine", center=coordinates[3, :])
    @test all(isapprox.(mg.atgrids[1].points, atgrid1.points))
    @test all(isapprox.(mg.atgrids[2].points, atgrid2.points))
    @test all(isapprox.(mg.atgrids[3].points, atgrid3.points))

    # radial grid test with dict
    mg = MolGrid(
        numbers,
        coordinates,
        Dict(1 => rad1, 8 => rad3),
        Dict(1 => "fine", 8 => "veryfine"),
        becke,
        rotate=0,
        store=true,
    )
    # dist0 = sqrt.(sum(abs2, coordinates[1] .- mg.points, dims=2))
    # dist1 = sqrt.(sum(abs2, coordinates[2] .- mg.points, dims=2))
    # dist2 = sqrt.(sum(abs2, coordinates[3] .- mg.points, dims=2))
    dist0 = [norm(coordinates[1, :] - vec(point)) for point in eachrow(mg.points)]
    dist1 = [norm(coordinates[2, :] - vec(point)) for point in eachrow(mg.points)]
    dist2 = [norm(coordinates[3, :] - vec(point)) for point in eachrow(mg.points)]
    fn = (exp.(-2 * dist0) + exp.(-2 * dist1) + exp.(-2 * dist2)) / π
    occupation = integrate(mg, fn)
    @test isapprox(occupation, 3, rtol=1e-3)

    atgrid1 = from_preset(rad1, atnum=numbers[1], preset="fine", center=coordinates[1, :])
    atgrid2 = from_preset(rad3, atnum=numbers[2], preset="veryfine", center=coordinates[2, :])
    atgrid3 = from_preset(rad1, atnum=numbers[3], preset="fine", center=coordinates[3, :])
    @test all(isapprox.(mg.atgrids[1].points, atgrid1.points))
    @test all(isapprox.(mg.atgrids[2].points, atgrid2.points))
    @test all(isapprox.(mg.atgrids[3].points, atgrid3.points))
end

function test_integrate_hydrogen_pair_1s(rgrid)
    """Test molecular integral in H2."""
    coordinates = [0.0 0.0 -0.5; 0.0 0.0 0.5]
    atg1 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[1, :],
    )
    atg2 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[2, :],
    )
    becke = BeckeWeights(nothing, 3)
    mg = MolGrid([1, 1], [atg1, atg2], becke)
    # dist0 = sqrt.(sum(abs2, coordinates[1] .- mg.points, dims=2))
    # dist1 = sqrt.(sum(abs2, coordinates[2] .- mg.points, dims=2))
    dist0 = [norm(coordinates[1, :] - vec(point)) for point in eachrow(mg.points)]
    dist1 = [norm(coordinates[2, :] - vec(point)) for point in eachrow(mg.points)]
    fn = exp.(-2 * dist0) / π + exp.(-2 * dist1) / π
    occupation = integrate(mg, fn)
    @test isapprox(occupation, 2.0, atol=1e-6)
end

function test_integrate_hydrogen_trimer_1s(rgrid)
    """Test molecular integral in H3."""
    coordinates = [0.0 0.0 -0.5; 0.0 0.0 0.5; 0.0 0.5 0.0]
    atg1 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[1, :],
    )
    atg2 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[2, :],
    )
    atg3 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[3, :],
    )
    becke = BeckeWeights(nothing, 3)
    mg = MolGrid([1, 1, 1], [atg1, atg2, atg3], becke)
    # dist0 = sqrt.(sum(abs2, coordinates[1] .- mg.points, dims=2))
    # dist1 = sqrt.(sum(abs2, coordinates[2] .- mg.points, dims=2))
    # dist2 = sqrt.(sum(abs2, coordinates[3] .- mg.points, dims=2))
    dist0 = [norm(coordinates[1, :] - vec(point)) for point in eachrow(mg.points)]
    dist1 = [norm(coordinates[2, :] - vec(point)) for point in eachrow(mg.points)]
    dist2 = [norm(coordinates[3, :] - vec(point)) for point in eachrow(mg.points)]
    fn = exp.(-2 * dist0) / π + exp.(-2 * dist1) / π + exp.(-2 * dist2) / π
    occupation = integrate(mg, fn)
    @test isapprox(occupation, 3.0, atol=1e-4)
end

function test_integrate_hydrogen_8_1s(rgrid)
    """Test molecular integral in H2."""
    centers = [-0.5 -0.5 -0.5; -0.5 -0.5 0.5; 0.5 -0.5 -0.5; 0.5 -0.5 0.5; -0.5 0.5 -0.5; -0.5 0.5 0.5; 0.5 0.5 -0.5; 0.5 0.5 0.5]
    atgs = [
        from_pruned(
            rgrid,
            radius=0.5,
            sectors_r=[],
            sectors_degree=[17],
            center=center,
        )
        for center in eachrow(centers)
    ]

    becke = BeckeWeights(nothing, 3)
    mg = MolGrid(fill(1, size(centers, 1)), atgs, becke)
    fn = zeros(Float64, mg.size)
    for center in eachrow(centers)
        # dist = sqrt.(sum(abs2, center .- mg.points, dims=2))
        dist = [norm(center - point) for point in eachrow(mg.points)]
        fn += exp.(-2 * dist) / π
    end
    occupation = integrate(mg, fn)
    @test isapprox(occupation, size(centers, 1), rtol=1e-2)
end

function test_molgrid_attrs_subgrid(rgrid)
    """Test sub atomic grid attributes."""
    # numbers = np.array([6, 8], int)
    coordinates = [0.0 0.2 -0.5; 0.1 0.0 0.5]
    atg1 = from_pruned(
        rgrid,
        radius=1.228,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[1, :],
    )
    atg2 = from_pruned(
        rgrid,
        radius=0.945,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[2, :],
    )
    becke = BeckeWeights(nothing, 3)
    mg = MolGrid([6, 8], [atg1, atg2], becke, store=true)
    # mg = BeckeMolGrid(coordinates, numbers, None, (rgrid, 110), mode='keep')

    @test mg.size == 2 * 110 * 100
    @test size(mg.points) == (mg.size, 3)
    @test size(mg.weights) == (mg.size,)
    @test size(mg.aim_weights) == (mg.size,)
    @test length(mg.indices) == 2 + 1

    for i in 1:2
        atgrid = mg[i]
        @test atgrid isa AtomGrid
        @test atgrid.size == 100 * 110
        @test size(atgrid.points) == (100 * 110, 3)
        @test size(atgrid.weights) == (100 * 110,)
        @test all(isapprox.(atgrid.center, coordinates[i, :]))
    end

    mg = MolGrid([6, 8], [atg1, atg2], becke)
    for i in 1:2
        atgrid = mg[i]
        @test atgrid isa LocalGrid
        @test all(isapprox.(atgrid.center, mg.atcoords[i, :]))
    end
end

function test_molgrid_attrs(rgrid)
    """Test MolGrid attributes."""
    # numbers = np.array([6, 8], int)
    coordinates = [0.0 0.2 -0.5; 0.1 0.0 0.5]
    atg1 = from_pruned(
        rgrid,
        radius=1.228,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[1, :],
    )
    atg2 = from_pruned(
        rgrid,
        radius=0.945,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[2, :],
    )

    becke = BeckeWeights(nothing, 3)
    mg = MolGrid([6, 8], [atg1, atg2], becke, store=true)

    @test mg.size == 2 * 110 * 100
    @test size(mg.points) == (mg.size, 3)
    @test size(mg.weights) == (mg.size,)
    @test size(mg.aim_weights) == (mg.size,)
    @test get_atomic_grid(mg, 1) === atg1
    @test get_atomic_grid(mg, 2) === atg2

    simple_ag1 = get_atomic_grid(mg, 1)
    simple_ag2 = get_atomic_grid(mg, 2)
    @test all(isapprox.(simple_ag1.points, atg1.points))
    @test all(isapprox.(simple_ag1.weights, atg1.weights))
    @test all(isapprox.(simple_ag2.weights, atg2.weights))

    # test molgrid is not stored
    mg2 = MolGrid([6, 8], [atg1, atg2], becke, store=false)
    @test mg2.atgrids == AtomGrid[]
    simple2_ag1 = get_atomic_grid(mg2, 1)
    simple2_ag2 = get_atomic_grid(mg2, 2)
    @test simple2_ag1 isa LocalGrid
    @test simple2_ag2 isa LocalGrid
    @test all(isapprox.(simple2_ag1.points, atg1.points))
    @test all(isapprox.(simple2_ag1.weights, atg1.weights, atol=1e-7))
    @test all(isapprox.(simple2_ag2.weights, atg2.weights, atol=1e-7))
end

function test_different_aim_weights_h2(rgrid)
    """Test different aim_weights for molgrid."""
    coordinates = [0.0 0.0 -0.5; 0.0 0.0 0.5]
    atg1 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[1, :],
    )
    atg2 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[2, :],
    )
    # use an array as aim_weights
    mg = MolGrid([1, 1], [atg1, atg2], ones(22000))
    dist0 = [norm(coordinates[1, :] - vec(point)) for point in eachrow(mg.points)]
    dist1 = [norm(coordinates[2, :] - vec(point)) for point in eachrow(mg.points)]
    fn = exp.(-2 * dist0) / π + exp.(-2 * dist1) / π
    occupation = integrate(mg, fn)
    @test isapprox(occupation, 4.0, atol=1e-4)
end

function test_from_size(rgrid)
    """Test horton style grid."""
    nums = [1, 1]
    coors = [0 0 -0.5; 0 0 0.5]
    becke = BeckeWeights(nothing, 3)
    mol_grid = MolGrid(nums, coors, rgrid, 110, becke, rotate=0)
    atg1 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coors[1, :],
    )
    atg2 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coors[2, :],
    )
    ref_grid = MolGrid(nums, [atg1, atg2], becke, store=true)
    @test all(isapprox.(ref_grid.points, mol_grid.points))
    @test all(isapprox.(ref_grid.weights, mol_grid.weights))
end

function test_from_pruned(rgrid)
    r"""Test MolGrid construction via from_pruned method."""
    nums = [1, 1]
    coors = [0 0 -0.5; 0 0 0.5]
    becke = BeckeWeights(nothing, 3)
    radius = [1.0, 0.5]
    sectors_r = [[0.5, 1.0, 1.5], [0.25, 0.5]]
    sectors_deg = [[3, 7, 5, 3], [3, 2, 2]]
    mol_grid = MolGrid(
        nums,
        coors,
        rgrid,
        radius,
        becke,
        sectors_r=sectors_r,
        sectors_degree=sectors_deg,
        rotate=0
    )
    atg1 = from_pruned(
        rgrid,
        radius=radius[1],
        sectors_r=sectors_r[1],
        sectors_degree=sectors_deg[1],
        center=coors[1, :],
    )
    atg2 = from_pruned(
        rgrid,
        radius=radius[2],
        sectors_r=sectors_r[2],
        sectors_degree=sectors_deg[2],
        center=coors[2, :],
    )
    ref_grid = MolGrid(nums, [atg1, atg2], becke, store=true)
    @test all(isapprox.(ref_grid.points, mol_grid.points))
    @test all(isapprox.(ref_grid.weights, mol_grid.weights))
end

function test_get_localgrid_1s(rgrid)
    """Test local grid for a molecule with one atom."""
    nums = [1]
    coords = [0.0, 0.0, 0.0]

    # initialize MolGrid with atomic grid
    atg1 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coords,
    )
    grid = MolGrid(nums, [atg1], BeckeWeights(), store=false)
    r = vec(sqrt.(sum(abs2, grid.points, dims=2)))
    fn = exp.(-2 * r)
    @test isapprox(integrate(grid, fn), π)
    # conventional local grid
    localgrid = get_localgrid(grid, coords, 12.0)
    local_r = vec(sqrt.(sum(abs2, localgrid.points, dims=2)))
    localfn = exp.(-2 * local_r)
    @test localgrid.size < grid.size
    @test localgrid.size == 10560
    @test isapprox(integrate(localgrid, localfn), π)
    @test all(isapprox.(fn[localgrid.indices], localfn))
    # "whole" local grid, useful for debugging code using local grids
    wholegrid = get_localgrid(grid, coords, Inf)
    @test wholegrid.size == grid.size
    @test all(isapprox.(wholegrid.points, grid.points))
    @test all(isapprox.(wholegrid.weights, grid.weights))
    @test all(isapprox.(wholegrid.indices, collect(1:grid.size)))

    # initialize MolGrid like horton
    grid = MolGrid(
        nums, reshape(coords, 1, 3), rgrid, 110, BeckeWeights(), store=true
    )
    r = vec(sqrt.(sum(abs2, grid.points, dims=2)))
    fn = exp.(-4.0 * r)
    @test isapprox(integrate(grid, fn), π / 8)
    localgrid = get_localgrid(grid, coords, 5.0)
    local_r = vec(sqrt.(sum(abs2, localgrid.points, dims=2)))
    localfn = exp.(-4.0 * local_r)
    @test localgrid.size < grid.size
    @test localgrid.size == 9900
    @test isapprox(integrate(localgrid, localfn), π / 8, rtol=1e-5)
    @test all(isapprox.(fn[localgrid.indices], localfn))
end

function test_get_localgrid_1s1s(rgrid)
    """Test local grid for a molecule with one atom."""
    nums = [1, 3]
    coords = [0.0 0.0 -0.5; 0.0 0.0 0.5]
    grid = MolGrid(
        nums, coords, rgrid, 110, BeckeWeights(), rotate=0, store=true
    )

    r0 = vec(sqrt.(sum(abs2, grid.points .- coords[1, :][newaxis, :], dims=2)))
    r1 = vec(sqrt.(sum(abs2, grid.points .- coords[2, :][newaxis, :], dims=2)))
    fn0 = exp.(-4.0 * r0)
    fn1 = exp.(-8.0 * r1)
    @test isapprox(integrate(grid, fn0), π / 8, rtol=1e-5)
    @test isapprox(integrate(grid, fn1), π / 64, atol=1e-8)
    # local grid centered on atom 0 to evaluate fn0
    local0 = get_localgrid(grid, coords[1, :], 5.0)
    @test local0.size < grid.size
    local_r0 = vec(sqrt.(sum(abs2, local0.points .- coords[1, :][newaxis, :], dims=2)))
    localfn0 = exp.(-4.0 * local_r0)
    @test all(isapprox.(fn0[local0.indices], localfn0))
    @test isapprox(integrate(local0, localfn0), π / 8, rtol=1e-5)
    # local grid centered on atom 1 to evaluate fn1
    local1 = get_localgrid(grid, coords[2, :], 2.5)
    @test local1.size < grid.size
    local_r1 = vec(sqrt.(sum(abs2, local1.points .- coords[2, :][newaxis, :], dims=2)))
    localfn1 = exp.(-8.0 * local_r1)
    @test isapprox(integrate(local1, localfn1), π / 64, rtol=1e-6)
    @test all(isapprox.(fn1[local1.indices], localfn1))
    # approximate the sum of fn0 and fn2 by combining results from local grids.
    fnsum = zeros(grid.size)
    fnsum[local0.indices] .+= localfn0
    fnsum[local1.indices] .+= localfn1
    @test isapprox(integrate(grid, fnsum), π * (1 / 8 + 1 / 64), rtol=1e-5)
end

function test_integrate_hirshfeld_weights_single_1s()
    """Test molecular integral in H atom with Hirshfeld weights."""
    pts = UniformInteger(100)
    tf = ExpRTransform(1e-5, 2e1)
    rgrid = transform_1d_grid(tf, pts)
    coordinates = [0.0, 0.0, -0.5]
    atg1 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates,
    )
    mg = MolGrid([7], [atg1], HirshfeldWeights())
    # dist0 = sqrt.(sum(abs2, coordinates .- mg.points, dims=2))
    dist0 = [norm(coordinates .- vec(point)) for point in eachrow(mg.points)]
    fn = exp.(-2 * dist0) / π
    @test isapprox(integrate(mg, fn), 1.0, atol=1e-6)
end

function test_integrate_hirshfeld_weights_pair_1s(rgrid)
    """Test molecular integral in H2."""
    coordinates = [0.0 0.0 -0.5; 0.0 0.0 0.5]
    atg1 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[1, :],
    )
    atg2 = from_pruned(
        rgrid,
        radius=0.5,
        sectors_r=[],
        sectors_degree=[17],
        center=coordinates[2, :],
    )
    mg = MolGrid([1, 1], [atg1, atg2], HirshfeldWeights())
    dist0 = [norm(coordinates[1, :] .- vec(point)) for point in eachrow(mg.points)]
    dist1 = [norm(coordinates[2, :] .- vec(point)) for point in eachrow(mg.points)]
    # dist0 = sqrt.(sum(abs2, coordinates[1] .- mg.points, dims=2))
    # dist1 = sqrt.(sum(abs2, coordinates[2] .- mg.points, dims=2))
    fn = exp.(-2 * dist0) / π + 1.5 * exp.(-2 * dist1) / π
    @test isapprox(integrate(mg, fn), 2.5, atol=1e-4)
end

function test_interpolation_with_gaussian_center()
    r"""Test interpolation with molecular grid of sum of two Gaussian examples."""
    coordinates = [0.0 0.0 -1.5; 0.0 0.0 1.5]

    pts = Trapezoidal(400)
    tf = LinearFiniteRTransform(1e-8, 10.0)
    rgrid = transform_1d_grid(tf, pts)

    atg1 = AtomGrid(rgrid, degrees=[15], center=coordinates[1, :])
    atg2 = AtomGrid(rgrid, degrees=[17], center=coordinates[2, :])
    mg = MolGrid([1, 1], [atg1, atg2], BeckeWeights(), store=true)

    function gaussian_func(points)
        r1 = [norm(coordinates[1, :] .- vec(point)) for point in eachrow(points)]
        r2 = [norm(coordinates[2, :] .- vec(point)) for point in eachrow(points)]

        # r1 = vec(sqrt.(sum(abs2, pts .- coordinates[1, :][newaxis, :], dims=2)))
        # r2 = vec(sqrt.(sum(abs2, pts .- coordinates[2, :][newaxis, :], dims=2)))
        return exp.(-5.0 * r1 .^ 2.0) .+ exp.(-3.5 * r2 .^ 2.0)
    end

    gaussians = gaussian_func(mg.points)
    interpolate_func = interpolate(mg, gaussians)
    result = interpolate_func(mg.points)
    @test all(isapprox.(result, gaussians, atol=1e-3))

    # Test interpolation at new points
    new_pts = rand(Uniform(1.0, 2.0), (100, 3))
    @test all(isapprox.(interpolate_func(new_pts), gaussian_func(new_pts), atol=1e-3))
end



@testset "TestMolGrid" begin
    # Set up radial grid for integral tests
    pts = UniformInteger(100)
    tf = ExpRTransform(1e-5, 2e1)
    rgrid = transform_1d_grid(tf, pts)

    test_integrate_hydrogen_single_1s(rgrid)
    test_make_grid_integral()
    test_make_grid_different_grid_type()
    test_make_grid_different_rad_type()
    test_integrate_hydrogen_pair_1s(rgrid)
    test_integrate_hydrogen_trimer_1s(rgrid)
    test_integrate_hydrogen_8_1s(rgrid)
    test_molgrid_attrs_subgrid(rgrid)
    test_molgrid_attrs(rgrid)
    test_different_aim_weights_h2(rgrid)
    test_from_size(rgrid)
    test_from_pruned(rgrid)
    test_get_localgrid_1s(rgrid)
    test_get_localgrid_1s1s(rgrid)
    test_integrate_hirshfeld_weights_single_1s()
    test_integrate_hirshfeld_weights_pair_1s(rgrid)
end

@testset "interpolate" begin
    test_interpolation_with_gaussian_center()
end

