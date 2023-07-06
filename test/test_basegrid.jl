using IntegralGrid.BaseGrid
using Test

function test_init(grid, ref_points, ref_weights)
    @test isa(grid, Grid)
    @test isapprox(grid.points, ref_points, atol=1e-7)
    @test isapprox(grid.weights, ref_weights)
    @test size(grid.weights) == size(ref_weights)
end

function test_init_diff_inputs()
    Grid([1, 2, 3], [2, 3, 4])
    Grid([1.0, 2.0, 3.0], [2, 3, 4])
    Grid([1, 2, 3], [2.0, 3.0, 4.0])
    return true
end

function test_integrate(grid)
    result = integrate(grid, ones(21))
    @test isapprox(result, 2.1, atol=1e-6)

    value1 = collect(-1:0.1:1)
    value2 = value1 .^ 2
    result2 = integrate(grid, value1, value2)
    @test isapprox(result2, 0, atol=1e-7)

end

function test_get_localgrid_radius_inf(grid)
    """Test the creation of the local grid with an infinite radius."""
    localgrid = get_localgrid(grid, grid.points[4], Inf)
    # Just make sure we are testing with a real local grid
    @test localgrid.size == grid.size
    @test all(isapprox.(localgrid.points, grid.points))
    @test all(isapprox.(localgrid.weights, grid.weights))
    @test all(isapprox.(localgrid.indices, 1:grid.size))
end

function test_get_localgrid(grid, _ref_points, _ref_weights)
    """Test the creation of the local grid with a normal radius."""
    center = grid.points[4]
    radius = 0.2
    localgrid = get_localgrid(grid, center, radius)
    # Just make sure we are testing with an actual local grid with less (but
    # not zero) points.
    @test localgrid.size > 0
    @test localgrid.size < grid.size
    # Test that the local grid contains the correct results.
    @test ndims(localgrid.points) == ndims(grid.points)
    @test ndims(localgrid.weights) == ndims(grid.weights)
    @test all(isapprox.(localgrid.points, grid.points[localgrid.indices]))
    @test all(isapprox.(localgrid.weights, grid.weights[localgrid.indices]))
    if ndims(_ref_points) == 2
        @test all(norm.(localgrid.points .- center, dims=2) .<= radius)
    else
        @test all(abs.(localgrid.points .- center) .<= radius)
    end
end

function test_getitem(grid, _ref_points, _ref_weights)
    # test index
    grid_index = grid[10]
    ref_grid = Grid(_ref_points[10:10], _ref_weights[10:10])
    @test all(isapprox.(grid_index._points, ref_grid._points))
    @test all(isapprox.(grid_index.weights, ref_grid.weights))
    @test grid_index isa Grid
    # test slice
    ref_grid_slice = Grid(_ref_points[1:11], _ref_weights[1:11])
    grid_slice = grid[1:11]
    @test all(isapprox.(grid_slice._points, ref_grid_slice._points))
    @test all(isapprox.(grid_slice.weights, ref_grid_slice.weights))
    @test grid_slice isa Grid
    a = [1, 3, 5]
    ref_smt_index = grid[a]
    @test all(isapprox.(ref_smt_index.points, _ref_points[a]))
    @test all(isapprox.(ref_smt_index.weights, _ref_weights[a]))
end

function test_localgird()
    local_grid = LocalGrid([1, 2, 3], [1, 3, 4], 1)
    @test local_grid.points == [1, 2, 3]
    @test local_grid.weights == [1, 3, 4]
    @test local_grid.center == 1
    @test isnothing(local_grid.indices) == true
end

function test_onedgrid()
    @testset "TestOneDGrid" begin
        arr_1d = collect(0:9)

        @test_throws ArgumentError OneDGrid(arr_1d, arr_1d[1:end-1])
        @test_throws ArgumentError OneDGrid(arr_1d, arr_1d, (0, 5))
        @test_throws ArgumentError OneDGrid(arr_1d, arr_1d, (1, 5))
        @test_throws ArgumentError OneDGrid(arr_1d, arr_1d, (1, 9))
        @test_throws ArgumentError OneDGrid(arr_1d, arr_1d, (9, 0))

        @testset "test_getitem" begin
            points = collect(0:19)
            weights = collect(0:19) .* 0.1
            grid = OneDGrid(points, weights)

            @test grid.size == 20
            @test grid.domain === nothing

            subgrid = grid[1]
            @test subgrid.size == 1
            @test isapprox(subgrid.points, [points[1]])
            @test isapprox(subgrid.weights, [weights[1]])
            @test subgrid.domain === nothing

            subgrid = grid[4:7]
            @test subgrid.size == 4
            @test isapprox(subgrid.points, points[4:7])
            @test isapprox(subgrid.weights, weights[4:7])
            @test subgrid.domain === nothing
        end
    end
end

@testset "BaseGrid.jl" begin
    # Test setup function
    _ref_points = collect(-1:0.1:1)
    _ref_weights = ones(21) .* 0.1
    grid = Grid(_ref_points, _ref_weights)
    test_init(grid, _ref_points, _ref_weights)
    test_init_diff_inputs()
    test_integrate(grid)
    test_getitem(grid, _ref_points, _ref_weights)
    test_get_localgrid_radius_inf(grid)
    test_get_localgrid(grid, _ref_points, _ref_weights)
    test_localgird()
    test_onedgrid()
end
