using IntegralGrid.BaseGrid
using Test

function test_init(grid, ref_points, ref_weights)
    @test isa(grid, Grid)
    @test isapprox(get_points(grid), ref_points, atol=1e-7)
    @test isapprox(get_weights(grid), ref_weights)
    @test size(get_weights(grid)) == size(ref_weights)
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
    localgrid = get_localgrid(grid, get_points(grid)[4], Inf)
    # Just make sure we are testing with a real local grid
    @test get_size(localgrid) == get_size(grid)
    @test all(isapprox.(get_points(localgrid), get_points(grid)))
    @test all(isapprox.(get_weights(localgrid), get_weights(grid)))
    @test all(isapprox.(get_indices(localgrid), 1:get_size(grid)))
end

function test_get_localgrid(grid, _ref_points, _ref_weights)
    """Test the creation of the local grid with a normal radius."""
    center = get_points(grid)[4]
    radius = 0.2
    localgrid = get_localgrid(grid, center, radius)
    # Just make sure we are testing with an actual local grid with less (but
    # not zero) points.
    @test get_size(localgrid) > 0
    @test get_size(localgrid) < get_size(grid)
    # Test that the local grid contains the correct results.
    @test ndims(get_points(localgrid)) == ndims(get_points(grid))
    @test ndims(get_weights(localgrid)) == ndims(get_weights(grid))
    @test all(isapprox.(get_points(localgrid), get_points(grid)[get_indices(localgrid)]))
    @test all(isapprox.(get_weights(localgrid), get_weights(grid)[get_indices(localgrid)]))
    if ndims(_ref_points) == 2
        @test all(norm.(get_points(localgrid) .- center, dims=2) .<= radius)
    else
        @test all(abs.(get_points(localgrid) .- center) .<= radius)
    end
end

function test_getitem(grid, _ref_points, _ref_weights)
    # test index
    grid_index = grid[10]
    ref_grid = Grid(_ref_points[10:10], _ref_weights[10:10])
    @test all(isapprox.(get_points(grid_index), get_points(ref_grid)))
    @test all(isapprox.(get_weights(grid_index), get_weights(ref_grid)))
    @test grid_index isa Grid
    # test slice
    ref_grid_slice = Grid(_ref_points[1:11], _ref_weights[1:11])
    grid_slice = grid[1:11]
    @test all(isapprox.(get_points(grid_slice), get_points(ref_grid_slice)))
    @test all(isapprox.(get_weights(grid_slice), get_weights(ref_grid_slice)))
    @test grid_slice isa Grid
    a = [1, 3, 5]
    ref_smt_index = grid[a]
    @test all(isapprox.(get_points(ref_smt_index), _ref_points[a]))
    @test all(isapprox.(get_weights(ref_smt_index), _ref_weights[a]))
end

function test_localgird()
    local_grid = LocalGrid([1, 2, 3], [1, 3, 4], 1)
    @test get_points(local_grid) == [1, 2, 3]
    @test get_weights(local_grid) == [1, 3, 4]
    @test get_center(local_grid) == 1
    @test isnothing(get_indices(local_grid)) == true
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

            @test get_size(grid) == 20
            @test get_domain(grid) === nothing

            subgrid = grid[1]
            @test get_size(subgrid) == 1
            @test isapprox(get_points(subgrid), [points[1]])
            @test isapprox(get_weights(subgrid), [weights[1]])
            @test get_domain(subgrid) === nothing

            subgrid = grid[4:7]
            @test get_size(subgrid) == 4
            @test isapprox(get_points(subgrid), points[4:7])
            @test isapprox(get_weights(subgrid), weights[4:7])
            @test get_domain(subgrid) === nothing
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
