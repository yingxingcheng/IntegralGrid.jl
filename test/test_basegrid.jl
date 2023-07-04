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

function test_getitem(grid, _ref_points, _ref_weights)
    # test index
    grid_index = grid[10]
    ref_grid = Grid(_ref_points[10:10], _ref_weights[10:10])
    @test all(isapprox.(grid_index._points, ref_grid._points))
    @test all(isapprox.(grid_index._weights, ref_grid._weights))
    @test grid_index isa Grid
    # test slice
    ref_grid_slice = Grid(_ref_points[1:11], _ref_weights[1:11])
    grid_slice = grid[1:11]
    @test all(isapprox.(grid_slice._points, ref_grid_slice._points))
    @test all(isapprox.(grid_slice._weights, ref_grid_slice._weights))
    @test grid_slice isa Grid
    a = [1, 3, 5]
    ref_smt_index = grid[a]
    @test all(isapprox.(ref_smt_index._points, _ref_points[a]))
    @test all(isapprox.(ref_smt_index._weights, _ref_weights[a]))
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

end
