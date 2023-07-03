using IntegralGrid.BaseGrid
using Test

@testset "BaseGrid.jl" begin
    # Test setup function
    _ref_points = collect(-1:0.1:1)
    _ref_weights = ones(21) .* 0.1
    grid = Grid(_ref_points, _ref_weights)

    # test test_init_grid
    @test typeof(grid) == Grid
    @test isapprox(grid.points, _ref_points, atol=1e-7)
    @test isapprox(grid.weights, _ref_weights)
    @test size(grid.weights) == size(_ref_weights)

    # test_integrate
    result = integrate(grid, ones(21))
    @test isapprox(result, 2.1, atol=1e-6)

    value1 = collect(-1:0.1:1)
    value2 = value1 .^ 2
    # value2 = collect(Int, -10:1:10)
    result2 = integrate(grid, value1, value2)
    @test isapprox(result2, 0, atol=1e-7)

    println(grid[1])

end
