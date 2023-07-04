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

@testset "BaseGrid.jl" begin
    # Test setup function
    _ref_points = collect(-1:0.1:1)
    _ref_weights = ones(21) .* 0.1
    grid = Grid(_ref_points, _ref_weights)
    test_init(grid, _ref_points, _ref_weights)
    test_init_diff_inputs()
    test_integrate(grid)

end
