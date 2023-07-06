using IntegralGrid.OnedGrid
using Test
using FastGaussQuadrature

function test_gausslaguerre()
    """Test Guass Laguerre polynomial grid."""
    (points, weights) = gausslaguerre(10)
    weights = weights .* exp.(points) .* (points .^ 0)
    grid = GaussLaguerre(10)
    @test all(isapprox.(grid.points, points, rtol=1e-5))
    @test all(isapprox.(grid.weights, weights, rtol=1e-5))
end

function test_gausslegendre()
    points, weights = gausslegendre(10)
    grid = GaussLegendre(10)
    @test all(isapprox.(grid.points, points, rtol=1e-5))
    @test all(isapprox.(grid.weights, weights, rtol=1e-5))


end

@testset "OnedGrid.jl" begin
    # Test setup function
    test_gausslaguerre()
    test_gausslegendre()

end
