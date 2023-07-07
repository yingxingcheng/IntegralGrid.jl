using IntegralGrid.BaseGrid, IntegralGrid.OnedGrid
using Test
using FastGaussQuadrature

function test_gausslaguerre()
    """Test Guass Laguerre polynomial grid."""
    (points, weights) = gausslaguerre(10)
    weights = weights .* exp.(points) .* (points .^ 0)
    grid = GaussLaguerre(10)
    @test all(isapprox.(get_points(grid), points, rtol=1e-5))
    @test all(isapprox.(get_weights(grid), weights, rtol=1e-5))
end

function test_gausslegendre()
    points, weights = gausslegendre(10)
    grid = GaussLegendre(10)
    @test all(isapprox.(get_points(grid), points, rtol=1e-5))
    @test all(isapprox.(get_weights(grid), weights, rtol=1e-5))
end

function test_gausschebyshev()
    # Test Gauss Chebyshev polynomial grid.
    points, weights = gausschebyshev(10)
    weights = weights .* sqrt.(1 .- points .^ 2)
    grid = GaussChebyshev(10)
    @test get_points(grid) ≈ sort(points)
    @test get_weights(grid) ≈ weights
end

function test_horton_linear()
    # Test horton linear grids.
    grid = UniformInteger(10)
    @test get_points(grid) ≈ collect(0:9)
    @test get_weights(grid) ≈ ones(10)
end

function test_gausschebyshev2()
    # Test Gauss Chebyshev type 2 polynomial grid.
    points, weights = gausschebyshev(10, 2)
    grid = GaussChebyshevType2(10)
    weights ./= sqrt.(1 .- points .^ 2)
    @test get_points(grid) ≈ points
    @test get_weights(grid) ≈ weights
end

function test_gausschebyshevlobatto()
    # Test Gauss Chebyshev Lobatto grid.
    grid = GaussChebyshevLobatto(10)

    idx = collect(0:9)
    weights = ones(10)
    idx = idx .* π ./ 9

    points = cos.(idx)
    points = sort(points)

    weights .= weights .* π ./ 9
    weights .= weights .* sqrt.(1 .- points .^ 2)
    weights[1] /= 2
    weights[10] /= 2

    @test get_points(grid) ≈ points
    @test get_weights(grid) ≈ weights
end

function test_trapezoidal()
    # Test for Trapezoidal rule.
    grid = Trapezoidal(10)

    idx = collect(0:9)
    points = -1 .+ (2 .* idx ./ 9)

    weights = 2 .* ones(10) ./ 9
    weights[1] /= 2
    weights[10] = weights[1]

    @test get_points(grid) ≈ points
    @test get_weights(grid) ≈ weights
end


@testset "OnedGrid.jl" begin
    # Test setup function
    test_gausslaguerre()
    test_gausslegendre()
    test_gausschebyshev()
    test_gausschebyshev2()
    test_gausschebyshevlobatto()
    test_horton_linear()
    test_trapezoidal()

end
