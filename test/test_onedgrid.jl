using IntegralGrid.BaseGrid, IntegralGrid.OnedGrid
using Test
using FastGaussQuadrature
using SpecialFunctions
using LinearAlgebra

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

function test_gausschebyshev()
    # Test Gauss Chebyshev polynomial grid.
    points, weights = gausschebyshev(10)
    weights = weights .* sqrt.(1 .- points .^ 2)
    grid = GaussChebyshev(10)
    @test grid.points ≈ sort(points)
    @test grid.weights ≈ weights
end

function test_horton_linear()
    # Test horton linear grids.
    grid = UniformInteger(10)
    @test grid.points ≈ collect(0:9)
    @test grid.weights ≈ ones(10)
end

function test_gausschebyshev2()
    # Test Gauss Chebyshev type 2 polynomial grid.
    points, weights = gausschebyshev(10, 2)
    grid = GaussChebyshevType2(10)
    weights ./= sqrt.(1 .- points .^ 2)
    @test grid.points ≈ points
    @test grid.weights ≈ weights
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

    @test grid.points ≈ points
    @test grid.weights ≈ weights
end

function test_trapezoidal()
    # Test for Trapezoidal rule.
    grid = Trapezoidal(10)

    idx = collect(0:9)
    points = -1 .+ (2 .* idx ./ 9)

    weights = 2 .* ones(10) ./ 9
    weights[1] /= 2
    weights[10] = weights[1]

    @test grid.points ≈ points
    @test grid.weights ≈ weights
end

function test_tanhsinh()
    """Test for Tanh - Sinh rule."""
    delta = 0.1 * π / sqrt(11)
    grid = TanhSinh(11, delta)

    jmin = -5
    points = zeros(11)
    weights = zeros(11)

    for i in collect(1:11)
        j = jmin + i - 1
        arg = π * sinh(j * delta) / 2

        points[i] = tanh(arg)

        weights[i] = π * delta * cosh(j * delta) * 0.5
        weights[i] /= cosh(arg)^2
    end

    @test grid.points ≈ points
    @test grid.weights ≈ weights
end

function test_simpson()
    grid = Simpson(11)
    idx = collect(0:10)
    points = @. -1 + (2 * idx / 10)
    weights = @. 2 * ones(11) / 30

    for i in collect(2:10)
        if i % 2 == 1
            weights[i] *= 2
        else
            weights[i] *= 4
        end
    end
    @test grid.points ≈ points
    @test grid.weights ≈ weights
end

function test_midpoint()
    """Test for midpoint rule."""
    grid = MidPoint(10)
    points = zeros(10)
    weights = @. ones(10) * 2 / 10
    idx = collect(0:9)
    points = -1 .+ (2 * idx .+ 1) / 10
    @test grid.points ≈ points
    @test grid.weights ≈ weights
end

function test_ClenshawCurtis()
    """Test for ClenshawCurtis."""
    grid = ClenshawCurtis(10)
    points = zeros(10)
    weights = 2 * ones(10)

    theta = zeros(10)

    for i in 0:9
        theta[i+1] = (9 - i) * π / 9
    end

    points .= cos.(theta)
    weights .= zeros(10)

    jmed = div(9, 2)

    for i in 1:10
        weights[i] = 1

        for j in 0:jmed-1
            if 2 * (j + 1) == 9
                b = 1
            else
                b = 2
            end

            weights[i] -= b * cos(2 * (j + 1) * theta[i]) / (4 * j * (j + 2) + 3)
        end
    end

    for i in 2:9
        weights[i] *= 2 / 9
    end

    weights[1] /= 9
    weights[10] /= 9

    @test grid.points ≈ points
    @test grid.weights ≈ weights
end

function test_FejerFirst()
    """Test for Fejer first rule."""
    grid = FejerFirst(10)

    theta = π * (2 * collect(0:9) .+ 1) / 20

    points = cos.(theta)
    weights = zeros(10)

    nsum = div(10, 2)

    for k in 1:10
        serie = 0.0

        for m in 1:nsum-1
            serie += cos(2 * m * theta[k]) / (4 * m^2 - 1)
        end

        serie = 1 - 2 * serie

        weights[k] = (2 / 10) * serie
    end

    points = reverse(points)
    weights = reverse(weights)

    @test grid.points ≈ points
    @test grid.weights ≈ weights
end

function test_FejerSecond()
    """Test for Fejer second rule."""
    grid = FejerSecond(10)

    theta = π * (collect(0:9) .+ 1) / 11

    points = cos.(theta)
    weights = zeros(10)

    nsum = div(11, 2)

    for k in 1:10
        serie = 0.0

        for m in 1:nsum-1
            serie += sin((2 * m - 1) * theta[k]) / (2 * m - 1)
        end

        weights[k] = (4 * sin(theta[k]) / 11) * serie
    end

    points = reverse(points)
    weights = reverse(weights)

    @test grid.points ≈ points
    @test grid.weights ≈ weights
end


function test_AuxiliarTrefethenSausage()
    # Test for Auxiliary functions using in Trefethen Sausage.
    xref = [-1.0, 0.0, 1.0]
    d2ref = [1.5100671140939597, 0.8053691275167785, 1.5100671140939597]
    d3ref = [1.869031249411366, 0.7594793648401741, 1.869031249411366]
    newxg2 = _g2(xref)
    newxg3 = _g3(xref)
    newd2 = _derg2(xref)
    newd3 = _derg3(xref)

    @test all(isequal(xref, newxg2))
    @test all(isequal(xref, newxg3))
    @test all(isequal(d2ref, newd2))
    @test all(isequal(d3ref, newd3))
end

function test_TrefethenCC_d2()
    # Test for Trefethen - Sausage Clenshaw Curtis and parameter d=5.
    grid = TrefethenCC(10, 5)

    tmp = ClenshawCurtis(10)

    new_points = _g2(tmp.points)
    new_weights = _derg2(tmp.points) .* tmp.weights

    @test all(isequal(grid.points, new_points))
    @test all(isequal(grid.weights, new_weights))
end

function test_TrefethenCC_d3()
    # Test for Trefethen - Sausage Clenshaw Curtis and parameter d=9.
    grid = TrefethenCC(10, 9)

    tmp = ClenshawCurtis(10)

    new_points = _g3(tmp.points)
    new_weights = _derg3(tmp.points) .* tmp.weights

    @test all(isequal(grid.points, new_points))
    @test all(isequal(grid.weights, new_weights))
end

function test_TrefethenCC_d0()
    # Test for Trefethen - Sausage Clenshaw Curtis and parameter d=1.
    grid = TrefethenCC(10, 1)

    tmp = ClenshawCurtis(10)

    @test all(isequal(grid.points, tmp.points))
    @test all(isequal(grid.weights, tmp.weights))
end

function test_TrefethenGC2_d2()
    # Test for Trefethen - Sausage GaussChebyshev2 and parameter d=5.
    grid = TrefethenGC2(10, 5)

    tmp = GaussChebyshevType2(10)

    new_points = _g2(tmp.points)
    new_weights = _derg2(tmp.points) .* tmp.weights

    @test all(isequal(grid.points, new_points))
    @test all(isequal(grid.weights, new_weights))
end

function test_TrefethenGC2_d3()
    # Test for Trefethen - Sausage GaussChebyshev2 and parameter d=9.
    grid = TrefethenGC2(10, 9)

    tmp = GaussChebyshevType2(10)

    new_points = _g3(tmp.points)
    new_weights = _derg3(tmp.points) .* tmp.weights

    @test all(isequal(grid.points, new_points))
    @test all(isequal(grid.weights, new_weights))
end

function test_TrefethenGC2_d0()
    # Test for Trefethen - Sausage GaussChebyshev2 and parameter d=1.
    grid = TrefethenGC2(10, 1)

    tmp = GaussChebyshevType2(10)

    @test all(isequal(grid.points, tmp.points))
    @test all(isequal(grid.weights, tmp.weights))
end

function test_TrefethenGeneral_d2()
    # Test for Trefethen - Sausage General and parameter d=5.
    grid = TrefethenGeneral(10, ClenshawCurtis, 5)
    new = TrefethenCC(10, 5)

    @test all(isequal(grid.points, new.points))
    @test all(isequal(grid.weights, new.weights))
end

function test_TrefethenGeneral_d3()
    # Test for Trefethen - Sausage General and parameter d=9.
    grid = TrefethenGeneral(10, ClenshawCurtis, 9)
    new = TrefethenCC(10, 9)

    @test all(isequal(grid.points, new.points))
    @test all(isequal(grid.weights, new.weights))
end

function test_TrefethenGeneral_d0()
    # Test for Trefethen - Sausage General and parameter d=1.
    grid = TrefethenGeneral(10, ClenshawCurtis, 1)
    new = TrefethenCC(10, 1)

    @test all(isequal(grid.points, new.points))
    @test all(isequal(grid.weights, new.weights))
end

function test_TrefethenStripCC()
    grid = TrefethenStripCC(10, 1.1)
    tmp = ClenshawCurtis(10)

    new_points = _gstrip(1.1, tmp.points)
    new_weights = _dergstrip(1.1, tmp.points) .* tmp.weights

    @test isapprox(grid.points, new_points)
    @test isapprox(grid.weights, new_weights)
end

function test_TrefethenStripGC2()
    grid = TrefethenStripGC2(10, 1.1)
    tmp = GaussChebyshevType2(10)

    new_points = _gstrip(1.1, tmp.points)
    new_weights = _dergstrip(1.1, tmp.points) .* tmp.weights

    @test isapprox(grid.points, new_points)
    @test isapprox(grid.weights, new_weights)
end

function test_TrefethenStripGeneral()
    grid = TrefethenStripGeneral(10, ClenshawCurtis, 3)
    new = TrefethenStripCC(10, 3)

    @test isapprox(grid.points, new.points)
    @test isapprox(grid.weights, new.weights)
end

function test_ExpSinh()
    grid = ExpSinh(11, 0.1)

    k = -5:5
    points = exp.(π .* sinh.(k .* 0.1) ./ 2)
    weights = points .* π .* 0.1 .* cosh.(k .* 0.1) ./ 2

    @test isapprox(grid.points, points)
    @test isapprox(grid.weights, weights)
end

function test_LogExpSinh()
    grid = LogExpSinh(11, 0.1)

    k = -5:5
    points = log.(exp.(π .* sinh.(k .* 0.1) ./ 2) .+ 1)
    weights = exp.(π .* sinh.(k .* 0.1) ./ 2) .* π .* 0.1 .* cosh.(k .* 0.1) ./ 2
    weights ./= exp.(π .* sinh.(k .* 0.1) ./ 2) .+ 1

    @test isapprox(grid.points, points)
    @test isapprox(grid.weights, weights)
end

function test_ExpExp()
    grid = ExpExp(11, 0.1)

    k = -5:5
    points = exp.(k .* 0.1) .* exp.(-exp.(-k .* 0.1))
    weights = 0.1 .* exp.(-exp.(-k .* 0.1)) .* (exp.(k .* 0.1) .+ 1)

    @test isapprox(grid.points, points)
    @test isapprox(grid.weights, weights)
end

function test_SingleTanh()
    grid = SingleTanh(11, 0.1)

    k = -5:5
    points = tanh.(k .* 0.1)
    weights = 0.1 ./ cosh.(k .* 0.1).^2

    @test isapprox(grid.points, points)
    @test isapprox(grid.weights, weights)
end

function test_SingleExp()
    grid = SingleExp(11, 0.1)

    k = -5:5
    points = exp.(k .* 0.1)
    weights = 0.1 .* points

    @test isapprox(grid.points, points)
    @test isapprox(grid.weights, weights)
end

function test_SingleArcSinhExp()
    grid = SingleArcSinhExp(11, 0.1)

    k = -5:5
    points = asinh.(exp.(k .* 0.1))
    weights = 0.1 .* exp.(k .* 0.1) ./ sqrt.(exp.(2 .* 0.1 .* k) .+ 1)

    @test isapprox(grid.points, points)
    @test isapprox(grid.weights, weights)
end

function test_AuxiliarTrefethenSausage()
    xref = [-1.0, 0.0, 1.0]
    d2ref = [1.5100671140939597, 0.8053691275167785, 1.5100671140939597]
    d3ref = [1.869031249411366, 0.7594793648401741, 1.869031249411366]
    newxg2 = _g2(xref)
    newxg3 = _g3(xref)
    newd2 = _derg2(xref)
    newd3 = _derg3(xref)

    @test isapprox(xref, newxg2)
    @test isapprox(xref, newxg3)
    @test isapprox(d2ref, newd2)
    @test isapprox(d3ref, newd3)
end

function test_AuxiliarTrefethenStrip()
    xref = [-1.0, 0.0, 1.0]
    dref = [10.7807092, 0.65413403, 10.7807092]
    newx = _gstrip(1.1, xref)
    newd = _dergstrip(1.1, xref)

    @test isapprox(xref, newx)
    @test isapprox(dref, newd)
end


@testset "OnedGrid.jl" begin
    test_gausslaguerre()
    test_gausslegendre()
    test_gausschebyshev()
    test_gausschebyshev2()
    test_gausschebyshevlobatto()
    test_horton_linear()
    test_trapezoidal()
    test_tanhsinh()
    test_simpson()
    test_midpoint()
    test_ClenshawCurtis()
    test_FejerFirst()
    test_FejerSecond()
    test_AuxiliarTrefethenSausage()
    test_TrefethenCC_d2()
    test_TrefethenCC_d3()
    test_TrefethenCC_d0()
    test_TrefethenGC2_d2()
    test_TrefethenGC2_d3()
    test_TrefethenGC2_d0()
    test_TrefethenGeneral_d2()
    test_TrefethenGeneral_d3()
    test_TrefethenGeneral_d0()
    test_TrefethenStripCC()
    test_TrefethenStripGC2()
    test_TrefethenStripGeneral()
    test_ExpSinh()
    test_LogExpSinh()
    test_ExpExp()
    test_SingleTanh()
    test_AuxiliarTrefethenSausage()
    test_AuxiliarTrefethenStrip()

end
