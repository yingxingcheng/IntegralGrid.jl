module OnedGrid

using IntegralGrid.BaseGrid
using FastGaussQuadrature
using LinearAlgebra

export GaussLaguerre, GaussLegendre, GaussChebyshev, UniformInteger, GaussChebyshevType2
export GaussChebyshevLobatto, Trapezoidal, RectangleRuleSineEndPoints
export TanhSinh, Simpson, MidPoint, ClenshawCurtis, FejerFirst, FejerSecond
export _g2, _g3, _derg2, _derg3
export TrefethenCC, TrefethenGC2, TrefethenGeneral
export _gstrip, _dergstrip
export TrefethenStripCC, TrefethenStripGC2, TrefethenStripGeneral
export ExpSinh, LogExpSinh, ExpExp, SingleTanh


function GaussLaguerre(npoints::Int, alpha::T=0) where {T<:Real}
    if npoints <= 1
        throw(ArgumentError("Argument npoints must be an integer > 1, given $npoints"))
    end
    if alpha <= -1
        throw(ArgumentError("Argument alpha must be larger than -1, given $alpha"))
    end
    points, weights = gausslaguerre(npoints, alpha)
    if any(isnan.(weights))
        throw(RuntimeError("Generation of the weights for Gauss-generalized Laguerre quadrature contains nans. This issue is related to SciPy."))
    end
    weights .= weights .* exp.(points) .* points .^ (-alpha)
    return OneDGrid(points, weights, (0, Inf))
end

function GaussLegendre(npoints::Int)
    if npoints <= 1
        throw(ArgumentError("Argument npoints must be an integer > 1, given $npoints"))
    end
    points, weights = gausslegendre(npoints)
    return OneDGrid(points, weights, (-1, 1))
end

function GaussChebyshev(npoints::Int)
    if npoints <= 1
        throw(ArgumentError("Argument npoints must be an integer > 1, given $npoints"))
    end
    points, weights = gausschebyshev(npoints, 1)
    weights .= weights .* sqrt.(1 .- points .^ 2)
    return OneDGrid(points, weights, (-1, 1))
end

function UniformInteger(npoints::Int)
    if npoints <= 1
        throw(ArgumentError("Argument npoints must be an integer > 1, given $npoints"))
    end
    points = collect(0:npoints-1)
    weights = ones(npoints)
    return OneDGrid(points, weights, (0, Inf))
end

function GaussChebyshevType2(npoints::Int)
    if npoints < 1
        throw(ArgumentError("Argument npoints must be an integer > 1, given $npoints"))
    end
    points, weights = gausschebyshev(npoints, 2)
    weights .= weights ./ sqrt.(1 .- points .^ 2)
    return OneDGrid(points, weights, (-1, 1))
end

function GaussChebyshevLobatto(npoints::Int)
    if npoints <= 1
        throw(ArgumentError("Argument npoints must be an integer > 1, given $npoints"))
    end
    points = cos.(range(0, stop=npoints - 1) * π / (npoints - 1))
    points = reverse(points)
    weights = π * sqrt.(1 .- points .^ 2) / (npoints - 1)
    weights[1] /= 2
    weights[npoints] /= 2
    return OneDGrid(points, weights, (-1, 1))
end

function Trapezoidal(npoints::Int)
    @assert npoints > 1  # Use @assert for validation instead of throwing an error

    points = Vector(range(-1, stop=1, length=npoints))
    weights = fill(2 / (npoints - 1), npoints)  # Use fill function to initialize weights
    weights[1] /= 2
    weights[end] /= 2
    return OneDGrid(points, weights, (-1, 1))
end


function RectangleRuleSineEndPoints(npoints::Int)
    if npoints <= 1
        throw(ArgumentError("Argument npoints must be an integer > 1, given $npoints"))
    end
    points = collect(1:npoints) ./ (npoints + 1)
    m = collect(1:npoints)
    bm = @. (1.0 - cos(m * π)) / (m * π)
    sim = @. sin(m' * π * points)
    weights = @. bm * sim
    weights = 2 / (npoints + 1) * weights
    points = 2 * points .- 1
    weights = 2 * weights
    return OneDGrid(points, weights, (-1, 1))
end


function TanhSinh(npoints::Int, delta::Number=0.1)::OneDGrid
    if npoints <= 1
        throw(ArgumentError("Argument npoints must be an integer > 1, given $npoints"))
    end
    if npoints % 2 == 0
        throw(ArgumentError("Argument npoints must be an odd integer, given $npoints"))
    end

    j = Int(round((1 - npoints) / 2)) .+ collect(0:npoints-1)
    theta = j * delta
    points = @. tanh(0.5 * π * sinh(theta))
    weights = @. cosh(theta) / cosh(0.5 * π * sinh(theta))^2
    weights = @. 0.5 * π * delta * weights
    return OneDGrid(points, weights, (-1, 1))
end

function Simpson(npoints::Int)::OneDGrid
    if npoints <= 1
        throw(ArgumentError("npoints must be greater than one, given $npoints"))
    end
    if npoints % 2 == 0
        throw(ArgumentError("npoints must be odd, given $npoints"))
    end
    points = collect(LinRange(-1, 1, npoints))
    weights = @. 2 * ones(npoints) / (3 * (npoints - 1))
    weights[2:2:npoints-1] *= 4.0
    weights[3:2:npoints-1] *= 2.0
    return OneDGrid(points, weights, (-1, 1))
end

function MidPoint(npoints::Int)::OneDGrid
    if npoints <= 1
        throw(ArgumentError("Argument npoints must be an integer > 1, given $npoints"))
    end
    points = -1 .+ (2 * collect(0:npoints-1) .+ 1) / npoints
    weights = @. 2 * ones(npoints) / npoints
    return OneDGrid(points, weights, (-1, 1))
end

function ClenshawCurtis(npoints::Int)::OneDGrid
    if npoints <= 1
        throw(ArgumentError("npoints must be greater than one, given $npoints"))
    end

    theta = π * collect(0:npoints-1) / (npoints - 1)
    theta = reverse(theta)
    points = cos.(theta)

    jmed = div(npoints - 1, 2)
    bj = 2.0 * ones(jmed)
    if 2 * jmed + 1 == npoints
        bj[jmed] = 1.0
    end

    j = collect(0:jmed-1)
    bj ./= 4 * j .* (j .+ 2) .+ 3
    cij = cos.(2 * (j .+ 1) * theta')
    wi = vec(bj' * cij)

    weights = 2 * (1 .- wi) / (npoints - 1)
    weights[1] /= 2
    weights[npoints] /= 2

    return OneDGrid(points, weights, (-1, 1))
end

function FejerFirst(npoints::Int)::OneDGrid
    if npoints <= 1
        throw(ArgumentError("npoints must be greater than one, given $npoints"))
    end

    theta = π * (2 * collect(0:npoints-1) .+ 1) / (2 * npoints)
    points = cos.(theta)

    nsum = div(npoints, 2)
    j = collect(0:nsum-2) .+ 1

    bj = 2.0 * ones(nsum - 1) ./ (4 * j .^ 2 .- 1)
    cij = cos.(2 * j * theta')
    di = vec(bj' * cij)
    weights = 1 .- di

    points = reverse(points)
    weights = reverse(weights) * (2 / npoints)

    OneDGrid(points, weights, (-1, 1))
end

function FejerSecond(npoints::Int)::OneDGrid
    if npoints <= 1
        throw(ArgumentError("npoints must be greater than one, given $npoints"))
    end

    theta = π * (collect(0:npoints-1) .+ 1) / (npoints + 1)

    points = cos.(theta)

    nsum = div(npoints + 1, 2)
    j = collect(0:nsum-2) .+ 1

    bj = ones(nsum - 1) ./ (2 * j .- 1)
    sij = sin.((2 * j .- 1) * theta')
    wi = vec(bj' * sij)
    weights = 4 * sin.(theta) .* wi

    points = reverse(points)
    weights = reverse(weights) / (npoints + 1)

    OneDGrid(points, weights, (-1, 1))
end


# Auxiliary functions for Trefethen "sausage" transformation
# g2 is the function and derg2 is the first derivative.
# g3 is other function with the same boundary conditions of g2 and
# derg3 is the first derivative.
# these functions work for TrefethenCC, TrefethenGC2, and TrefethenGeneral
function _g2(x)
    return (1 / 149) * (120 * x + 20 * x .^ 3 + 9 * x .^ 5)
end

function _derg2(x)
    return (1 / 149) * (120 .+ 60 * x .^ 2 + 45 * x .^ 4)
end

function _g3(x)
    return (1 / 53089) * (40320 * x + 6720 * x .^ 3 + 3024 * x .^ 5 + 1800 * x .^ 7 + 1225 * x .^ 9)
end

function _derg3(x)
    return (1 / 53089) * (40320 .+ 20160 * x .^ 2 + 15120 * x .^ 4 + 12600 * x .^ 6 + 11025 * x .^ 8)
end

function TrefethenCC(npoints::Int, d::Int=9)
    # Generate 1D grid on [-1,1] interval based on Trefethen-Clenshaw-Curtis

    grid = ClenshawCurtis(npoints)

    if d == 1
        points = grid.points
        weights = grid.weights
    elseif d == 5
        points = _g2(grid.points)
        weights = _derg2(grid.points) .* grid.weights
    elseif d == 9
        points = _g3(grid.points)
        weights = _derg3(grid.points) .* grid.weights
    else
        error("Degree $d should be either 1, 5, 9.")
    end

    return OneDGrid(points, weights, (-1, 1))
end

function TrefethenGC2(npoints::Int, d::Int=9)
    # Generate 1D grid on [-1,1] interval based on Trefethen-Gauss-Chebyshev

    grid = GaussChebyshevType2(npoints)

    if d == 1
        points = grid.points
        weights = grid.weights
    elseif d == 5
        points = _g2(grid.points)
        weights = _derg2(grid.points) .* grid.weights
    elseif d == 9
        points = _g3(grid.points)
        weights = _derg3(grid.points) .* grid.weights
    else
        error("Degree $d should be either 1, 5, 9.")
    end

    return OneDGrid(points, weights, (-1, 1))
end

function TrefethenGeneral(npoints::Int, quadrature::Function, d::Int=9)
    # Generate 1D grid on [-1,1] interval based on Trefethen-General

    if !(typeof(quadrature) <: Function)
        error("Quadrature $(typeof(quadrature)) should be of type Function.")
    end

    grid = quadrature(npoints)

    if d == 1
        points = grid.points
        weights = grid.weights
    elseif d == 5
        points = _g2(grid.points)
        weights = _derg2(grid.points) .* grid.weights
    elseif d == 9
        points = _g3(grid.points)
        weights = _derg3(grid.points) .* grid.weights
    else
        error("Degree $d should be either 1, 5, 9.")
    end

    return OneDGrid(points, weights, (-1, 1))
end


function _gstrip(rho, s)
    tau = pi / log(rho)
    termd = 0.5 + 1 / (exp(tau * pi) + 1)
    u = asin.(s)

    cn = 1 / (log(1 + exp(-tau * pi)) - log(2) + pi * tau * termd / 2)

    g = cn * (
        log.(1 .+ exp.(-tau * (pi / 2 .+ u))) .-
        log.(1 .+ exp.(-tau * (pi / 2 .- u))) .+
        termd * tau * u
    )

    return g
end


function _dergstrip(rho, s)
    tau = pi / log(rho)
    termd = 0.5 + 1 / (exp(tau * pi) + 1)

    cn = 1 / (log(1 + exp(-tau * pi)) - log(2) + pi * tau * termd / 2)

    gp = zeros(Float64, length(s))

    # get true label
    mask_true = abs.(abs.(s) .- 1) .< 1.0e-8
    # get false label
    mask_false = .~mask_true
    u = asin.(s)
    gp[mask_true] .= cn * tau^2 / 4 * tanh(tau * pi / 2)^2
    gp[mask_false] .= (
        1 ./ (exp.(tau * (pi / 2 .+ u[mask_false])) .+ 1)
        .+
        1 ./ (exp.(tau * (pi / 2 .- u[mask_false])) .+ 1)
        .-
        termd
    ) .* (-cn * tau ./ sqrt.(1 .- s[mask_false] .^ 2))

    return gp
end

function TrefethenStripCC(npoints::Int, rho::Real=1.1)
    grid = ClenshawCurtis(npoints)
    points = _gstrip(rho, grid.points)
    weights = _dergstrip(rho, grid.points) .* grid.weights
    return OneDGrid(points, weights, (-1, 1))
end

function TrefethenStripGC2(npoints::Int, rho::Real=1.1)
    grid = GaussChebyshevType2(npoints)
    points = _gstrip(rho, grid.points)
    weights = _dergstrip(rho, grid.points) .* grid.weights
    return OneDGrid(points, weights, (-1, 1))
end

function TrefethenStripGeneral(npoints::Int, quadrature, rho::Real=1.1)
    grid = quadrature(npoints)
    points = _gstrip(rho, grid.points)
    weights = _dergstrip(rho, grid.points) .* grid.weights
    return OneDGrid(points, weights, (-1, 1))
end

function ExpSinh(npoints::Int, h::Real=1.0)
    if h <= 0
        error("The value of h must be bigger than 0, given ", h)
    end
    if npoints < 1
        error("npoints must be bigger than 1, given ", npoints)
    end
    if npoints % 2 == 0
        error("npoints must be odd, given ", npoints)
    end
    m = div(npoints - 1, 2)
    k = range(-m, stop=m)
    points = exp.(pi .* sinh.(k * h) / 2)
    weights = points .* pi * h .* cosh.(k * h) / 2
    return OneDGrid(points, weights, (0, Inf))
end

function LogExpSinh(npoints::Int, h::Real=0.1)
    if h <= 0
        error("The value of h must be bigger than 0, given ", h)
    end
    if npoints < 1
        error("npoints must be bigger than 1, given ", npoints)
    end
    if npoints % 2 == 0
        error("npoints must be odd, given ", npoints)
    end
    m = div(npoints - 1, 2)
    k = range(-m, stop=m)
    points = log.(exp.(pi * sinh.(k * h) / 2) .+ 1)
    weights = exp.(pi * sinh.(k * h) / 2) .* pi * h .* cosh.(k * h) / 2
    weights ./= exp.(pi * sinh.(k * h) / 2) .+ 1
    return OneDGrid(points, weights, (0, Inf))
end

function ExpExp(npoints::Int, h::Real=0.1)
    if h <= 0
        error("The value of h must be bigger than 0, given ", h)
    end
    if npoints < 1
        error("npoints must be bigger than 1, given ", npoints)
    end
    if npoints % 2 == 0
        error("npoints must be odd, given ", npoints)
    end
    m = div(npoints - 1, 2)
    k = range(-m, stop=m)
    points = exp.(k * h) .* exp.(-exp.(-k * h))
    weights = h * exp.(-exp.(-k * h)) .* (exp.(k * h) .+ 1)
    return OneDGrid(points, weights, (0, Inf))
end

function SingleTanh(npoints::Int, h::Real=0.1)
    if h <= 0
        error("The value of h must be bigger than 0, given ", h)
    end
    if npoints < 1
        error("npoints must be bigger than 1, given ", npoints)
    end
    if npoints % 2 == 0
        error("npoints must be odd, given ", npoints)
    end
    m = div(npoints - 1, 2)
    k = range(-m, stop=m)
    points = tanh.(k * h)
    weights = h ./ cosh.(k * h) .^ 2
    return OneDGrid(points, weights, (-1, 1))
end


end
