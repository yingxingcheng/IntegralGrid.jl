module OnedGrid

using IntegralGrid.BaseGrid
using FastGaussQuadrature

export GaussLaguerre, GaussLegendre, GaussChebyshev, UniformInteger, GaussChebyshevType2
export GaussChebyshevLobatto, Trapezoidal, RectangleRuleSineEndPoints


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
    bm = (1.0 .- cos.(m .* π)) ./ (m .* π)
    sim = sin.(m' .* π .* points)
    weights = bm .* sim
    weights .= 2 ./ (npoints + 1) .* weights
    points = 2 .* points .- 1
    weights .= 2 .* weights
    return OneDGrid(points, weights, (-1, 1))
end


end