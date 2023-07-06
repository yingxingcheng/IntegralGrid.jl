module OnedGrid

using IntegralGrid.BaseGrid: AbstractOneDGrid, check_input
using FastGaussQuadrature

export GaussLaguerre, GaussLegendre


raw"""
    GaussLaguerre <: OneDGrid

Gauss Laguerre integral quadrature class.

The definition of generalized Gauss-Laguerre quadrature is:

.. math::
    \int_{0}^{\infty} x^\alpha e^{-x} f(x) dx \approx \sum_{i=1}^n w_i f(x_i),

where :math:`\alpha > -1`.

However, to integrate function :math:`g(x)` over :math:`[0, \infty)`, this is re-written as:

.. math::
    \int_{0}^{\infty} g(x)dx \approx
    \sum_{i=1}^n \left(\frac{w_i}{x_i^\alpha e^{-x_i}}\right) g(x_i) = \sum_{i=1}^n w_i' g(x_i)
"""
struct GaussLaguerre <: AbstractOneDGrid
    _points::Vector{<:Real}
    weights::Vector{<:Number}
    domain::Union{Nothing,Tuple{<:Real,<:Real}}

    function GaussLaguerre(
        points::Vector{<:Real},
        weights::Vector{<:Real},
        domain::Union{Nothing,Tuple{<:Real,<:Real}}=nothing)
        check_input(points, weights, domain=domain)
        new(points, weights, domain)
    end
end

raw"""
    GaussLaguerre(npoints::Integer, alpha::T = 0) where {T<:Real}

Generate grid on :math:`[0, \infty)` based on generalized Gauss-Laguerre quadrature.

# Arguments
- `npoints::Integer`: Number of grid points.
- `alpha<:Real=0`: Value of the parameter :math:`alpha` which must be larger than -1.

# Returns
- `::GaussLaguerre`: A 1-D grid instance containing points and weights.
"""
function GaussLaguerre(npoints::Integer, alpha::T=0) where {T<:Real}
    if npoints <= 1
        throw(ArgumentError("Argument npoints must be an integer > 1, given $npoints"))
    end
    if alpha <= -1
        throw(ArgumentError("Argument alpha must be larger than -1, given $alpha"))
    end
    # compute points and weights for Generalized Gauss-Laguerre quadrature
    points, weights = gausslaguerre(npoints, alpha)
    if any(isnan.(weights))
        throw(ErrorException(
            "Generation of the weights for Gauss-generalized Laguerre quadrature contains " *
            "nans. This issue is related to the Julia package used."
        ))
    end
    weights .*= exp.(points) .* (points .^ -alpha)
    GaussLaguerre(points, weights, (0, Inf))
end

raw"""
    GaussLegendre <: OneDGrid

Gauss-Legendre integral quadrature class.

The definition of Gauss-Legendre quadrature is:

.. math::
    \int_{-1}^{1} f(x) dx \approx \sum_{i=1}^n w_i f(x_i),

where :math:`w_i` are the quadrature weights and :math:`x_i` are the
roots of the nth Legendre polynomial.
"""
struct GaussLegendre <: AbstractOneDGrid
    _points::Vector{<:Real}
    weights::Vector{<:Real}
    domain::Union{Nothing,Tuple{<:Real,<:Real}}

    function GaussLegendre(
        points::Vector{<:Real},
        weights::Vector{<:Real},
        domain::Union{Nothing,Tuple{<:Real,<:Real}}=nothing)
        check_input(points, weights, domain=domain)
        new(points, weights, domain)
    end
end

raw"""
    GaussLegendre(npoints::Integer)

Generate grid on :math:`[-1, 1]` interval based on Gauss-Legendre quadrature.

# Arguments
- `npoints::Integer`: Number of grid points.

# Returns
- `::GaussLegendre`: A 1-D grid instance containing points and weights.

# Notes
- Only known to be accurate up to `npoints`=100 and may cause problems after
that amount.
"""
function GaussLegendre(npoints::Integer)
    if npoints <= 1
        throw(ArgumentError("Argument npoints must be an integer > 1, given $npoints"))
    end
    # compute points and weights for Gauss-Legendre quadrature
    # according to numpy's leggauss, the accuracy is only known up to `npoints=100`.
    points, weights = gausslegendre(npoints)
    GaussLegendre(points, weights, (-1, 1))
end


end