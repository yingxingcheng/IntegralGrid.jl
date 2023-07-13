module Utils
using LinearAlgebra, SpecialFunctions
using SphericalHarmonics

export get_cov_radii, generate_real_spherical_harmonics, convert_cart_to_sph, solid_harmonics, generate_orders_horton_order
export sph_harm, generate_derivative_real_spherical_harmonics

_bragg = [
    0.47243153,
    NaN,
    2.74010289,
    1.98421244,
    1.60626721,
    1.32280829,
    1.22832199,
    1.13383568,
    0.94486307,
    NaN,
    3.40150704,
    2.8345892,
    2.36215767,
    2.07869875,
    1.88972613,
    1.88972613,
    1.88972613,
    NaN,
    4.15739749,
    3.40150704,
    3.02356181,
    2.64561659,
    2.55113028,
    2.64561659,
    2.64561659,
    2.64561659,
    2.55113028,
    2.55113028,
    2.55113028,
    2.55113028,
    2.45664397,
    2.36215767,
    2.17318505,
    2.17318505,
    2.17318505,
    NaN,
    4.44085641,
    3.77945227,
    3.40150704,
    2.92907551,
    2.74010289,
    2.74010289,
    2.55113028,
    2.45664397,
    2.55113028,
    2.64561659,
    3.02356181,
    2.92907551,
    2.92907551,
    2.74010289,
    2.74010289,
    2.64561659,
    2.64561659,
    NaN,
    4.91328795,
    4.06291119,
    3.68496596,
    3.49599335,
    3.49599335,
    3.49599335,
    3.49599335,
    3.49599335,
    3.49599335,
    3.40150704,
    3.30702073,
    3.30702073,
    3.30702073,
    3.30702073,
    3.30702073,
    3.30702073,
    3.30702073,
    2.92907551,
    2.74010289,
    2.55113028,
    2.55113028,
    2.45664397,
    2.55113028,
    2.55113028,
    2.55113028,
    2.8345892,
    3.59047965,
    3.40150704,
    3.02356181,
    3.59047965,
    NaN,
    NaN
]

_cambridge = [
    0.5858151,
    0.52912332,
    2.41884945,
    1.81413709,
    1.58736995,
    1.43619186,
    1.34170556,
    1.24721925,
    1.0771439,
    1.09604116,
    3.13694538,
    2.66451385,
    2.28656862,
    2.09759601,
    2.02200696,
    1.98421244,
    1.92752066,
    2.0031097,
    3.83614405,
    3.325918,
    3.21253443,
    3.02356181,
    2.89128098,
    2.62671933,
    2.62671933,
    2.4944385,
    2.38105493,
    2.34326041,
    2.4944385,
    2.30546588,
    2.30546588,
    2.26767136,
    2.2487741,
    2.26767136,
    2.26767136,
    2.19208232,
    4.15739749,
    3.68496596,
    3.59047965,
    3.30702073,
    3.09915086,
    2.91017825,
    2.77789742,
    2.75900016,
    2.68341111,
    2.62671933,
    2.74010289,
    2.72120563,
    2.68341111,
    2.62671933,
    2.62671933,
    2.60782206,
    2.62671933,
    2.64561659,
    4.61093177,
    4.06291119,
    3.9117331,
    3.85504131,
    3.83614405,
    3.79834953,
    3.76055501,
    3.74165775,
    3.74165775,
    3.70386322,
    3.6660687,
    3.62827418,
    3.62827418,
    3.57158239,
    3.59047965,
    3.53378787,
    3.53378787,
    3.30702073,
    3.21253443,
    3.06135634,
    2.85348646,
    2.72120563,
    2.66451385,
    2.57002754,
    2.57002754,
    2.4944385,
    2.74010289,
    2.75900016,
    2.79679468,
    2.64561659,
    2.8345892,
    2.8345892,
]


_alvarez = [
    0.5858151,
    0.52912332,
    2.41884945,
    1.81413709,
    1.58736995,
    1.43619186,
    1.34170556,
    1.24721925,
    1.0771439,
    1.09604116,
    3.13694538,
    2.66451385,
    2.28656862,
    2.09759601,
    2.02200696,
    1.98421244,
    1.92752066,
    2.0031097,
    3.83614405,
    3.325918,
    3.21253443,
    3.02356181,
    2.89128098,
    2.62671933,
    2.62671933,
    2.4944385,
    2.38105493,
    2.34326041,
    2.4944385,
    2.30546588,
    2.30546588,
    2.26767136,
    2.2487741,
    2.26767136,
    2.26767136,
    2.19208232,
    4.15739749,
    3.68496596,
    3.59047965,
    3.30702073,
    3.09915086,
    2.91017825,
    2.77789742,
    2.75900016,
    2.68341111,
    2.62671933,
    2.74010289,
    2.72120563,
    2.68341111,
    2.62671933,
    2.62671933,
    2.60782206,
    2.62671933,
    2.64561659,
    4.61093177,
    4.06291119,
    3.9117331,
    3.85504131,
    3.83614405,
    3.79834953,
    3.76055501,
    3.74165775,
    3.74165775,
    3.70386322,
    3.6660687,
    3.62827418,
    3.62827418,
    3.57158239,
    3.59047965,
    3.53378787,
    3.53378787,
    3.30702073,
    3.21253443,
    3.06135634,
    2.85348646,
    2.72120563,
    2.66451385,
    2.57002754,
    2.57002754,
    2.4944385,
    2.74010289,
    2.75900016,
    2.79679468,
    2.64561659,
    2.8345892,
    2.8345892,
    4.91328795,
    4.17629476,
    4.06291119,
    3.89283584,
    3.77945227,
    3.70386322,
    3.59047965,
    3.53378787,
    3.40150704,
    3.19363717,
]

function get_cov_radii(atnums, type="bragg")
    raw"""Get the covalent radii for given atomic number(s).

    Parameters
    ----------
    atnums : int or np.ndarray
        atomic number of interested
    type : str, default to bragg
        types of covalent radii for elements.
        "bragg": Bragg-Slater empirically measured covalent radii
        "cambridge": Covalent radii from analysis of the Cambridge Database"
        "alvarez": Covalent radii from https://doi.org/10.1039/B801115J

    Returns
    -------
    np.ndarray
        covalent radii of desired atom(s)

    Raises
    ------
    ValueError
        Invalid covalent type, or input atomic number is 0
    """

    if typeof(atnums) <: Integer
        atnums = [atnums]
    end
    if any(atnum -> atnum == 0, atnums)
        throw(ArgumentError("0 is not a valid atomic number"))
    end
    if type == "bragg"
        return _bragg[atnums]
    elseif type == "cambridge"
        return _cambridge[atnums]
    elseif type == "alvarez"
        return _alvarez[atnums]
    else
        throw(ArgumentError("Not supported radii type, got $type"))
    end
end


function generate_real_spherical_harmonics(l_max::Int, theta::AbstractVector{<:Real}, phi::AbstractVector{<:Real})
    raw"""
    Compute the real spherical harmonics recursively up to a maximum angular degree l.

    .. math::
        Y_l^m(\theta, \phi) = \frac{(2l + 1) (l - m)!}{4\pi (l + m)!} f(m, \theta)
        P_l^m(\cos(\phi)),

    where :math:`l` is the angular degree, :math:`m` is the order and
    :math:`f(m, \theta) = \sqrt{2} \cos(m \theta)` when :math:`m>0` otherwise
    :math:`f(m, \theta) = \sqrt{2} \sin(m\theta)`
    when :math:`m<0`, and equal to one when :math:`m= 0`.  :math:`P_l^m` is the associated
    Legendre polynomial without the Conway phase factor.
    The angle :math:`\theta \in [0, 2\pi]` is the azimuthal angle and :math:`\phi \in [0, \pi]`
    is the polar angle.

    Parameters
    ----------
    l_max : int
        Largest angular degree of the spherical harmonics.
    theta : np.ndarray(N,)
        Azimuthal angle :math:`\theta \in [0, 2\pi]` that are being evaluated on.
        If this angle is outside of bounds, then periodicity is used.
    phi : np.ndarray(N,)
        Polar angle :math:`\phi \in [0, \pi]` that are being evaluated on.
        If this angle is outside of bounds, then periodicity is used.

    Returns
    -------
    ndarray((l_max + 1)**2, N)
        Value of real spherical harmonics of all orders :math:`m`,and degree
        :math:`l` spherical harmonics. For each degree :math:`l`, the orders :math:`m` are
        in Horton 2 order, i.e. :math:`m=0, 1, -1, 2, -2, \cdots, l, -l`.

    Notes
    -----
    - The associated Legendre polynomials are computed using the forward recursion:
      :math:`P_l^m(\phi) = \frac{2l + 1}{l - m + 1}\cos(\phi) P_{l-1}^m(x) -
      \frac{(l + m)}{l - m + 1} P_{l-1}^m(x)`, and
      :math:`P_l^l(\phi) = (2l + 1) \sin(\phi) P_{l-1}^{l-1}(\phi)`.
    - For higher maximum degree :math:`l_{max} > 1900` with double precision the computation
      of spherical harmonic will underflow within the range
      :math:`20^\circ \leq \phi \leq 160^\circ`.  This code does not implement the
      modified recursion formulas in [2] and instead opts for higher precision defined
      by the computer system and np.longdouble.
    - The mapping from :math:`(l, m)` to the correct row index in the spherical harmonic array is
      given by :math:`(l + 1)^2 + 2 * m - 1` if :math:`m > 0`, else it is :math:`(l+1)^2 + 2 |m|`.

    References
    ----------
    .. [1] Colombo, Oscar L. Numerical methods for harmonic analysis on the sphere.
       Ohio State Univ Columbus Dept of Geodetic Science And Surveying, 1981.
    .. [2] Holmes, Simon A., and Will E. Featherstone. "A unified approach to the Clenshaw
       summation and the recursive computation of very high degree and order normalised
       associated Legendre functions." Journal of Geodesy 76.5 (2002): 279-299.

    """

    numb_pts = length(theta)
    sin_phi = sin.(phi)
    cos_phi = cos.(phi)
    spherical_harm = zeros((l_max + 1)^2, numb_pts)

    # Forward recursion requires P_{l-1}^m, P_{l-2}^m, these are the two columns, respectively
    # the rows are the order m which ranges from 0 to l_max and p_leg[:l, :] gets updated every l
    p_leg = zeros(l_max + 1, 2, numb_pts)
    p_leg[1, :, :] .= 1.0  # Initial conditions: P_0^0 = 1.0

    # the coefficients of the forward recursions and initial factor of spherical harmonic.
    a_k(deg, ord) = (2.0 * (deg - 1.0) + 1) / (deg - 1.0 - ord + 1.0)
    b_k(deg, ord) = (deg - 1.0 + ord) / (deg - ord)
    fac_sph(deg, ord) = sqrt((2.0 * deg + 1) / (4.0 * π))  # Note (l-m)!/(l+m)! is moved

    # Go through each degree and then order and fill out
    spherical_harm[1, :] .= fac_sph(0.0, 0.0)  # Compute Y_0^0
    i_sph = 2  # Index to start of spherical_harm
    factorial = 0.0
    for l_deg in 1:l_max
        for m_ord in 0:l_deg
            if l_deg == m_ord
                # Do diagonal spherical harmonic Y_l^m, when l=m.
                # Diagonal recursion: P_m^m = sin(phi) * P_{m-1}^{m-1} * (2 (l - 1) + 1)
                p_leg[m_ord+1, 1, :] = p_leg[m_ord, 2, :] .* (2 * (l_deg - 1.0) + 1) .* sin_phi
            else
                # Do forward recursion here and fill out Y_l^m and Y_l^{-m}
                # Compute b_k P_{l-2}^m,  since m < l, then m < l - 2
                second_fac = m_ord <= l_deg - 2 ? b_k(l_deg, m_ord) .* p_leg[m_ord+1, 2, :] : 0.0
                # Update/re-define P_{l-2}^m to be equal to P_{l-1}^m
                p_leg[m_ord+1, 2, :] = p_leg[m_ord+1, 1, :]
                # Then update P_{l-1}^m := P_l^m := a_k cos(\phi) P_{l-1}^m - b_k P_{l, -2}^m
                p_leg[m_ord+1, 1, :] = a_k(l_deg, m_ord) .* cos_phi .* p_leg[m_ord+1, 1, :] .- second_fac
            end
            # Compute Y_l^{m} that has cosine(theta) and Y_l^{-m} that has sin(theta)
            if m_ord == 0
                # init factorial needed to compute (l-m)!/(l+m)!
                factorial = (l_deg + 1.0) * l_deg
                spherical_harm[i_sph, :] = fac_sph(l_deg, m_ord) .* p_leg[m_ord+1, 1, :]
            else
                common_fact = (
                    (p_leg[m_ord+1, 1, :] ./ sqrt(factorial)) .* fac_sph(l_deg, m_ord) .* sqrt(2.0)
                )
                spherical_harm[i_sph, :] = common_fact .* cos.(m_ord .* theta)
                i_sph += 1
                spherical_harm[i_sph, :] = common_fact .* sin.(m_ord .* theta)
                # Update (l-m)!/(l+m)!
                factorial *= (l_deg + m_ord + 1.0) * (l_deg - m_ord)
            end
            i_sph += 1
        end
    end
    return spherical_harm
end

generate_real_spherical_harmonics(; l_max, theta, phi) = generate_real_spherical_harmonics(l_max, theta, phi)


function sph_harm(m::Int, l::Int, theta::Real, phi::Real)
    # In scipy, phi is polar, theta is azimuth angle, while in SphericalHarmonics, theta is polar, phi is azimuth angle
    return (m > l || m < -l) ? nothing : computeYlm(phi, theta, lmax=l, m=m)[(l, m)]
end

function sph_harm(m::Int, l::Int, theta::AbstractVector{<:Real}, phi::AbstractVector{<:Real})
    if abs(m) > l
        return fill(Inf, size(theta))
    end

    out = computeYlm.(phi, theta, lmax=l, m=m)
    res = [ele[(l, m)] for ele in out]
    return res
end

function sph_harm(m::AbstractVector{Int}, l::AbstractVector{Int}, theta::AbstractVector{<:Real}, phi::AbstractVector{<:Real})
    res = similar(theta)
    for (i, (m_i, l_i, theta_i, phi_i)) in enumerate(zip(m, l, theta, phi))
        res[i] = sph_harm(m_i, l_i, theta_i, phi_i)
    end
    return res
end


function generate_derivative_real_spherical_harmonics(l_max::Int, theta::AbstractVector{<:Real}, phi::AbstractVector{<:Real})
    raw"""
    Generate derivatives of real spherical harmonics.

    If ϕ is zero, then the first component of the derivative wrt to
    ϕ is set to zero.

    Parameters
    ----------
    l_max : int
        Largest angular degree of the spherical harmonics.
    theta : Array{Float64}
        Azimuthal angle θ ∈ [0, 2π] that are being evaluated on.
        If this angle is outside of bounds, then periodicity is used.
    phi : Array{Float64}
        Polar angle ϕ ∈ [0, π] that are being evaluated on.
        If this angle is outside of bounds, then periodicity is used.

    Returns
    -------
    Array{Float64, 3}
        Derivative of spherical harmonics, (theta first, then phi) of all degrees up to
        l_max and orders m in Horton 2 order, i.e.
        m = 0, 1, -1, ⋯, l, -l.
    """

    num_pts = length(theta)
    # Shape (Derivs, Spherical, Pts)
    output = zeros(Float64, 2, (l_max + 1)^2, num_pts)

    complex_expon = exp.(-theta * im)  # Needed for derivative wrt to phi
    l_list = 0:l_max
    sph_harm_vals = generate_real_spherical_harmonics(l_max, theta, phi)
    i_output = 1
    for l_val in l_list
        for m in vcat(0, [i for i in Iterators.flatten([(i, -i) for i in 1:l_val])])
            # Take all spherical harmonics at degree l_val
            sph_harm_degree = sph_harm_vals[(l_val)^2+1:(l_val+1)^2, :]

            # Take derivative wrt to theta:
            # for complex spherical harmonic it is   i m Y^m_l,
            # Note ie^(i |m| x) = -sin(|m| x) + i cos(|m| x), then take real/imaginary component.
            # hence why the negative is in (-m).
            # index_m maps m to index where (l, m)  is located in `sph_harm_degree`.
            index_m(m) = m > 0 ? 2 * m : Int(2 * abs(m)) + 1

            output[1, i_output, :] = -m * sph_harm_degree[index_m(-m), :]

            # Take derivative wrt to phi:
            cot_tangent = 1.0 ./ tan.(phi)
            cot_tangent[abs.(tan.(phi)).<1e-10] .= 0.0
            # Calculate the derivative in two pieces:
            fac = sqrt.((l_val - abs(m)) .* (l_val + abs(m) + 1))
            output[2, i_output, :] = abs(m) .* cot_tangent .* sph_harm_degree[index_m(m), :]
            # Compute it using SciPy, removing conway phase (-1)^m and multiply by 2^0.5.
            sph_harm_m = (
                fac
                * sph_harm(abs(m) + 1, l_val, theta, phi)
                * sqrt(2)
                * (-1.0)^m)
            if m >= 0
                if m < l_val  # When m == l_val, then fac = 0
                    output[2, i_output, :] .+= real.(complex_expon .* sph_harm_m)
                end
            elseif m < 0
                # generate order m=negative real spherical harmonic
                if m > -l_val
                    output[2, i_output, :] .+= imag.(complex_expon .* sph_harm_m)
                end
            end
            if m == 0
                # sqrt(2.0) isn't included in Y_l^m only m ≠ 0
                output[2, i_output, :] ./= sqrt(2.0)
            end
            i_output += 1
        end
    end
    return output
end


function solid_harmonics(l_max::Int, sph_pts::Array{Float64,2})
    raw"""
    Generate the solid harmonics from zero to a maximum angular degree.

    R_l^m(r, θ, ϕ) = √((4π)/(2l + 1)) r^l Y_l^m(θ, ϕ)

    Parameters
    ----------
    l_max : int
        Largest angular degree of the spherical harmonics.
    sph_pts : Array{Float64, 2}
        Three-dimensional points in spherical coordinates (r, θ, ϕ), where
        r ≥ 0, θ ∈ [0, 2π], and ϕ ∈ [0, π].

    Returns
    -------
    Array{Float64, 2}
        The solid harmonics from degree l=0, ⋯, l_max and for all orders m,
        ordered as m=0, 1, -1, 2, -2, ⋯, l, -l evaluated on M points.
    """

    r, theta, phi = sph_pts[:, 1], sph_pts[:, 2], sph_pts[:, 3]
    spherical_harm = generate_real_spherical_harmonics(l_max, theta, phi)
    degrees = [Float64(l_deg) for l_deg in 0:l_max for _ in 1:(2*l_deg+1)]
    return (
        spherical_harm .* r' .^ degrees .* sqrt.(4.0 * π ./ (2 .* degrees .+ 1))
    )
end


function convert_derivative_from_spherical_to_cartesian(deriv_r::Real, deriv_theta::Real, deriv_phi::Real, r::Real, theta::Real, phi::Real)
    jacobian = reshape([
            cos(theta) * sin(phi),
            -sin(theta) / (r * sin(phi)),
            cos(theta) * cos(phi) / r,
            sin(theta) * sin(phi),
            cos(theta) / (r * sin(phi)),
            sin(theta) * cos(phi) / r,
            cos(phi),
            0.0,
            -sin(phi) / r,
        ], (3, 3))
    # If the radial component is zero, then put all zeros on the derivs of theta and phi
    if abs(r) < 1e-10
        jacobian[:, 2] .= 0.0
        jacobian[:, 3] .= 0.0
    end
    # If phi angle is zero, then set the derivative wrt to theta to zero
    if abs(phi) < 1e-10
        jacobian[:, 2] .= 0.0
    end
    return jacobian * [deriv_r, deriv_theta, deriv_phi]
end

function convert_derivative_from_spherical_to_cartesian(deriv_r::AbstractVector{<:Real}, deriv_theta::AbstractVector{<:Real}, deriv_phi::AbstractVector{<:Real}, r::AbstractVector{<:Real}, theta::AbstractVector{<:Real}, phi::AbstractVector{<:Real})
    return [convert_derivative_from_spherical_to_cartesian(dr, dt, dp, rr, tt, pp) for (dr, dt, dp, rr, tt, pp) in zip(deriv_r, deriv_theta, deriv_phi, r, theta, phi)]
end


function convert_cart_to_sph(points, center=nothing)
    raw"""
    Convert a set of points from cartesian to spherical coordinates.

    The convention that θ ∈ [-π, π] and φ ∈ [0, π) is chosen.

    Parameters
    ----------
    points : Array{Float64}(n, 3)
        The (3-dimensional) Cartesian coordinates of points.
    center : Array{Float64}(3,), optional
        Cartesian coordinates of the center of spherical coordinate system.
        If `nothing`, origin is used.

    Returns
    -------
    Array{Float64}(N, 3)
        Spherical coordinates of atoms respect to the center
        [radius r, azumuthal θ, polar φ]

    """
    if ndims(points) != 2 || size(points)[2] != 3
        throw(ArgumentError("points array requires shape (N, 3), got: ($(ndims(points)), $(size(points)[2]) )"))
    end

    center = isnothing(center) ? zeros(Float64, 3) : center
    if length(center) != 3
        throw(ArgumentError("center needs to be of length (3), got:$(length(center))"))
    end

    relat_pts = points .- center'
    # compute r
    r = vec(norm.(eachrow(relat_pts)))
    # polar angle: arccos(z / r)
    phi = @. acos(relat_pts[:, 3] / r)
    # fix NaN generated when point is [0.0, 0.0, 0.0]
    phi[r.==0.0] .= 0.0
    # azimuthal angle arctan2(y / x)
    theta = atan.(relat_pts[:, 2], relat_pts[:, 1])
    return hcat(r, theta, phi)
end

function generate_orders_horton_order(order::Int, type_ord::String, dim::Int=3)
    # Check input type
    if !isa(order, Int)
        throw(ArgumentError("Order $order should be integer type."))
    end
    if !(type_ord in ["cartesian", "radial", "pure", "pure-radial"])
        throw(ArgumentError("Type $type_ord is not recognized."))
    end

    orders = []
    if type_ord == "cartesian"
        if dim == 3
            for m_x in order:-1:0
                for m_y in order-m_x:-1:0
                    push!(orders, [m_x, m_y, order - m_x - m_y])
                end
            end
        elseif dim == 2
            for m_x in order:-1:0
                push!(orders, [m_x, order - m_x])
            end
        elseif dim == 1
            return Array(0:order)
        else
            throw(ArgumentError("dim $dim parameter should be either 1, 2, 3."))
        end
    elseif type_ord == "radial"
        return [order]
    elseif type_ord == "pure"
        # Add the (l, 0)
        push!(orders, [order, 0])
        # Add orders (i, l) (i, -l) i=1, to l
        for x in 1:order
            append!(orders, [[order, x], [order, -x]])
        end
    elseif type_ord == "pure-radial"
        # Generate (n, l=0,1, 2 ..., (n-1), m=0, 1, -1, ... l -l)
        for l_deg in 0:order
            for m_ord in 0:l_deg
                if m_ord != 0
                    append!(orders, [[order, l_deg, m_ord], [order, l_deg, -m_ord]])
                else
                    push!(orders, [order, l_deg, m_ord])
                end
            end
        end
    else
        throw(ArgumentError("Type $type_ord is not recognized."))
    end
    orders = hcat(orders...)
    return orders'
end


end
