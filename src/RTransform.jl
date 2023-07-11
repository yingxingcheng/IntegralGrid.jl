module RTransform

using IntegralGrid.BaseGrid

export BaseTransform, transform, inverse, deriv, deriv2, deriv3, domain, codomain, transform_1d_grid
export IdentityRTransform, LinearInfiniteRTransform, ExpRTransform, PowerRTransform, HyperbolicRTransform

# TODO: unittests
export BeckeRTransform, LinearFiniteRTransform, InverseRTransform
export MultiExpRTransform, KnowlesRTransform, HandyRTransform, HandyModRTransform


const TypePoints1D = Union{AbstractVector{<:Real},<:Real}

#--------------------------------------------------------------------------------------------------
# BaseTransform
#--------------------------------------------------------------------------------------------------
abstract type BaseTransform end
transform(rtf::BaseTransform, x::TypePoints1D) = error("Not implemented for $(typeof(tf))!")
inverse(rtf::BaseTransform, r::TypePoints1D) = error("Not implemented for $(typeof(tf))!")
deriv(rtf::BaseTransform, x::TypePoints1D) = error("Not implemented for $(typeof(tf))!")
deriv2(rtf::BaseTransform, x::TypePoints1D) = error("Not implemented for $(typeof(tf))!")
deriv3(rtf::BaseTransform, x::TypePoints1D) = error("Not implemented for $(typeof(tf))!")

function transform_1d_grid(rtf::BaseTransform, oned_grid::OneDGrid)
    # check domain
    if oned_grid.domain[1] < rtf.domain[1] || oned_grid.domain[2] > rtf.domain[2]
        throw(ArgumentError("Given 1D grid domain does not match the transformation domain.\n grid domain: $(oned_grid.domain),  rtf domain: , $(rtf.domain)"))
    end

    new_points = transform(rtf, oned_grid.points)
    new_weights = deriv(rtf, oned_grid.points) .* oned_grid.weights
    new_domain = oned_grid.domain
    if new_domain !== nothing
        # Some transformation (Issue #125) reverses the order of points i.e.
        #    [-1, 1] maps to [infinity, 0].  This sort here fixes the problem here.
        new_domain = sort(transform(rtf, collect(oned_grid.domain)))
        new_domain = tuple(new_domain...)
    end
    return OneDGrid(new_points, new_weights, new_domain)
end

function _convert_inf!(rtf::BaseTransform, array::Union{<:Number,TypePoints1D}; replace_inf=1e16)
    if typeof(array) <: Number
        new_v = isinf(array) ? sign(array) * replace_inf : array
    else
        new_v = similar(array)
        new_v[new_v.==Inf] .= replace_inf
        new_v[new_v.==-Inf] .= -replace_inf
    end
    return new_v
end

#--------------------------------------------------------------------------------------------------
# BeckeRTransform
#--------------------------------------------------------------------------------------------------
mutable struct BeckeRTransform <: BaseTransform
    rmin::Real
    R::Real
    trim_inf::Bool
    domain::Tuple{<:Real,<:Real}
    codomain::Tuple{<:Real,<:Real}

end
BeckeRTransform(rmin::Real, R::Real) = BeckeRTransform(rmin, R, true, (-1, 1), (rmin, Inf))

function find_parameter(array::TypePoints1D, rmin::Float64, radius::Float64)
    if rmin > radius
        error("rmin needs to be smaller than radius, rmin: $rmin, radius: $radius.")
    end
    size = length(array)
    mid_value = ifelse(isodd(size), array[size÷2], (array[size÷2-1] + array[size÷2]) / 2)
    return (radius - rmin) * (1 - mid_value) / (1 + mid_value)
end

function transform(rtf::BeckeRTransform, x::TypePoints1D)
    rf_array = rtf.R * (1 .+ x) ./ (1 .- x) .+ rtf.rmin
    if rtf.trim_inf
        rf_array = _convert_inf!(rtf, rf_array)
    end
    return rf_array
end

inverse(rtf::BeckeRTransform, r::TypePoints1D) = (r .- rtf.rmin .- rtf.R) ./ (r .- rtf.rmin .+ rtf.R)

function deriv(rtf::BeckeRTransform, x::TypePoints1D)
    deriv = 2 * rtf.R ./ ((1 .- x) .^ 2)
    if rtf.trim_inf
        deriv = _convert_inf!(rtf, deriv)
    end
    return deriv
end

deriv2(rtf::BeckeRTransform, x::TypePoints1D) = 4 * rtf.R ./ (1 .- x) .^ 3
deriv3(rtf::BeckeRTransform, x::TypePoints1D) = 12 * rtf.R ./ (1 .- x) .^ 4

function _convert_inf!(rtf::BeckeRTransform, array::TypePoints1D; replace_inf=1e16)
    if typeof(array) <: Real
        new_v = isinf(array) ? sign(array) * replace_inf : array
    else
        new_v = similar(array)
        @. new_v = ifelse(array == Inf, replace_inf, array)
        @. new_v = ifelse(array == -Inf, -replace_inf, array)
    end
    return new_v
end


#--------------------------------------------------------------------------------------------------
# LinearFiiniteRTransform
#--------------------------------------------------------------------------------------------------
mutable struct LinearFiniteRTransform <: BaseTransform
    rmin::Real
    rmax::Real
    domain::Tuple{<:Real,<:Real}
    codomain::Tuple{<:Real,<:Real}
end

transform(rtf::LinearFiniteRTransform, x::TypePoints1D) = (1 .+ x) * (rtf.rmax - rtf.rmin) / 2 .+ rtf.rmin
deriv(rtf::LinearFiniteRTransform, x::TypePoints1D) = fill((rtf.rmax - rtf.rmin) / 2, length(x))
deriv2(rtf::LinearFiniteRTransform, x::TypePoints1D) = zeros(length(x))
deriv3(rtf::LinearFiniteRTransform, x::TypePoints1D) = zeros(length(x))
inverse(rtf::LinearFiniteRTransform, r::TypePoints1D) = (2 .* r .- (rtf.rmax + rtf.rmin)) ./ (rtf.rmax - rtf.rmin)


#--------------------------------------------------------------------------------------------------
# InverseRTransform
#--------------------------------------------------------------------------------------------------
mutable struct InverseRTransform <: BaseTransform
    tfm::BaseTransform
    domain::Tuple{<:Real,<:Real}
    codomain::Tuple{<:Real,<:Real}

    function InverseRTransform(tfm::BaseTransform)
        return new(tfm, tfm.domain, tfm.codomain)
    end
end


function transform(rtf::InverseRTransform, r::TypePoints1D)
    return inverse(rtf.tfm, r)
end

function inverse(rtf::InverseRTransform, x::TypePoints1D)
    return transform(rtf.tfm, x)
end

function _d1(rtf::InverseRTransform, r::TypePoints1D)
    d1 = deriv(rtf.tfm, r)
    if any(d1 .== 0)
        throw(ArgumentError("First derivative of original transformation has 0 value"))
    end
    return d1
end

function deriv(rtf::InverseRTransform, r::TypePoints1D)
    r = inverse(rtf.tfm, r)
    return 1 ./ _d1(rtf, r)
end

function deriv2(rtf::InverseRTransform, r::TypePoints1D)
    r = inverse(rtf.tfm, r)
    d2 = deriv2(rtf.tfm, r)
    return -d2 ./ (_d1(rtf, r) .^ 3)
end

function deriv3(rtf::InverseRTransform, r::TypePoints1D)
    r = inverse(rtf.tfm, r)
    d1 = deriv(rtf.tfm, r)
    d2 = deriv2(rtf.tfm, r)
    d3 = deriv3(rtf.tfm, r)
    return (3 * d2 .^ 2 - d1 .* d3) ./ (_d1(tf, r) .^ 5)
end

#--------------------------------------------------------------------------------------------------
# IdentityRTransform
#--------------------------------------------------------------------------------------------------
mutable struct IdentityRTransform <: BaseTransform
    domain::Tuple{<:Real,<:Real}
    codomain::Tuple{<:Real,<:Real}

end
IdentityRTransform() = IdentityRTransform((0, Inf), (0, Inf))

transform(tf::IdentityRTransform, x::TypePoints1D) = x
deriv(tf::IdentityRTransform, x::TypePoints1D) = typeof(x) <: Real ? 1 : ones(length(x))
deriv2(tf::IdentityRTransform, x::TypePoints1D) = typeof(x) <: Real ? 0 : zeros(length(x))
deriv3(tf::IdentityRTransform, x::TypePoints1D) = typeof(x) <: Real ? 0 : zeros(length(x))
inverse(tf::IdentityRTransform, r::TypePoints1D) = r

#--------------------------------------------------------------------------------------------------
# LinearInfiniteRTransform
#--------------------------------------------------------------------------------------------------
mutable struct LinearInfiniteRTransform <: BaseTransform
    rmin::Real
    rmax::Real
    b::Union{<:Real,Nothing}
    domain::Tuple{<:Real,<:Real}
    codomain::Tuple{<:Real,<:Real}
end

function LinearInfiniteRTransform(rmin::Real, rmax::Real, b::Union{Float64,Nothing}=nothing)
    if rmin >= rmax
        throw(ArgumentError("rmin needs to be larger than rmax.\n  rmin: $rmin, rmax: $rmax"))
    end
    return LinearInfiniteRTransform(rmin, rmax, b, (0, Inf), (rmin, rmax))
end

function set_maximum_parameter_b!(rtf::LinearInfiniteRTransform, x::TypePoints1D)
    if isnothing(rtf.b)
        rtf.b = maximum(x)
        if abs(rtf.b) < 1e-16
            throw(ArgumentError("The parameter b $(rtf.b) is taken from the maximum of the grid and can't be zero."))
        end
    end
end

function transform(tf::LinearInfiniteRTransform, x::TypePoints1D)
    set_maximum_parameter_b!(tf, x)
    alpha = (tf.rmax - tf.rmin) / tf.b
    return alpha .* x .+ tf.rmin
end

function deriv(tf::LinearInfiniteRTransform, x::TypePoints1D)
    set_maximum_parameter_b!(tf, x)
    alpha = (tf.rmax - tf.rmin) / tf.b
    return fill(alpha, length(x))
end

deriv2(tf::LinearInfiniteRTransform, x::TypePoints1D) = zeros(length(x))
deriv3(tf::LinearInfiniteRTransform, x::TypePoints1D) = zeros(length(x))

function inverse(tf::LinearInfiniteRTransform, r::TypePoints1D)
    set_maximum_parameter_b!(tf, r)
    alpha = (tf.rmax - tf.rmin) / tf.b
    return (r .- tf.rmin) / alpha
end

#--------------------------------------------------------------------------------------------------
# ExpRTransform
#--------------------------------------------------------------------------------------------------
mutable struct ExpRTransform <: BaseTransform
    rmin::Real
    rmax::Real
    b::Union{<:Real,Nothing}
    domain::Tuple{<:Real,<:Real}
    codomain::Tuple{<:Real,<:Real}

end

function ExpRTransform(rmin::Real, rmax::Real, b::Union{<:Real,Nothing}=nothing)
    if rmin < 0 || rmax < 0
        throw(ArgumentError("rmin or rmax need to be positive\n  rmin: $rmin, rmax: $rmax"))
    end
    if rmin >= rmax
        throw(ArgumentError("rmin need to be smaller than rmax\n  rmin: $rmin, rmax: $rmax"))
    end
    ExpRTransform(rmin, rmax, b, (0, Inf), (rmin, rmax))
end

function set_maximum_parameter_b!(rtf::ExpRTransform, x::TypePoints1D)
    if isnothing(rtf.b)
        rtf.b = maximum(x)
        if abs(rtf.b) < 1e-16
            throw(ArgumentError("The parameter b $(rtf.b) is taken from the maximum of the grid and can't be zero."))
        end
    end
end

function transform(rtf::ExpRTransform, x::TypePoints1D)
    set_maximum_parameter_b!(rtf, x)
    alpha = log(rtf.rmax / rtf.rmin) / rtf.b
    return rtf.rmin * exp.(x * alpha)
end

function deriv(rtf::ExpRTransform, x::TypePoints1D)
    set_maximum_parameter_b!(rtf, x)
    alpha = log(rtf.rmax / rtf.rmin) / rtf.b
    return transform(rtf, x) .* alpha
end

function deriv2(rtf::ExpRTransform, x::TypePoints1D)
    set_maximum_parameter_b!(rtf, x)
    alpha = log(rtf.rmax / rtf.rmin) / rtf.b
    return deriv(rtf, x) .* alpha
end

function deriv3(rtf::ExpRTransform, x::TypePoints1D)
    set_maximum_parameter_b!(rtf, x)
    alpha = log(rtf.rmax / rtf.rmin) / rtf.b
    return deriv2(rtf, x) .* alpha
end

function inverse(rtf::ExpRTransform, r::TypePoints1D)
    set_maximum_parameter_b!(rtf, r)
    alpha = log(rtf.rmax / rtf.rmin) / rtf.b
    return log.(r ./ rtf.rmin) ./ alpha
end

#--------------------------------------------------------------------------------------------------
# PowerRTransform
#--------------------------------------------------------------------------------------------------
mutable struct PowerRTransform <: BaseTransform
    rmin::Real
    rmax::Real
    b::Union{<:Real,Nothing}
    domain::Tuple{<:Real,<:Real}
    codomain::Tuple{<:Real,<:Real}
end

function PowerRTransform(rmin::Real, rmax::Real, b::Union{<:Real,Nothing}=nothing)
    if rmin >= rmax
        throw(ArgumentError("rmin must be smaller rmax."))
    end
    if rmin <= 0 || rmax <= 0
        throw(ArgumentError("rmin and rmax must be positive."))
    end
    PowerRTransform(rmin, rmax, b, (0, Inf), (rmin, rmax))
end

function set_maximum_parameter_b!(rtf::PowerRTransform, x::TypePoints1D)
    if isnothing(rtf.b)
        rtf.b = maximum(x)
        if abs(rtf.b) < 1e-16
            throw(ArgumentError("The parameter b $(rtf.b) is taken from the maximum of the grid and can't be zero."))
        end
    end
end

function transform(rtf::PowerRTransform, x::TypePoints1D)
    set_maximum_parameter_b!(rtf, x)
    power = (log(rtf.rmax) - log(rtf.rmin)) / log(rtf.b + 1)
    if power < 2
        @warn "Power need to be larger than 2!"
    end
    return rtf.rmin * (x .+ 1) .^ power
end

function deriv(rtf::PowerRTransform, x::TypePoints1D)
    set_maximum_parameter_b!(rtf, x)
    power = (log(rtf.rmax) - log(rtf.rmin)) / log(rtf.b + 1)
    return power * rtf.rmin .* (x .+ 1) .^ (power - 1)
end

function deriv2(rtf::PowerRTransform, x::TypePoints1D)
    set_maximum_parameter_b!(rtf, x)
    power = (log(rtf.rmax) - log(rtf.rmin)) / log(rtf.b + 1)
    return power * (power - 1) * rtf.rmin .* (x .+ 1) .^ (power - 2)
end

function deriv3(rtf::PowerRTransform, x::TypePoints1D)
    set_maximum_parameter_b!(rtf, x)
    power = (log(rtf.rmax) - log(rtf.rmin)) / log(rtf.b + 1)
    return power * (power - 1) * (power - 2) * rtf.rmin .* (x .+ 1) .^ (power - 3)
end

function inverse(rtf::PowerRTransform, r::TypePoints1D)
    set_maximum_parameter_b!(rtf, r)
    power = (log(rtf.rmax) - log(rtf.rmin)) / log(rtf.b + 1)
    return (r ./ rtf.rmin) .^ (1.0 / power) .- 1
end


#--------------------------------------------------------------------------------------------------
# HyperbolicRTransform
#--------------------------------------------------------------------------------------------------
mutable struct HyperbolicRTransform <: BaseTransform
    a::Real
    b::Real
    domain::Tuple{<:Real,<:Real}
    codomain::Tuple{<:Real,<:Real}
end

function HyperbolicRTransform(a::Real, b::Real)
    if a <= 0
        throw(ArgumentError("a must be strictly positive."))
    end
    if b <= 0
        throw(ArgumentError("b must be strictly positive."))
    end
    return HyperbolicRTransform(a, b, (0, Inf), (0, Inf))
end


function transform(rtf::HyperbolicRTransform, x::TypePoints1D)
    if rtf.b * (length(x) - 1) >= 1.0
        throw(ArgumentError("b*(npoint-1) must be smaller than one."))
    end
    return rtf.a * x ./ (1 .- rtf.b .* x)
end

function deriv(rtf::HyperbolicRTransform, x::TypePoints1D)
    if rtf.b * (length(x) - 1) >= 1.0
        throw(ArgumentError("b*(npoint-1) must be smaller than one."))
    end
    x_vals = 1.0 ./ (1 .- rtf.b .* x)
    return rtf.a .* x_vals .* x_vals
end

function deriv2(rtf::HyperbolicRTransform, x::TypePoints1D)
    if rtf.b * (length(x) - 1) >= 1.0
        throw(ArgumentError("b*(npoint-1) must be smaller than one."))
    end
    x_vals = 1.0 ./ (1 .- rtf.b .* x)
    return 2.0 * rtf.a * rtf.b * x_vals .^ 3
end

function deriv3(rtf::HyperbolicRTransform, x::TypePoints1D)
    if rtf.b * (length(x) - 1) >= 1.0
        throw(ArgumentError("b*(npoint-1) must be smaller than one."))
    end
    x_vals = 1.0 ./ (1 .- rtf.b .* x)
    return 6.0 * rtf.a * rtf.b * rtf.b * x_vals .^ 4
end

function inverse(rtf::HyperbolicRTransform, r::TypePoints1D)
    if rtf.b * (length(r) - 1) >= 1.0
        throw(ArgumentError("b*(npoint-1) must be smaller than one."))
    end
    return r ./ (rtf.a .+ rtf.b .* r)
end

#--------------------------------------------------------------------------------------------------
# MultiExpRTransform
#--------------------------------------------------------------------------------------------------
mutable struct MultiExpRTransform <: BaseTransform
    rmin::Real
    R::Real
    trim_inf::Bool
    domain::Tuple{<:Real,<:Real}
    codomain::Tuple{<:Real,<:Real}
end

function MultiExpRTransform(rmin::Real, R::Real; trim_inf::Bool=true)
    return MultiExpRTransform(rmain, R, trim_inf, (-1, 1), (rmain, Inf))
end

function transform(rtf::MultiExpRTransform, x::TypePoints1D)
    rf_array = -rtf.R * log.((x .+ 1) ./ 2) + rtf.rmin
    if rtf.trim_inf
        rf_array = _convert_inf!(rtf, rf_array)
    end
    return rf_array
end

function inverse(rtf::MultiExpRTransform, r::TypePoints1D)
    return 2 * exp.(-(r .- rtf.rmin) ./ rtf.R) .- 1
end

function deriv(rtf::MultiExpRTransform, x::TypePoints1D)
    return -rtf.R ./ (1 .+ x)
end

function deriv2(rtf::MultiExpRTransform, x::TypePoints1D)
    return rtf.R ./ (1 .+ x) .^ 2
end

function deriv3(rtf::MultiExpRTransform, x::TypePoints1D)
    return -2 * rtf.R ./ (1 .+ x) .^ 3
end

function _convert_inf!(rtf::MultiExpRTransform, arr::TypePoints1D)
    arr[isinf.(arr)] = 1e16
    return arr
end


#--------------------------------------------------------------------------------------------------
# KnowlesRTransform
#--------------------------------------------------------------------------------------------------
mutable struct KnowlesRTransform <: BaseTransform
    rmin::Real
    R::Real
    k::Int
    trim_inf::Bool
    domain::Tuple{<:Real,<:Real}
    codomain::Tuple{<:Real,<:Real}
end

function KnowlesRTransform(rmin::Real, R::Real, k::Int; trim_inf::Bool=true)
    if k <= 0
        throw(ArgumentError("k needs to be greater than 0, got k = $k"))
    end
    KnowlesRTransform(rmin, R, k, trim_inf, (-1, 1), (rmin, Inf))
end

function transform(rtf::KnowlesRTransform, x::TypePoints1D)
    rf_array = -rtf.R * log.(1 .- 2 .^ (-rtf.k) .* (x .+ 1) .^ rtf.k) .+ rtf.rmin
    if rtf.trim_inf
        rf_array = _convert_inf!(rtf, rf_array)
    end
    return rf_array
end

function inverse(rtf::KnowlesRTransform, r::TypePoints1D)
    return -1 .+ 2 .* ((rtf.rmin .- r) ./ rtf.R) .^ (1 / rtf.k)
end

function deriv(rtf::KnowlesRTransform, x::TypePoints1D)
    qi = 1 .+ x
    deriv = rtf.R * rtf.k * (qi .^ (rtf.k - 1)) ./ (2 .^ rtf.k .- qi .^ rtf.k)
    if rtf.trim_inf
        deriv = _convert_inf!(rtf, rf_array)
    end
    return deriv
end

function deriv2(rtf::KnowlesRTransform, x::TypePoints1D)
    qi = 1 .+ x
    dr = (
        rtf.R
        * rtf.k
        * (qi .^ (rtf.k - 2))
        .*
        (2^rtf.k * (rtf.k - 1) .+ qi .^ rtf.k)
        ./
        (2^rtf.k .- qi .^ rtf.k) .^ 2
    )
    return dr
end

function deriv3(rtf::KnowlesRTransform, x::TypePoints1D)
    qi = 1 .+ x
    dr = (
        rtf.R
        * rtf.k
        * (qi .^ (rtf.k - 3))
        .*
        (
            4^rtf.k * (rtf.k - 2) * (rtf.k - 1)
            .+
            2^rtf.k * (rtf.k - 1) * (rtf.k + 4) .* qi .^ rtf.k
            .+
            2 * qi .^ (2 * rtf.k)
        )
        ./
        (2^rtf.k .- qi .^ rtf.k) .^ 3
    )
    return dr
end

#--------------------------------------------------------------------------------------------------
# HandyRTransform
#--------------------------------------------------------------------------------------------------
mutable struct HandyRTransform <: BaseTransform
    rmin::Real
    R::Real
    m::Int
    trim_inf::Bool
    domain::Tuple{<:Real,<:Real}
    codomain::Tuple{<:Real,<:Real}
end

function HandyRTransform(rmin::Real, R::Real, m::Int; trim_inf::Bool=true)
    if m <= 0
        throw(ArgumentError("m needs to be greater than 0, got m = $m"))
    end
    HandyRTransform(rmin, R, k, trim_inf, (-1, 1), (rmin, Inf))
end

function transform(rtf::HandyRTransform, x::TypePoints1D)
    rf_array = rtf.R .* ((1 .+ x) ./ (1 .- x)) .^ rtf.m .+ rtf.rmin
    if rtf.trim_inf
        rf_array = _convert_inf!(rtf, rf_array)
    end
    return rf_array
end

function inverse(rtf::HandyRTransform, r::TypePoints1D)
    tmp_ri = (r .- rtf.rmin) .^ (1 / rtf.m)
    tmp_R = rtf.R .^ (1 / rtf.m)
    return (tmp_ri .- tmp_R) ./ (tmp_ri .+ tmp_R)
end

function deriv(rtf::HandyRTransform, x::TypePoints1D)
    dr = 2 * rtf.m * rtf.R * (1 .+ x) .^ (rtf.m - 1) ./ (1 .- x) .^ (rtf.m + 1)
    if rtf.trim_inf
        dr = _convert_inf!(rtf, dr)
    end
    return dr
end

function deriv2(rtf::HandyRTransform, x::TypePoints1D)
    dr = (
        4
        * rtf.m
        * rtf.R
        .*
        (rtf.m .+ x)
        .*
        (1 .+ x) .^ (rtf.m - 2)
        ./
        (1 .- x) .^ (rtf.m + 2)
    )
    if rtf.trim_inf
        dr = _convert_inf!(rtf, dr)
    end
    return dr
end

function deriv3(rtf::HandyRTransform, x::TypePoints1D)
    dr = (
        4
        * rtf.m
        * rtf.R
        .*
        (1 .+ 6 .* rtf.m .* x .+ 2 .* rtf.m^2 .+ 3 .* x .^ 2)
        .*
        (1 .+ x) .^ (rtf.m - 3)
        ./
        (1 .- x) .^ (rtf.m + 3)
    )
    if rtf.trim_inf
        dr = _convert_inf!(rtf, dr)
    end
    return dr
end


#--------------------------------------------------------------------------------------------------
# HandyModRTransform
#--------------------------------------------------------------------------------------------------
struct HandyModRTransform <: BaseTransform
    rmin::Real
    rmax::Real
    m::Int
    trim_inf::Bool
    domain::Tuple{<:Real,<:Real}
    codomain::Tuple{<:Real,<:Real}
end

function HandyModRTransform(rmin::Real, rmax::Real, m::Int; trim_inf::Bool=true)
    if m <= 0
        throw(ArgumentError("m needs to be greater than 0, got m = $m"))
    end
    if rmax < rmin
        throw(ArgumentError("rmax needs to be greater than rmin. rmax : $rmax, rmin : $rmin."))
    end
    HandyModRTransform(rmin, rmax, m, trim_inf, (-1, 1), (rmin, rmax))
end

function transform(rtf::HandyModRTransform, x::TypePoints1D)
    two_m = 2^rtf.m
    size_r = rtf.rmax - rtf.rmin
    qi = (1 .+ x) .^ rtf.m

    rf_array = qi .* size_r ./ (two_m .* (1 .- two_m .+ size_r) .- qi .* (size_r .- two_m))
    rf_array .+= rtf.rmin

    if rtf.trim_inf
        rf_array[isinf.(rf_array)] .= _convert_inf!(rtf, rf_array)
    end
    return rf_array
end

function inverse(rtf::HandyModRTransform, r::TypePoints1D)
    two_m = 2^rtf.m
    size_r = rtf.rmax - rtf.rmin
    tmp_r = (
        (r .- rtf.rmin) .* (size_r - two_m + 1) ./ ((r .- rtf.rmin) .* (size_r - two_m) .+ size_r)
    )
    return 2 .* (tmp_r) .^ (1 / rtf.m) .- 1
end

function deriv(rtf::HandyModRTransform, x::TypePoints1D)
    two_m = 2^rtf.m
    size_r = rtf.rmax - rtf.rmin
    deriv = (
        -(
            two_m
            * rtf.m
            * (two_m - size_r - 1)
            * size_r * (1 .+ x) .^ (rtf.m - 1)
        )
        ./
        (
            two_m * (1 - two_m + size_r)
            .+
            (two_m - size_r) * (1 .+ x) .^ rtf.m) .^ 2
    )
    if rtf.trim_inf
        deriv[isinf.(deriv)] .= _convert_inf!(rtf, deriv)
    end
    return deriv
end

function deriv2(rtf::HandyModRTransform, x::TypePoints1D)
    two_m = 2^rtf.m
    size_r = rtf.rmax - rtf.rmin
    res = (
        -(
            rtf.m
            * two_m
            * (two_m - size_r - 1)
            * size_r
            * (1 .+ x) .^ (rtf.m - 2)
            .*
            (
                -two_m * (rtf.m - 1) * (two_m - size_r - 1)
                .-
                (rtf.m + 1) * (two_m - size_r) .* (1 .+ x) .^ (rtf.m)
            )
        )
        ./
        (two_m * (1 - two_m + size_r) .+ (two_m - size_r) * (1 .+ x) .^ rtf.m) .^ 3
    )
    return res
end

function deriv3(rtf::HandyModRTransform, x::TypePoints1D)
    two_m = 2^rtf.m
    size_r = rtf.rmax - rtf.rmin
    res = (
        -(
            rtf.m
            * two_m
            * size_r
            * (two_m - size_r - 1)
            * (1 .+ x) .^ (rtf.m - 3)
            .*
            (
                2 * two_m * (rtf.m - 2) * (rtf.m - 1) * (1 - two_m + size_r)^2
                .+
                2^(rtf.m + 2)
                * (rtf.m - 1)
                * (rtf.m + 1)
                * (two_m - 1 - size_r)
                * (two_m - size_r)
                .*
                (1 .+ x) .^ rtf.m
                .+
                (rtf.m + 2)
                * (rtf.m + 1)
                * (two_m - size_r)^2
                .*
                (x .+ 1) .^ (2 * rtf.m)
            )
        )
        ./
        ((two_m * (1 - two_m + size_r) .+ (two_m - size_r) * (1 .+ x) .^ rtf.m)) .^ 4
    )
    return res
end

end # module
