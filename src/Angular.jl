module Angular

using NPZ
using IntegralGrid.BaseGrid

export LEBEDEV_NPOINTS, SPHERICAL_NPOINTS, LEBEDEV_DEGREES, SPHERICAL_DEGREES, LEBEDEV_CACHE, SPHERICAL_CACHE
export AngularGrid
export _load_precomputed_angular_grid, _get_size_and_degree
export convert_angular_sizes_to_degrees
export bisect_left

# Lebedev dictionary for converting number of grid points (keys) to grid's degrees (values)
const LEBEDEV_NPOINTS = Dict(
    6 => 3,
    18 => 5,
    26 => 7,
    38 => 9,
    50 => 11,
    74 => 13,
    86 => 15,
    110 => 17,
    146 => 19,
    170 => 21,
    194 => 23,
    230 => 25,
    266 => 27,
    302 => 29,
    350 => 31,
    434 => 35,
    590 => 41,
    770 => 47,
    974 => 53,
    1202 => 59,
    1454 => 65,
    1730 => 71,
    2030 => 77,
    2354 => 83,
    2702 => 89,
    3074 => 95,
    3470 => 101,
    3890 => 107,
    4334 => 113,
    4802 => 119,
    5294 => 125,
    5810 => 131,
)
const SPHERICAL_NPOINTS = Dict(
    2 => 1,
    6 => 3,
    12 => 5,
    32 => 7,
    48 => 9,
    70 => 11,
    94 => 13,
    120 => 15,
    156 => 17,
    192 => 19,
    234 => 21,
    278 => 23,
    328 => 25,
    380 => 27,
    438 => 29,
    498 => 31,
    564 => 33,
    632 => 35,
    706 => 37,
    782 => 39,
    864 => 41,
    948 => 43,
    1038 => 45,
    1130 => 47,
    1228 => 49,
    1328 => 51,
    1434 => 53,
    1542 => 55,
    1656 => 57,
    1772 => 59,
    1894 => 61,
    2018 => 63,
    2148 => 65,
    2280 => 67,
    2418 => 69,
    2558 => 71,
    2704 => 73,
    2852 => 75,
    3006 => 77,
    3162 => 79,
    3324 => 81,
    3488 => 83,
    3658 => 85,
    3830 => 87,
    4008 => 89,
    4188 => 91,
    4374 => 93,
    4562 => 95,
    4756 => 97,
    4952 => 99,
    5154 => 101,
    5358 => 103,
    5568 => 105,
    5780 => 107,
    5998 => 109,
    6218 => 111,
    6444 => 113,
    6672 => 115,
    6906 => 117,
    7142 => 119,
    7384 => 121,
    7628 => 123,
    7878 => 125,
    8130 => 127,
    8388 => 129,
    8648 => 131,
    8914 => 133,
    9182 => 135,
    9456 => 137,
    9732 => 139,
    10014 => 141,
    10298 => 143,
    10588 => 145,
    10880 => 147,
    11178 => 149,
    11478 => 151,
    11784 => 153,
    12092 => 155,
    12406 => 157,
    12722 => 159,
    13044 => 161,
    13368 => 163,
    13698 => 165,
    14030 => 167,
    14368 => 169,
    14708 => 171,
    15054 => 173,
    15402 => 175,
    15756 => 177,
    16112 => 179,
    16474 => 181,
    16838 => 183,
    17208 => 185,
    17580 => 187,
    17958 => 189,
    18338 => 191,
    18724 => 193,
    19112 => 195,
    19506 => 197,
    19902 => 199,
    20304 => 201,
    20708 => 203,
    21118 => 205,
    21530 => 207,
    21948 => 209,
    22368 => 211,
    22794 => 213,
    23222 => 215,
    23656 => 217,
    24092 => 219,
    24534 => 221,
    24978 => 223,
    25428 => 225,
    25880 => 227,
    26338 => 229,
    26798 => 231,
    27264 => 233,
    27732 => 235,
    28206 => 237,
    28682 => 239,
    29164 => 241,
    29648 => 243,
    30138 => 245,
    30630 => 247,
    31128 => 249,
    31628 => 251,
    32134 => 253,
    32642 => 255,
    33156 => 257,
    33672 => 259,
    34194 => 261,
    34718 => 263,
    35248 => 265,
    35780 => 267,
    36318 => 269,
    36858 => 271,
    37404 => 273,
    37952 => 275,
    38506 => 277,
    39062 => 279,
    39624 => 281,
    40188 => 283,
    40758 => 285,
    41330 => 287,
    41908 => 289,
    42488 => 291,
    43074 => 293,
    43662 => 295,
    44256 => 297,
    44852 => 299,
    45454 => 301,
    46058 => 303,
    46668 => 305,
    47280 => 307,
    47898 => 309,
    48518 => 311,
    49144 => 313,
    49772 => 315,
    50406 => 317,
    51042 => 319,
    51684 => 321,
    52328 => 323,
    52978 => 325,
)

# Lebedev/Spherical dictionary for converting grid's degrees (keys) to numb of grid points (values)
LEBEDEV_DEGREES = Dict(v => k for (k, v) in LEBEDEV_NPOINTS)
SPHERICAL_DEGREES = Dict(v => k for (k, v) in SPHERICAL_NPOINTS)

LEBEDEV_CACHE = Dict()
SPHERICAL_CACHE = Dict()

function bisect_left(arr, value)
    low = 1
    high = length(arr) + 1

    while low < high
        mid = div((low + high), 2)
        if arr[mid] < value
            low = mid + 1
        else
            high = mid
        end
    end

    return low
end


mutable struct AngularGrid <: AbstractGrid
    _grid::Grid
    degree::Union{Int,Nothing}
    use_spherical::Bool
end

function Base.getproperty(grid::AngularGrid, key::Symbol)
    return key in keys(_d_grid) ? _d_grid[key](grid) : getfield(grid, key)
end


function AngularGrid(
    points::Union{Matrix{<:Real},Nothing}=nothing,
    weights::Union{Vector{<:Real},Nothing}=nothing;
    degree::Union{Int,Nothing}=nothing,
    size::Union{Int,Nothing}=nothing,
    cache::Bool=true,
    use_spherical::Bool=false
)
    if !isnothing(points) && !isnothing(weights)
        if !isnothing(degree) || !isnothing(size)
            throw(ArgumentError(
                "degree or size are not used for generating grids " *
                "because points and weights are provided"
            ))
        end
    else
        degree, size = _get_size_and_degree(degree=degree, size=size, use_spherical=use_spherical)
        cache_dict = use_spherical ? SPHERICAL_CACHE : LEBEDEV_CACHE
        if !(degree in keys(cache_dict))
            points, weights = _load_precomputed_angular_grid(degree, size, use_spherical)
            if cache
                cache_dict[degree] = (points, weights)
            end
        else
            points, weights = cache_dict[degree]
        end
        points, weights = points, weights .* 4 * π
    end

    if !use_spherical && any(weights .< 0.0)
        @warn "Lebedev weights are negative which can introduce round-off errors."
    end

    AngularGrid(Grid(points, weights), degree, use_spherical)
end

function convert_angular_sizes_to_degrees(sizes::Vector{<:Int}, use_spherical::Bool)
    degrees = similar(sizes, Int)
    for (idx, _size) in enumerate(unique(sizes))
        deg = _get_size_and_degree(degree=nothing, size=_size, use_spherical=use_spherical)[1]
        degrees[findall(x -> x == _size, sizes)] .= deg
    end
    return degrees
end
convert_angular_sizes_to_degrees(sizes::Vector{<:Int}; use_spherical::Bool=false) = convert_angular_sizes_to_degrees(sizes, use_spherical)

function _get_size_and_degree(; degree::Union{<:Int,Nothing}=nothing, size::Union{<:Int,Nothing}=nothing, use_spherical::Bool=false)
    degrees = use_spherical ? SPHERICAL_DEGREES : LEBEDEV_DEGREES
    npoints = use_spherical ? SPHERICAL_NPOINTS : LEBEDEV_NPOINTS

    if !isnothing(size) && !isinteger(size)
        throw(ArgumentError("size $size should be of type int, not boolean. May be confused with use_spherical."))
    end

    if !isnothing(degree) && !isnothing(size)
        @warn "Both degree and size arguments are given, so only degree is used!"
    end

    if !isnothing(degree)
        ang_degs = sort(collect(keys(degrees)))
        max_degree = maximum(ang_degs)
        if degree < 0 || degree > max_degree
            throw("Argument degree should be a positive integer: $max_degree, got $degree ")
        end
        degree = degree ∈ ang_degs ? degree : ang_degs[bisect_left(ang_degs, degree)]
        return degree, degrees[degree]
    elseif !isnothing(size)
        ang_npts = sort(collect(keys(npoints)))
        max_size = maximum(ang_npts)
        if size < 0 || size > max_size
            throw("Argument size should be a positive integer: $max_size, got $size ")
        end
        size = size ∈ ang_npts ? size : ang_npts[bisect_left(ang_npts, size)]
        return npoints[size], size
    else
        throw("Provide degree and/or size arguments!")
    end
end

function _load_precomputed_angular_grid(degree::Int, grid_size::Int, use_spherical::Bool)
    degrees = use_spherical ? SPHERICAL_DEGREES : LEBEDEV_DEGREES
    npoints = use_spherical ? SPHERICAL_NPOINTS : LEBEDEV_NPOINTS
    type = use_spherical ? "spherical" : "lebedev"
    package_data_path = joinpath(@__DIR__, "..", "src", "data")
    sub_file_path = use_spherical ? "spherical_design" : "lebedev"
    file_path = joinpath(package_data_path, sub_file_path)

    if !(degree in keys(degrees))
        throw("Given degree=$degree is not supported, choose from $degrees")
    end

    if !(grid_size in keys(npoints))
        throw("Given size=$grid_size is not supported, choose from $npoints")
    end

    filename = "$(type)_$(degree)_$(grid_size).npz"
    npz_file = joinpath(file_path, filename)

    data = npzread(npz_file)
    if length(data["weights"]) == 1
        weights = ones(size(data["points"])[1]) * data["weights"][1]
        return data["points"], weights
    else
        return data["points"], data["weights"]
    end
end


end
