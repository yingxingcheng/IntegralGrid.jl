
module Hirshfeld


using NPZ
using PyCall
using IntegralGrid.Utils: newaxis
export HirshfeldWeights, _load_npz_proatom, _get_proatom_density, generate_proatom
export call


struct HirshfeldWeights end

function _load_npz_proatom(num::Int)
    # Return radial grid points and neutral density for a given atomic number
    package_data_path = joinpath(@__DIR__, "data")
    fname = joinpath(package_data_path, "proatoms", "a" * lpad(num, 3, "0") * ".npz")
    data = npzread(fname)
    return data["r"], data["dn"]
end

function _get_proatom_density(num::Int, coords_radial::AbstractVector{<:Real})
    # Evaluate density of pro-atom on the given points
    rad, rho = _load_npz_proatom(num)
    scipy = pyimport("scipy.interpolate")
    cspline = scipy.CubicSpline(rad, rho, bc_type="natural", extrapolate=true)
    out = vec(cspline(coords_radial))
    return out
end

function generate_proatom(
    points::AbstractMatrix{<:Real},
    coord::AbstractVector{<:Real},
    num::Int
)
    # Evaluate pro-atom densities on the given grid points
    # dist = norm.(points .- coord[newaxis, 3])
    dist = vec(sqrt.(sum(abs2, points .- coord[newaxis, 3], dims=2)))
    return _get_proatom_density(num, dist)
end

function (hw::HirshfeldWeights)(
    points::AbstractMatrix{<:Real},
    atcoords::AbstractMatrix{<:Real},
    atnums::AbstractVector{<:Int},
    indices::AbstractVector{<:Int}
)
    # Evaluate integration weights on the given grid points
    npoint = size(points, 1)
    aim_weights = zeros(npoint)
    promolecule = zeros(npoint)
    for (index, atnum) in enumerate(atnums)
        proatom = generate_proatom(points, atcoords[index, :], atnum)
        promolecule .+= proatom
        istart, iend = indices[index], indices[index+1] - 1
        aim_weights[istart:iend] .= proatom[istart:iend]
    end
    aim_weights ./= promolecule
    return aim_weights
end


end # the end of module