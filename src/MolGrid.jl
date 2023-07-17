module MolecularGrid

using NPZ
using IntegralGrid.AtomicGrid
using IntegralGrid.BaseGrid
using IntegralGrid.Becke: BeckeWeights
using IntegralGrid.Hirshfeld: HirshfeldWeights

import IntegralGrid.AtomicGrid.interpolate
import IntegralGrid.AtomicGrid.from_preset
import IntegralGrid.AtomicGrid.from_pruned

export MolGrid, get_atomic_grid, interpolate, from_size

mutable struct MolGrid <: AbstractExtendedGrid
    _grid::Grid
    atgrids::Vector{AtomGrid}
    indices::AbstractVector{<:Real}
    aim_weights::AbstractVector{<:Real}
    atcoords::AbstractMatrix{<:Real}
    atweights::AbstractVector{<:Real}

    function MolGrid(
        atnums::AbstractVector,
        atgrids::Vector{AtomGrid},
        aim_weights::Any;
        store::Bool=false
    )
        natom = length(atgrids)
        _atcoords = zeros(Float64, natom, 3)
        _indices = ones(Int, natom + 1)
        npoint = sum([atomgrid.size for atomgrid in atgrids])
        _points = zeros(Float64, npoint, 3)
        _atweights = zeros(Float64, npoint)
        _atgrids = store ? atgrids : AtomGrid[]

        for (i, atom_grid) in enumerate(atgrids)
            _atcoords[i, :] .= atom_grid.center
            _indices[i+1] = _indices[i] + atom_grid.size
            start, finish = _indices[i], _indices[i+1] - 1
            _points[start:finish, :] .= atom_grid.points  # centers it at the atomic grid.
            _atweights[start:finish] .= atom_grid.weights
        end

        if isa(aim_weights, BeckeWeights) || isa(aim_weights, HirshfeldWeights)
            _aim_weights = aim_weights(_points, _atcoords, atnums, _indices)
        elseif isa(aim_weights, AbstractVector)
            println("aim_weights is provided with an array format.")
            if size(aim_weights, 1) != size(_atweights, 1)
                throw(ArgumentError("aim_weights is not the same size as grid.\naim_weights.size: $(size(aim_weights, 1)), grid.size: $size."))
            end
            _aim_weights = aim_weights
        else
            throw(ArgumentError("Not supported aim_weights type, got $(typeof(aim_weights))."))
        end

        new(Grid(_points, _atweights .* _aim_weights), _atgrids, _indices, _aim_weights, _atcoords, _atweights)
    end
end


function save(grid::MolGrid, filename::AbstractString)
    dict_save = Dict{AbstractString,Any}(
        "points" => grid.points,
        "weights" => grid.weights,
        "atweights" => grid.atweights,
        "atcoords" => grid.atcoords,
        "aim_weights" => grid.aim_weights,
        "indices" => grid.indices
    )
    # Save each attribute of the atomic grid.
    for (i, atomgrid) in enumerate(grid.atgrids)
        dict_save["atgrid_$(i)_points"] = atomgrid.points
        dict_save["atgrid_$(i)_weights"] = atomgrid.weights
        dict_save["atgrid_$(i)_center"] = atomgrid.center
        dict_save["atgrid_$(i)_degrees"] = atomgrid.degrees
        dict_save["atgrid_$(i)_indices"] = atomgrid.indices
        dict_save["atgrid_$(i)_rgrid_pts"] = atomgrid.rgrid.points
        dict_save["atgrid_$(i)_rgrid_weights"] = atomgrid.rgrid.weights
    end
    npzwrite(filename, dict_save)
end

function interpolate(grid::MolGrid, func_vals::AbstractVector{<:Number})
    if isempty(grid.atgrids)
        throw(ArgumentError("Atomic grids need to be stored in molecular grid for this to work. Turn `store` attribute to true."))
    end

    # Multiply f by the nuclear weight function w_n(r) for each atom grid segment.
    func_vals_atom = func_vals .* grid.aim_weights

    # Go through each atomic grid and construct interpolation of f*w_n.
    interpolate_funcs = []
    for i in 1:size(grid.atcoords, 1)
        start_index = grid.indices[i]
        final_index = grid.indices[i+1] - 1
        push!(interpolate_funcs, interpolate(grid[i], func_vals_atom[start_index:final_index]))
    end

    function interpolate_low(
        points::AbstractMatrix{<:Real};
        deriv::Int=0,
        deriv_spherical::Bool=false,
        only_radial_deriv::Bool=false
    )
        output = interpolate_funcs[1](
            points,
            deriv=deriv,
            deriv_spherical=deriv_spherical,
            only_radial_deriv=only_radial_deriv)
        for func in interpolate_funcs[2:end]
            output .+= func(
                points,
                deriv=deriv,
                deriv_spherical=deriv_spherical,
                only_radial_deriv=only_radial_deriv
            )
        end
        return output
    end

    return interpolate_low
end

function from_preset(
    atnums::AbstractVector{<:Int},
    atcoords::AbstractMatrix{<:Real},
    rgrid::Union{OneDGrid,Vector{OneDGrid},Dict{<:Int,OneDGrid}},
    preset::Union{<:AbstractString,Vector{<:AbstractString},Dict{<:Int,<:AbstractString}},
    aim_weights::Any;
    rotate::Int=0,
    store::Bool=false
)
    if length(atnums) != size(atcoords, 1)
        throw(ArgumentError("shape of atomic nums does not match with coordinates\natomic numbers: $(size(atnums)), coordinates: $(size(atcoords))"))
    end

    total_atm = length(atnums)
    atomic_grids = AtomGrid[]
    for i in 1:total_atm
        # get proper radial grid
        if rgrid isa OneDGrid
            rad = rgrid
        elseif rgrid isa Vector
            rad = rgrid[i]
        elseif rgrid isa Dict
            rad = rgrid[atnums[i]]
        else
            throw(ArgumentError("Not supported radial grid input; got input type: $(typeof(rgrid))"))
        end

        # get proper grid type
        if preset isa AbstractString
            gd_type = preset
        elseif preset isa Vector
            gd_type = preset[i]
        elseif preset isa Dict
            gd_type = preset[atnums[i]]
        else
            throw(ArgumentError("Not supported preset type; got preset $preset with type $(typeof(preset))"))
        end

        at_grid = from_preset(
            rad,
            atnum=atnums[i],
            preset=gd_type,
            center=atcoords[i, :],
            rotate=rotate
        )
        push!(atomic_grids, at_grid)
    end

    return MolGrid(atnums, atomic_grids, aim_weights, store=store)
end

function from_size(
    atnums::AbstractVector{<:Int},
    atcoords::AbstractMatrix{<:Real},
    rgrid::OneDGrid,
    angular_size::Int,
    aim_weights::Any;
    rotate::Int=0,
    store::Bool=false
)
    at_grids = AtomGrid[]
    natom = size(atcoords, 1)
    for i in 1:natom
        push!(at_grids, AtomGrid(rgrid, sizes=[angular_size], center=atcoords[i, :], rotate=rotate))
    end
    return MolGrid(atnums, at_grids, aim_weights, store=store)
end

function from_pruned(
    atnums::AbstractVector{<:Real},
    atcoords::AbstractMatrix{<:Real},
    rgrid::Union{OneDGrid,AbstractVector{<:Real}},
    radius::Union{<:Real,AbstractVector{<:Real}},
    aim_weights::Any;
    sectors_r::Vector{<:Vector{<:Real}},
    sectors_degree::Union{Vector{<:Vector{<:Int}},Nothing}=nothing,
    sectors_size::Union{Vector{<:Vector{<:Int}},Nothing}=nothing,
    rotate::Int=0,
    store::Bool=false
)
    at_grids = AtomGrid[]
    num_atoms = size(atcoords, 1)
    # List of None is created, so that indexing is possible in the for-loop.
    sectors_degree = isnothing(sectors_degree) ? [nothing for _ in 1:num_atoms] : sectors_degree
    sectors_size = isnothing(sectors_size) ? [nothing for _ in 1:num_atoms] : sectors_size
    radius_atom = radius isa Real ? fill(radius, num_atoms) : radius
    for i in 1:num_atoms
        # get proper radial grid
        if rgrid isa OneDGrid
            rad = rgrid
        elseif rgrid isa Vector
            rad = rgrid[i]
        else
            throw(ArgumentError("Not supported radial grid input; got input type: $(typeof(rgrid))"))
        end

        at_grid = from_pruned(
            rad,
            radius=radius_atom[i],
            sectors_r=sectors_r[i],
            sectors_degree=sectors_degree[i],
            sectors_size=sectors_size[i],
            center=atcoords[i, :],
            rotate=rotate
        )

        push!(at_grids, at_grid)
    end

    return MolGrid(atnums, at_grids, aim_weights, store=store)
end

function get_atomic_grid(grid::MolGrid, index::Int)
    if index < 0
        throw(ArgumentError("index should be non-negative, got $index"))
    end
    # get atomic grid if stored
    if !isempty(grid.atgrids)
        return grid.atgrids[index]
    end
    # make a local grid
    s_ind = grid.indices[index]
    f_ind = grid.indices[index+1] - 1
    return LocalGrid(grid.points[s_ind:f_ind, :], grid.atweights[s_ind:f_ind], grid.atcoords[index, :])
end

function Base.getindex(grid::MolGrid, index::Int)
    if isempty(grid.atgrids)
        s_ind = grid.indices[index]
        f_ind = grid.indices[index+1] - 1
        subgrid = LocalGrid(grid.points[s_ind:f_ind, :], grid.weights[s_ind:f_ind], grid.atcoords[index, :])
    else
        subgrid = grid.atgrids[index]
    end
    return subgrid
end



end # the end of module