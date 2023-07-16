module Becke

using IntegralGrid.Utils
using PyCall

export BeckeWeights, _calculate_alpha, _switch_func, generate_weights, compute_atom_weight, compute_weights, call


struct BeckeWeights
    radii::Union{Dict{<:Int,<:Real},Nothing}
    order::Int

    function BeckeWeights(radii::Union{Dict{<:Int,<:Real},Nothing}=nothing, order::Int=3)
        data = get_cov_radii(collect(1:86), "bragg")
        _radii = Dict([(i, radius) for (i, radius) in enumerate(data)])

        if !isnothing(radii)
            merge!(_radii, radii)
        end
        new(_radii, order)
    end
end


function _calculate_alpha(radii::AbstractVector{<:Real}, cutoff::Real=0.45)
    # radii_col = @view reshape(radii, :, 1)
    # radii_row = @view reshape(radii, 1, :)
    radii_col = @view radii[:, newaxis]
    radii_row = @view radii[newaxis, :]
    u_ab = (radii_col .- radii_row) ./ (radii_col .+ radii_row)
    alpha = similar(u_ab)
    alpha = u_ab ./ (u_ab .^ 2 .- 1)
    alpha[alpha.>cutoff] .= cutoff
    alpha[alpha.<-cutoff] .= -cutoff
    return alpha
end

function _switch_func(x, order::Int=3)
    for i in 1:order
        x = 1.5 * x - 0.5 * x .^ 3
    end
    return x
end

function generate_weights(
    bw::BeckeWeights,
    points::AbstractMatrix{<:Real},
    atcoords::AbstractMatrix{<:Real},
    atnums::AbstractVector{<:Int},
    select::Union{AbstractVector{<:Int},<:Int,Nothing}=nothing,
    pt_ind::Union{AbstractVector{<:Int},Nothing}=nothing
)
    if isnothing(select)
        select = collect(1:size(atcoords, 1))
    elseif isa(select, Integer)
        select = [select]
    end

    if isnothing(pt_ind)
        pt_ind = []
    elseif length(pt_ind) == 1
        throw(ArgumentError("pt_ind needs to include the ends of each section"))
    end

    sectors = maximum([length(pt_ind) - 1, 1])
    if sectors != length(select)
        throw(ArgumentError("length of select ($(sectors)) does not equal to length of indices ($(length(select)))."))
    end

    numpy = pyimport("numpy")
    weights = zeros(size(points, 1))
    n_p = sqrt.(sum(abs2, atcoords[:, newaxis, :] .- points[newaxis, :, :], dims=3))
    n_n_p = n_p[:, newaxis, :] .- n_p[newaxis, :, :]
    atomic_dist = sqrt.(sum(abs2, atcoords[:, newaxis, :] .- atcoords[newaxis, :, :], dims=3))[:, :, 1]

    tmp = numpy.transpose(n_n_p, [2, 0, 1])
    mu_p_n_n = tmp ./ atomic_dist[newaxis, :, :]
    specified_radius = [bw.radii[num] for num in atnums]
    indices = findall(isnan.(specified_radius))

    if !isempty(indices)
        @warn "Covalent radii for the following atom numbers $(atnums[indices]) is nan. Instead the radii with 1 less the atomic number is used."
    end

    radii = Float64[]
    for num in atnums
        if !isnan(bw.radii[num])
            push!(radii, bw.radii[num])
        else
            if !isnan(bw.radii[num-1])
                push!(radii, bw.radii[num-1])
            elseif !isnan(bw.radii[num-2])
                push!(radii, bw.radii[num-2])
            end
        end
    end

    alpha = _calculate_alpha(radii)
    # mu_p_n_n shape (#points, #uncs, #uncs)
    # alpha shape (#uncs, #uncs)
    v_pp = mu_p_n_n .+ alpha[newaxis, :, :] .* (1 .- mu_p_n_n .^ 2)
    s_ab = 0.5 .* (1 .- _switch_func(v_pp, bw.order))

    s_ab[isnan.(s_ab)] .= 1
    s_ab = prod(s_ab, dims=3)[:, :, 1]

    if sectors == 1
        weights .+= s_ab[:, select[1]] ./ sum(s_ab, dims=2)[:, 1]
    else
        len_s_ab = size(s_ab, 1)
        for i in 1:sectors
            start, finish = pt_ind[i], pt_ind[i+1] - 1
            if start <= len_s_ab && finish > len_s_ab
                finish = len_s_ab
            end
            if start <= finish && finish <= len_s_ab
                sub_s_ab = s_ab[start:finish, :]
                weights[start:finish] .+= sub_s_ab[:, select[i]] ./ sum(sub_s_ab, dims=2)[:, 1]
            end
        end
    end

    return weights
end


function compute_atom_weight(
    bw::BeckeWeights,
    points::AbstractMatrix{<:Real},
    atcoords::AbstractMatrix{<:Real},
    atnums::AbstractVector{<:Int},
    select::Int,
    cutoff::Real=0.45
)
    weights = zeros(size(points, 1))
    n_p = sqrt.(sum(abs2, atcoords[:, newaxis, :] .- points[newaxis, :, :], dims=3))
    n_n_p = n_p[:, newaxis, :] .- n_p[newaxis, :, :]
    atomic_dist = sqrt.(sum(abs2, atcoords[:, newaxis, :] .- atcoords[newaxis, :, :], dims=3))[:, :, 1]

    numpy = pyimport("numpy")
    tmp = numpy.transpose(n_n_p, [2, 0, 1])
    mu_p_n_n = tmp ./ atomic_dist[newaxis, :, :]

    radii = [bw.radii[num] for num in atnums]
    indices = findall(isnan.(radii))

    if !isempty(indices)
        @warn "Covalent radii for the following atom numbers $(atnums[indices]) is nan. Instead the radii with 1 less the atomic number is used."
    end

    radii = Float64[]
    for num in atnums
        if !isnan(bw.radii[num])
            push!(radii, bw.radii[num])
        else
            if !isnan(bw.radii[num-1])
                push!(radii, bw.radii[num-1])
            elseif !isnan(bw.radii[num-2])
                push!(radii, bw.radii[num-2])
            end
        end
    end

    alpha = _calculate_alpha(radii)
    v_pp = mu_p_n_n .+ alpha[newaxis, :, :] .* (1 .- mu_p_n_n .^ 2)
    s_ab = 0.5 .* (1 .- _switch_func(v_pp, bw.order))
    s_ab[isnan.(s_ab)] .= 1
    s_ab = prod(s_ab, dims=3)[:, :, 1]
    weights .+= s_ab[:, select[1]] ./ sum(s_ab, dims=2)[:, 1]
    return weights
end

function compute_weights(
    bw::BeckeWeights,
    points::AbstractMatrix{<:Real},
    atcoords::AbstractMatrix{<:Real},
    atnums::AbstractVector{<:Int},
    select::Union{AbstractVector{<:Int},<:Int,Nothing}=nothing,
    pt_ind::Union{AbstractVector{<:Int},Nothing}=nothing
)
    if isnothing(select)
        select = collect(1:size(atcoords, 1))
    elseif isa(select, Integer)
        select = [select]
    end

    if isnothing(pt_ind)
        pt_ind = []
    elseif length(pt_ind) == 1
        throw(ArgumentError("pt_ind needs to include the ends of each section"))
    end

    sectors = maximum([length(pt_ind) - 1, 1])
    if sectors != length(select)
        throw(ArgumentError("length of select ($(sectors)) does not equal to length of indices ($(length(select)))."))
    end

    weights = zeros(size(points, 1))
    if sectors == 1
        weights .+= compute_atom_weight(bw, points, atcoords, atnums, select[1])
    else
        for i in eachindex(select)
            ind_start = pt_ind[i]
            ind_end = pt_ind[i+1] - 1
            weights[ind_start:ind_end] .+= compute_atom_weight(bw, points[ind_start:ind_end, :], atcoords, atnums, i)
        end
    end
    return weights
end

function (bw::BeckeWeights)(
    points::AbstractMatrix{<:Real},
    atcoords::AbstractMatrix{<:Real},
    atnums::AbstractVector{<:Int},
    indices::AbstractVector{<:Int}
)

    npoint = size(points, 1)
    natom = size(atcoords, 1)
    chunk_size = maximum([1, div(10 * npoint, natom^2)])

    if chunk_size >= npoint
        return generate_weights(bw, points, atcoords, atnums, nothing, indices)
    end

    aim_weights = Float64[]
    chunk_pos = collect(1:chunk_size:npoint+1)
    push!(chunk_pos, npoint + 1)
    for i in 1:length(chunk_pos)-1
        ibegin, iend = chunk_pos[i], chunk_pos[i+1] - 1
        sub_indices = indices .- ibegin .+ 1
        sub_indices = clamp!(sub_indices, 1, Inf)
        weights = generate_weights(bw, points[ibegin:iend, :], atcoords, atnums, nothing, sub_indices)
        append!(aim_weights, weights)
    end
    total_aim_weights = vec(hcat(aim_weights))
    @assert size(total_aim_weights, 1) == npoint
    return total_aim_weights
end


end # end of Becke module