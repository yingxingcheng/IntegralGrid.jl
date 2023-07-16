using Test
using Random
using IntegralGrid.Becke

function test_becke_sum2_one()
    npoint = 100
    points = rand(npoint, 3) .* 10 .- 5

    nums = [1, 2]
    centers = [1.2 2.3 0.1; -0.4 0.0 -2.2]
    becke = BeckeWeights(Dict(1 => 0.5, 2 => 0.8), 3)

    weights0 = generate_weights(becke, points, centers, nums, [1])
    weights1 = generate_weights(becke, points, centers, nums, [2])

    @test all(isapprox.(weights0 .+ weights1, ones(npoint)))
end

function test_becke_sum3_one()
    npoint = 100
    points = rand(npoint, 3) .* 10 .- 5

    nums = [1, 2, 3]
    centers = [1.2 2.3 0.1; -0.4 0.0 -2.2; 2.2 -1.5 0.0]
    becke = BeckeWeights(Dict(1 => 0.5, 2 => 0.8, 3 => 5.0), 3)

    weights0 = generate_weights(becke, points, centers, nums, [1])
    weights1 = generate_weights(becke, points, centers, nums, [2])
    weights2 = generate_weights(becke, points, centers, nums, [3])

    @test all(isapprox.(weights0 .+ weights1 .+ weights2, ones(npoint)))
end

function test_becke_special_points()
    nums = [1, 2, 3]
    centers = [1.2 2.3 0.1; -0.4 0.0 -2.2; 2.2 -1.5 0.0]
    becke = BeckeWeights(Dict(1 => 0.5, 2 => 0.8, 3 => 5.0), 3)

    weights = generate_weights(becke, centers, centers, nums, 1)
    @test all(isapprox.(weights, [1, 0, 0]))

    weights = generate_weights(becke, centers, centers, nums, 2)
    @test all(isapprox.(weights, [0, 1, 0]))

    weights = generate_weights(becke, centers, centers, nums, 3)
    @test all(isapprox.(weights, [0, 0, 1]))

    weights = generate_weights(becke, centers, centers, nums, nothing, [1, 2, 3, 4])
    @test all(isapprox.(weights, [1, 1, 1]))

    weights = generate_weights(becke, centers, centers, nums, [1, 2], [1, 2, 4])
    @test all(isapprox.(weights, [1, 1, 0]))

    weights = generate_weights(becke, centers, centers, nums, [3, 1], [1, 3, 4])
    @test all(isapprox.(weights, [0, 0, 0]))
end

function test_becke_compute_atom_weight()
    npoint = 10
    points = rand(npoint, 3) .* 10 .- 5

    nums = [1, 2, 3]
    centers = [1.2 2.3 0.1; -0.4 0.0 -2.2; 2.2 -1.5 0.0]
    becke = BeckeWeights(Dict(1 => 0.5, 2 => 0.8, 3 => 5.0), 3)

    weights0 = generate_weights(becke, points, centers, nums, [1])
    weights1 = generate_weights(becke, points, centers, nums, [2])
    weights2 = generate_weights(becke, points, centers, nums, [3])
    aw_0 = compute_atom_weight(becke, points, centers, nums, 1)
    aw_1 = compute_atom_weight(becke, points, centers, nums, 2)
    aw_2 = compute_atom_weight(becke, points, centers, nums, 3)

    @test all(isapprox.(weights0, aw_0))
    @test all(isapprox.(weights1, aw_1))
    @test all(isapprox.(weights2, aw_2))
end

function test_compute_weights()
    npoint = 50
    points = rand(npoint, 3) .* 10 .- 5

    nums = [1, 2, 3]
    centers = [1.2 2.3 0.1; -0.4 0.0 -2.2; 2.2 -1.5 0.0]
    becke = BeckeWeights(Dict(1 => 0.5, 2 => 0.8, 3 => 5.0), 3)
    indices = [1, 14, 30, 51]

    weights_ref = generate_weights(becke, points, centers, nums, nothing, indices)
    weights_compute = compute_weights(becke, points, centers, nums, nothing, indices)

    @test all(isapprox.(weights_ref, weights_compute))

    weights_ref2 = generate_weights(becke, points, centers, nums, [2])
    weights_compute2 = compute_weights(becke, points, centers, nums, [2])

    @test all(isapprox.(weights_ref2, weights_compute2))
end

function test_noble_gas_radius()
    noble_list = [2, 10, 18, 36, 54, 85, 86]
    for i in noble_list
        nums = i != 86 ? [i, i - 1] : [i, i - 2]
        centers = [0.5 0.0 0.0; -0.5 0.0 0.0]
        pts = zeros(10, 3)
        pts[:, 2:end] .+= rand(10, 2)

        becke = BeckeWeights(nothing, 3)
        wts = generate_weights(becke, pts, centers, nums, nothing, [1, 6, 11])
        @test all(isapprox.(wts, 0.5))
        wts = compute_weights(becke, pts, centers, nums, nothing, [1, 6, 11])
        @test all(isapprox.(wts, 0.5))
    end
end

function test_call(npoint)
    points = rand(npoint, 3) .* 10 .- 5
    nums = [1, 2, 3, 2]
    centers = [1.2 2.3 0.1; -0.4 0.0 -2.2; 2.2 -1.5 0.0; 1.5 2.0 0.0]
    becke = BeckeWeights(Dict(1 => 0.5, 2 => 0.8, 3 => 5.0), 3)
    indices = [1, 14, 30, 46, npoint + 1]
    weights_ref = generate_weights(becke, points, centers, nums, nothing, indices)
    weights_call = becke(points, centers, nums, indices)
    @test all(isapprox.(weights_ref, weights_call))
end

@testset "Becke weight class tests" begin
    @testset "Test becke weights add up to one" begin
        test_becke_sum2_one()
    end

    @testset "Test becke weights add up to one with three centers" begin
        test_becke_sum3_one()
    end

    @testset "Test becke weights for special cases" begin
        test_becke_special_points()
    end

    @testset "Test becke compute pa function" begin
        test_becke_compute_atom_weight()
    end

    @testset "Tes compute weights function" begin
        test_compute_weights()
    end

    @testset "Test np.nan value to be handled properly" begin
        test_noble_gas_radius()
    end

    @testset "Tes compute weights function" begin
        test_call(50)
        test_call(200)
    end

end
