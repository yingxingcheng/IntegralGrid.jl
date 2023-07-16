using Test
using IntegralGrid.Hirshfeld


function test_call(npoint)
    points = rand(npoint, 3) .* 10 .- 5
    nums = [1, 6, 8]
    centers = [1.2 2.3 0.1; -0.4 0.0 -2.2; 2.2 -1.5 0.0]
    hf = HirshfeldWeights()
    indices = [1, 14, 30, npoint + 1]
    weights_call = hf(points, centers, nums, indices)
    @test ndims(weights_call) == 1
    @test size(weights_call, 1) == npoint
end


@testset "Tes compute weights function" begin
    test_call(50)
    test_call(150)
end