using LinearAlgebra
using IntegralGrid.BaseGrid, IntegralGrid.Angular, IntegralGrid.Utils
using Test


function test_consistency()
    for (k, v) in LEBEDEV_NPOINTS
        @test _get_size_and_degree(degree=v) == (v, k)
    end

    for (k, v) in LEBEDEV_DEGREES
        @test _get_size_and_degree(size=v) == (k, v)
    end
end

# test_lebedev_cache method
function test_lebedev_cache()
    degrees = rand(1:100, 50)
    for i in keys(degrees)
        AngularGrid(degree=i, cache=false)
    end
    @test length(LEBEDEV_CACHE) == 0

    for i in keys(degrees)
        AngularGrid(degree=i)
        ref_d = _get_size_and_degree(degree=i)[1]
        @test ref_d in keys(LEBEDEV_CACHE)
    end
end

# test_convert_lebedev_sizes_to_degrees method
function test_convert_lebedev_sizes_to_degrees()
    # first test
    nums = [38, 50, 74, 86, 110, 38, 50, 74]
    degs = convert_angular_sizes_to_degrees(nums)
    ref_degs = [9, 11, 13, 15, 17, 9, 11, 13]
    @test degs == ref_degs

    # second test
    nums = [6]
    degs = convert_angular_sizes_to_degrees(nums)
    ref_degs = [3]
    @test degs == ref_degs
end

@testset "ModuleAngular.jl" begin
    test_consistency()
    test_lebedev_cache()
    test_convert_lebedev_sizes_to_degrees()
    test_integration_of_spherical_harmonic_up_to_degree(5, false)
    test_integration_of_spherical_harmonic_up_to_degree(5, true)
    test_integration_of_spherical_harmonic_up_to_degree(10, false)
    test_integration_of_spherical_harmonic_up_to_degree(10, true)
end
