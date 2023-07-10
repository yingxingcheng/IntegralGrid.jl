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


function test_integration_of_spherical_harmonic_up_to_degree(degree, use_spherical)
    grid = AngularGrid(degree=degree, use_spherical=use_spherical)
    # Convert to spherical coordinates from Cartesian.
    # r = vec(norm.(grid.points, dims=2))
    r = vec(sqrt.(sum(abs2, grid.points, dims=2)))
    phi = acos.(grid.points[:, 3] ./ r)
    theta = atan.(grid.points[:, 2], grid.points[:, 1])
    # Generate All Spherical Harmonics Up To Degree = 10
    # Returns a three-dimensional array where [order m, degree l, points]
    sph_harm = generate_real_spherical_harmonics(degree, theta, phi)
    for l_deg in 0:degree-1
        for (i, m_ord) in enumerate(-l_deg:l_deg)
            sph_harm_one = sph_harm[l_deg^2+1:(l_deg+1)^2+1, :]
            if l_deg == 0 && m_ord == 0
                actual = sqrt(4.0 * pi)
            else
                actual = 0.0
            end
            @test isapprox(actual, integrate(grid, sph_harm_one[i, :]), atol=1e-7)
        end
    end
end


# Test integration of spherical harmonic of degree higher than grid is not accurate.
function test_integration_of_spherical_harmonic_not_accurate_beyond_degree(use_spherical)
    grid = AngularGrid(degree=3, use_spherical=use_spherical)
    r = vec(sqrt.(sum(abs2, grid.points, dims=2)))
    phi = acos.(grid.points[:, 3] ./ r)
    theta = atan.(grid.points[:, 2], grid.points[:, 1])

    sph_harm = generate_real_spherical_harmonics(l_max=6, theta=theta, phi=phi)
    # Check that l=4,m=0 gives inaccurate results
    @test abs(integrate(grid, sph_harm[4^2+1, :])) > 1e-8
    # Check that l=6,m=0 gives inaccurate results
    @test abs(integrate(grid, sph_harm[6^2+1, :])) > 1e-8
end

# Test orthogonality of spherical harmonic up to degree 3 is accurate.
function test_orthogonality_of_spherical_harmonic_up_to_degree_three(use_spherical)
    degree = 3
    grid = AngularGrid(degree=10, use_spherical=use_spherical)
    # Convert to spherical coordinates from Cartesian.
    r = vec(sqrt.(sum(abs2, grid.points, dims=2)))
    phi = acos.(grid.points[:, 3] ./ r)
    theta = atan.(grid.points[:, 2], grid.points[:, 1])
    # Generate All Spherical Harmonics Up To Degree = 3
    # Returns a three-dimensional array where [order m, degree l, points]
    sph_harm = generate_real_spherical_harmonics(degree, theta, phi)
    for l_deg in 0:3
        for (i, m_ord) in enumerate([0; 1:l_deg; -1:-1:-l_deg])
            for l2 in 0:3
                for (j, m2) in enumerate([0; 1:l2; -1:-1:-l2])
                    sph_harm_one = sph_harm[l_deg^2+1:(l_deg+1)^2, :]
                    sph_harm_two = sph_harm[l2^2+1:(l2+1)^2, :]
                    integral = integrate(grid, sph_harm_one[i, :] .* sph_harm_two[j, :])
                    if l2 != l_deg || m2 != m_ord
                        @test abs(integral) < 1e-8
                    else
                        @test abs(integral - 1.0) < 1e-8
                    end
                end
            end
        end
    end
end

# Test the sum of all points on the sphere is zero.
function test_that_symmetric_spherical_design_is_symmetric()
    for degree in keys(SPHERICAL_DEGREES)
        grid = AngularGrid(degree=degree, use_spherical=true, cache=false)
        @test all(abs.(sum(grid.points, dims=1)) .< 1e-8)
    end
end


@testset "ModuleAngular.jl" begin
    test_consistency()
    test_lebedev_cache()
    test_convert_lebedev_sizes_to_degrees()
    test_integration_of_spherical_harmonic_up_to_degree(5, false)
    test_integration_of_spherical_harmonic_up_to_degree(5, true)
    test_integration_of_spherical_harmonic_up_to_degree(10, false)
    test_integration_of_spherical_harmonic_up_to_degree(10, true)
    test_integration_of_spherical_harmonic_not_accurate_beyond_degree(false)
    test_integration_of_spherical_harmonic_not_accurate_beyond_degree(true)
    test_orthogonality_of_spherical_harmonic_up_to_degree_three(false)
    test_orthogonality_of_spherical_harmonic_up_to_degree_three(true)
    test_that_symmetric_spherical_design_is_symmetric()
end
