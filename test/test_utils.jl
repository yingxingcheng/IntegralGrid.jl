using IntegralGrid.Utils
using Test, Random, LinearAlgebra

function test_get_atomic_radii()
    bragg_slater = [0.25, NaN, 1.45, 1.05, 0.85, 0.7, 0.65, 0.6, 0.5, NaN,
        1.8, 1.5, 1.25, 1.1, 1.0, 1.0, 1.0, NaN, 2.2, 1.8, 1.6, 1.4, 1.35,
        1.4, 1.4, 1.4, 1.35, 1.35, 1.35, 1.35, 1.3, 1.25, 1.15, 1.15, 1.15,
        NaN, 2.35, 2.0, 1.8, 1.55, 1.45, 1.45, 1.35, 1.3, 1.35, 1.4, 1.6,
        1.55, 1.55, 1.45, 1.45, 1.4, 1.4, NaN, 2.6, 2.15, 1.95, 1.85,
        1.85, 1.85, 1.85, 1.85, 1.85, 1.8, 1.75, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75, 1.55, 1.45, 1.35, 1.35, 1.3, 1.35, 1.35, 1.35, 1.5,
        1.9, 1.8, 1.6, 1.9, NaN, NaN]

    cambridge = [0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
        1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.7, 1.6,
        1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19,
        1.20, 1.20, 1.16, 2.20, 1.95, 1.9, 1.75, 1.64, 1.54, 1.47, 1.46,
        1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44,
        2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92,
        1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.7, 1.62, 1.51, 1.44, 1.41,
        1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.5, 1.5]

    alvarez = [0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21,
        1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26,
        1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75, 1.64,
        1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44,
        2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90,
        1.87, 1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48,
        1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69]

    all_index = collect(1:86)
    bragg_bohr = get_cov_radii(all_index, "bragg")
    @test all(isapprox(bragg_bohr[i], bragg_slater[i] * 1.8897261339213, atol=0.001) for i in 1:length(bragg_bohr) if !isnan(bragg_bohr[i]) && !isnan(bragg_slater[i]))

    cambridge_bohr = get_cov_radii(all_index, "cambridge")
    @test isapprox(cambridge_bohr, cambridge[1:end] * 1.8897261339213, atol=1e-7)

    all_index = collect(1:96)
    alvaraz_bohr = get_cov_radii(all_index, "alvarez")
    @test isapprox(alvaraz_bohr, alvarez[1:end] * 1.8897261339213, atol=1e-7)
end


function test_convert_cart_to_sph()
    """
    Test convert_cart_sph accuracy.
    """
    for _ in 1:10
        pts = rand(10, 3)
        center = rand(3)

        # given center
        sph_coor = convert_cart_to_sph(pts, center)
        @test size(sph_coor) == size(pts)

        # Check Z
        z = sph_coor[:, 1] .* cos.(sph_coor[:, 3])
        @test all(isapprox.(z, pts[:, 3] .- center[3]))

        # Check x
        xy = sph_coor[:, 1] .* sin.(sph_coor[:, 3])
        x = xy .* cos.(sph_coor[:, 2])
        @test all(isapprox.(x, pts[:, 1] .- center[1]))

        # Check y
        y = xy .* sin.(sph_coor[:, 2])
        @test all(isapprox.(y, pts[:, 2] .- center[2]))

        # no center
        sph_coor = convert_cart_to_sph(pts)
        @test size(sph_coor) == size(pts)

        # Check Z
        z = sph_coor[:, 1] .* cos.(sph_coor[:, 3])
        @test all(isapprox.(z, pts[:, 3]))

        # Check x
        xy = sph_coor[:, 1] .* sin.(sph_coor[:, 3])
        x = xy .* cos.(sph_coor[:, 2])
        @test all(isapprox.(x, pts[:, 1]))

        # Check y
        y = xy .* sin.(sph_coor[:, 2])
        @test all(isapprox.(y, pts[:, 2]))
    end
end

function test_convert_cart_to_sph_origin()
    """
    Test convert_cart_sph at origin.
    """
    # point at origin
    pts = [0.0 0.0 0.0]
    sph_coor = convert_cart_to_sph(pts)
    @test all(isapprox.(sph_coor, 0.0))

    # point very close to origin
    pts = [1.0e-15 1.0e-15 1.0e-15]
    sph_coor = convert_cart_to_sph(pts)
    @test sph_coor[1, 1] < 1.0e-12
    @test all(sph_coor[1, 2:end] .< 1.0)

    # point very very close to origin
    pts = [1.0e-100 1.0e-100 1.0e-100]
    sph_coor = convert_cart_to_sph(pts)
    @test sph_coor[1, 1] < 1.0e-12
    @test all(sph_coor[1, 2:end] .< 1.0)
end



function test_generate_real_spherical_is_accurate()
    numb_pts = 100
    pts = rand(numb_pts, 3) .* 2 .- 1  # uniform random between -1 and 1
    sph_pts = convert_cart_to_sph(pts)
    r, theta, phi = sph_pts[:, 1], sph_pts[:, 2], sph_pts[:, 3]
    sph_h = generate_real_spherical_harmonics(3, theta, phi)  # l_max = 3
    @test isapprox(sph_h[1, :], ones(length(theta)) / sqrt(4π))
    @test isapprox(sph_h[2, :], sqrt(3 / (4π)) * pts[:, 3] ./ r)
    @test isapprox(sph_h[3, :], sqrt(3 / (4π)) * pts[:, 1] ./ r)
    @test isapprox(sph_h[4, :], sqrt(3 / (4π)) * pts[:, 2] ./ r)
end

# function test_generate_real_spherical_is_orthonormal()
#     atgrid = AngularGrid(7)
#     pts = atgrid.points
#     wts = atgrid.weights
#     r = norm(pts, dims=2)
#     phi = acos.(pts[:, 3] ./ r)
#     theta = atan2.(pts[:, 2], pts[:, 1])
#     sph_h = generate_real_spherical_harmonics(3, theta, phi)  # l_max = 3
#     @test size(sph_h) == (16, 26)
#     for _ in 1:100
#         n1, n2 = rand(0:15, 2)
#         re = sum(sph_h[n1+1, :] .* sph_h[n2+1, :] .* wts)
#         if n1 != n2
#             @test re ≈ 0 atol = 1e-7
#         else
#             @test re ≈ 1 atol = 1e-7
#         end
#     end
#     for i in 0:9
#         sph_h = generate_real_spherical_harmonics(i, theta, phi)
#         @test size(sph_h) == ((i + 1)^2, 26)
#     end
# end
# 
# function test_generate_real_sph_harms_integrates_correctly()
#     angular = AngularGrid(7)
#     pts = angular.points
#     wts = angular.weights
#     r = norm(pts, dims=2)
#     phi = acos.(pts[:, 3] ./ r)
#     theta = atan2.(pts[:, 2], pts[:, 1])
#     lmax = 3
#     sph_h = generate_real_spherical_harmonics(lmax, theta, phi)  # l_max = 3
#     @test size(sph_h) == (1 + 3 + 5 + 7, 26)
#     counter = 1
#     for l_value in 0:lmax
#         for m in [0; collect(1:l_value); collect(-l_value:-1)]
#             re = sum(sph_h[counter, :] .* wts)
#             if l_value == 0
#                 @test re ≈ sqrt(4π) atol = 1e-7
#             else
#                 @test re ≈ 0 atol = 1e-7
#             end
#             @test sum(isnan.(re)) == 0
#             counter += 1
#         end
#     end
# end

function vecnorm(A::AbstractMatrix{T}, dims::Integer=1) where {T}
    return sqrt.(sum(abs2, A, dims=dims))
end

function test_regular_solid_spherical_harmonics()
    """
    Test regular solid spherical harmonics against analytic forms.
    """
    npt = 20
    points = randn(npt, 3)
    r = sqrt.(sum(points .^ 2, dims=2))
    x, y, z = points[:, 1], points[:, 2], points[:, 3]

    lmax = 3
    # Comparison
    sph_pts = convert_cart_to_sph(points)
    result = solid_harmonics(lmax, sph_pts)

    # l = 1, m = 0
    @test result[2, :] ≈ z
    # l = 1, m = 1
    @test result[3, :] ≈ x
    @test result[4, :] ≈ y
    # l = 2, m = 0
    @test result[5, :] ≈ (3 * z .^ 2 - r .^ 2) / 2
    # l = 2, m = 1
    @test result[6, :] ≈ sqrt(3) * z .* x
    @test result[7, :] ≈ sqrt(3) * z .* y
    # l = 2, m = 2
    @test result[8, :] ≈ sqrt(3) / 2 * (x .^ 2 - y .^ 2)
    @test result[9, :] ≈ sqrt(3) * x .* y
    # l = 3, m = 0
    @test result[10, :] ≈ z .* (5 * z .^ 2 - 3 * r .^ 2) / 2
    # l=4, m=4
    @test result[14, :] ≈ sqrt(15) * x .* y .* z
end

function test_generate_orders_horton_order()
    res1 = generate_orders_horton_order(3, "cartesian", 1)
    @test res1 == [0, 1, 2, 3]
    res2 = generate_orders_horton_order(3, "cartesian", 2)
    @test res2 == [3 0; 2 1; 1 2; 0 3]
    res3 = generate_orders_horton_order(3, "cartesian", 3)
    @test res3 == [3 0 0; 2 1 0; 2 0 1; 1 2 0; 1 1 1; 1 0 2; 0 3 0; 0 2 1; 0 1 2; 0 0 3]
    res = generate_orders_horton_order(3, "pure")
    @test res == [3 0; 3 1; 3 -1; 3 2; 3 -2; 3 3; 3 -3]
end



@testset "Utils.jl" begin
    test_get_atomic_radii()
    test_convert_cart_to_sph()
    test_convert_cart_to_sph_origin()
    test_generate_real_spherical_is_accurate()
    test_regular_solid_spherical_harmonics()
    test_generate_orders_horton_order()

end
