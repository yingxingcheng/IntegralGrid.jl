using Test
using IntegralGrid.RTransform, IntegralGrid.OnedGrid
using LinearAlgebra
using Logging


function check_consistency(rtf::BaseTransform)
    """Check general consistency."""
    # ts = rand(Uniform(0, 99), 200)
    ts = rand(0:99, 200)

    # Consistency between radius and radius_array
    rs = transform(rtf, ts)
    for i in 1:length(ts)
        rf_gd = fill(ts[i], 200)
        @test rs[i] ≈ transform(rtf, rf_gd)[1]
    end

    # Consistency between deriv and deriv_array
    ds = deriv(rtf, ts)
    for i in 1:length(ts)
        rf_gd = fill(ts[i], 200)
        @test ds[i] ≈ deriv(rtf, rf_gd)[1]
    end

    # Consistency between deriv2 and deriv2_array
    d2s = deriv2(rtf, ts)
    for i in 1:length(ts)
        rf_gd = fill(ts[i], 200)
        @test d2s[i] ≈ deriv2(rtf, rf_gd)[1]
    end

    # Consistency between deriv3 and deriv3_array
    d3s = deriv3(rtf, ts)
    for i in 1:length(ts)
        rf_gd = fill(ts[i], 200)
        @test d3s[i] ≈ deriv3(rtf, rf_gd)[1]
    end

    # Consistency between inv and inv_array
    ts .= 0.0
    ts = inverse(rtf, rs)
    for i in 1:length(ts)
        rf_gd = fill(rs[i], 200)
        @test ts[i] ≈ inverse(rtf, rf_gd)[1]
    end

    # Consistency between inv and radius
    for i in 1:length(ts)
        rf_gd = fill(ts[i], 200)
        @test abs(ts[i] - inverse(rtf, transform(rtf, rf_gd))[1]) < 1e-10
    end
end


function check_deriv(rtf::BaseTransform)
    """Check derivative with fd."""
    # ts = rand(Uniform(0, 99), 200)
    ts = rand(0:99, 200)
    eps = 1e-5
    ts0 = ts .- eps / 2
    ts1 = ts .+ eps / 2
    fns = [
        (transform, deriv),
        (deriv, deriv2),
        (deriv2, deriv3)
    ]
    for (fnr, fnd) in fns
        ds = fnd(rtf, ts)
        dns = (fnr(rtf, ts1) .- fnr(rtf, ts0)) ./ eps
        @test maximum(abs.(ds .- dns)) < 1e-5
    end
end

function test_linear_basics()
    """Test linear tf."""
    gd = collect(0:99)
    rtf = LinearInfiniteRTransform(-0.7, 0.8)
    @test abs(transform(rtf, gd)[1] - (-0.7)) < 1e-15
    @test abs(transform(rtf, gd)[end] - 0.8) < 1e-15
    rtf = LinearInfiniteRTransform(-0.7, 0.8)
    gd .= 99
    @test abs(transform(rtf, gd)[1] - 0.8) < 1e-10
    check_consistency(rtf)
    check_deriv(rtf)
    # check_chop(rtf)
    # check_half(rtf)
end

function test_identity_basics()
    # gd = HortonLiner(100)
    rtf = IdentityRTransform()
    @test transform(rtf, 0.0) == 0.0
    @test transform(rtf, 99.0) == 99.0
    check_consistency(rtf)
    check_deriv(rtf)
end

function test_exp_basics()
    rtf = ExpRTransform(0.1, 10.0)
    gd = collect(0:99)
    @test abs(transform(rtf, gd)[1] - 0.1) < 1e-10
    @test abs(transform(rtf, gd)[end] - 10.0) < 1e-10
    gd = ones(100) * 99
    @test abs(transform(rtf, gd)[1] - 10.0) < 1e-10
    check_consistency(rtf)
    check_deriv(rtf)
    # check_chop(rtf)
    # check_half(rtf)
end

function get_power_cases()
    return [(1e-3, 1e2), (1e-3, 1e3), (1e-3, 1e4), (1e-3, 1e5)]
end

function test_power_basics()
    cases = get_power_cases()
    for (rmin, rmax) in cases
        gd = collect(0:99)
        rtf = PowerRTransform(rmin, rmax)
        @test abs(transform(rtf, gd)[end] - rmax) < 1e-9
        @test abs(transform(rtf, gd)[1] - rmin) < 1e-9
        check_consistency(rtf)
        check_deriv(rtf)
    end
end

function test_hyperbolic_basics()
    rtf = HyperbolicRTransform(0.4 / 450, 1.0 / 450)
    check_consistency(rtf)
    check_deriv(rtf)
end

function test_linear_properties()
    rtf = LinearInfiniteRTransform(-0.7, 0.8)
    @test rtf.rmin == -0.7
    @test rtf.rmax == 0.8
end

function test_exp_properties()
    rtf = ExpRTransform(0.1, 1e1)
    @test rtf.rmin == 0.1
    @test rtf.rmax == 1e1
end

function test_power_properties()
    cases = get_power_cases()
    for (rmin, rmax) in cases
        rtf = PowerRTransform(rmin, rmax)
        @test rtf.rmin == rmin
        @test rtf.rmax == rmax
    end
end

function test_hyperbolic_properties()
    rtf = HyperbolicRTransform(0.4 / 450, 1.0 / 450)
    @test rtf.a == 0.4 / 450
    @test rtf.b == 1.0 / 450
end

function test_domain()
    grid = GaussLegendre(10)
    rtfs = [
        IdentityRTransform(),
        LinearInfiniteRTransform(0.1, 1.5),
        ExpRTransform(0.1, 1e1),
        PowerRTransform(1e-3, 1e2),
        HyperbolicRTransform(0.4 / 450, 1.0 / 450),
    ]

    for rtf in rtfs
        @test_throws ArgumentError transform_1d_grid(rtf, grid)
    end
end


function test_linear_bounds()
    @test_throws ArgumentError LinearInfiniteRTransform(1.1, 0.9)
end

function test_exp_bounds()
    @test_throws ArgumentError ExpRTransform(-0.1, 1.0)
    @test_throws ArgumentError ExpRTransform(0.1, -1.0)
    @test_throws ArgumentError ExpRTransform(1.1, 0.9)
end

function test_power_bounds()
    @test_throws ArgumentError PowerRTransform(-1.0, 2.0)
    @test_throws ArgumentError PowerRTransform(0.1, -2.0)
    @test_logs (:warn, "Power need to be larger than 2!") begin
        a = ones(50)
        tf = PowerRTransform(1.0, 1.1)
        transform(tf, a)
    end
    @test_throws ArgumentError PowerRTransform(1.1, 1.0)
end

function test_hyperbolic_bounds()
    @test_throws ArgumentError HyperbolicRTransform(0, 1.0 / 450)
    @test_throws ArgumentError HyperbolicRTransform(-0.1, 1.0 / 450)
    a = ones(450)
    tf = HyperbolicRTransform(0.4, 1.0)
    @test_throws ArgumentError transform(tf, a)
    @test_throws ArgumentError deriv(tf, a)
    @test_throws ArgumentError deriv2(tf, a)
    @test_throws ArgumentError deriv3(tf, a)
    @test_throws ArgumentError inverse(tf, a)
    a = ones(3)
    tf = HyperbolicRTransform(0.4, 0.5)
    @test_throws ArgumentError transform(tf, a)
    @test_throws ArgumentError HyperbolicRTransform(0.2, 0.0)
    @test_throws ArgumentError HyperbolicRTransform(0.2, -1.0)
end


@testset "RTransform.jl" begin
    test_linear_basics()
    test_identity_basics()
    test_exp_basics()
    test_power_basics()
    test_hyperbolic_properties()
    test_domain()
    test_linear_bounds()
    test_exp_bounds()
    test_power_bounds()
    test_hyperbolic_bounds()
end
