test_files = ["test_basegrid.jl", "test_atomgrid.jl", "test_utils.jl"]

println("Running tests:")
for t in test_files
    include(t)
end