test_files = ["test_basegrid.jl", "test_atomgrid.jl"]

println("Running tests:")
for t in test_files
    include(t)
end