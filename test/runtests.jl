# test_files = [
#     "test_basegrid.jl",
#     "test_atomgrid.jl",
#     "test_utils.jl",
#     "test_onedgrid.jl",
#     "test_angular.jl",
#     "test_rtransform.jl"
# ]
# test_files = ["test_angular.jl"]
test_files = ["test_atomgrid.jl"]

println("Running tests:")
for t in test_files
    include(t)
end
