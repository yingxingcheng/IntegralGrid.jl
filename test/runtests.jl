test_files = [
    "test_basegrid.jl",
    "test_atomgrid.jl",
    "test_utils.jl",
    "test_onedgrid.jl",
    "test_angular.jl",
    "test_rtransform.jl",
    "test_atomgrid.jl",
    "test_becke.jl",
    "test_hirshfeld.jl",
    "test_molgrid.jl",
]

println("Running tests:")
for t in test_files
    include(t)
end
