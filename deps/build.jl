# using Conda
# 
# # Check if scipy is already installed
# if !("scipy" in Conda._installed_packages())
#     Conda.add("scipy")
# end

#./deps/build.jl
using Pkg;
using Logging;
@info "Initiating build"
ENV["PYTHON"] = ""
Pkg.add("Conda")
Pkg.add("PyCall")
Pkg.build("PyCall")
Pkg.build("Conda")
using PyCall
using Conda
## Add the two packages we need
# Conda.add("gcc=12.1.0"; channel="conda-forge")
# Conda.add("scikit-image")
# Pin this version, to avoid clashes with libgcc.34
Conda.add("scipy")