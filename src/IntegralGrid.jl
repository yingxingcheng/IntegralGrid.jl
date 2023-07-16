module IntegralGrid

using Conda

# Check if scipy is already installed
if !("scipy" in Conda._installed_packages())
    Conda.add("scipy")
end

# Write your package code here.
include("Utils.jl")
include("BaseGrid.jl")
include("OnedGrid.jl")
include("RTransform.jl")
include("Angular.jl")
include("AtomGrid.jl")
include("Becke.jl")
include("Hirshfeld.jl")
include("MolGrid.jl")
include("Cubic.jl")
include("ODE.jl")
include("PeriodicGrid.jl")
include("Poisson.jl")

end
