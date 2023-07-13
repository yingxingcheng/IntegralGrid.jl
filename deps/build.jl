using Conda

# Check if scipy is already installed
if !("scipy" in Conda._installed_packages())
    Conda.add("scipy")
end
