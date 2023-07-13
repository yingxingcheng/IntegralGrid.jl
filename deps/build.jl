using Conda

# Check if scipy is already installed
if !("scipy" in Conda.installed())
    Conda.add("scipy")
end
