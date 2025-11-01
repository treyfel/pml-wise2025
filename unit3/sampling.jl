"""
    SamplingPlots

A comprehensive module for demonstrating probabilistic graphical models through sampling
and visualization, featuring both the TrueSkill model and the Ising model.

# Overview

This module implements sampling algorithms for two important probabilistic models:
1. **TrueSkill Model**: Bayesian skill estimation for competitive games
2. **Ising Model**: Statistical physics model for magnetic interactions

Both models demonstrate key concepts in probabilistic machine learning including
forward sampling, conditional distributions, and Markov Chain Monte Carlo (MCMC).

# TrueSkill Graphical Model

The TrueSkill model estimates player skills in competitive scenarios through:
- **Hierarchical Bayesian structure**: Skills → Performances → Outcomes
- **Uncertainty quantification**: Models both skill and performance variability  
- **Inference capabilities**: Updates beliefs based on game results

## Model Variables
- **s₁, s₂**: Latent skill levels for players 1 and 2 (continuous, ℝ)
- **p₁, p₂**: Performance values in a specific game (continuous, ℝ)
- **y**: Game outcome (+1 if player 1 wins, -1 if player 2 wins)

## Generative Process
1. Sample skill levels: sᵢ ~ N(μᵢ, σᵢ²)
2. Sample performances: pᵢ ~ N(sᵢ, β²)  
3. Determine outcome: y = sign(p₁ - p₂)

# Ising Model

The Ising model simulates magnetic systems and demonstrates:
- **Spatial correlations**: Local interactions between neighboring spins
- **Phase transitions**: Emergent behavior from simple local rules
- **MCMC sampling**: Gibbs sampling for complex probability distributions

## Model Components
- **Lattice**: L×L grid of binary spins (±1)
- **External field h**: Bias toward +1 or -1 states
- **Interaction strength J**: Coupling between neighboring spins
- **Energy function**: E = -h∑ᵢxᵢ - J∑⟨i,j⟩xᵢxⱼ

# Key Algorithms

- **Forward sampling**: Direct generation from generative models
- **Rejection sampling**: Conditional inference via filtering
- **Gibbs sampling**: MCMC for complex posterior distributions
- **Visualization**: Histogram and heatmap plotting utilities

# Educational Applications

Perfect for teaching:
- Probabilistic graphical models
- Bayesian inference concepts
- MCMC methods
- Model comparison and validation
- Statistical physics connections

# Usage Examples

```julia
# TrueSkill sampling and visualization
using SamplingPlots
samples = SamplingPlots.sample(n=10000, μ1=0.0, μ2=1.0)
SamplingPlots.plot_histogram([[s[1] for s in samples]], xlabel="Player 1 Skill")

# Ising model simulation  
SamplingPlots.gibbs_sample(L=50, h=0.0, J=1.0, n_samples=1)

# Complete demonstration
SamplingPlots.main()
```

# Author
Ralf Herbrich, Hasso-Plattner Institute, 2025

# References
- Herbrich, R., Minka, T., & Graepel, T. (2006). TrueSkill: A Bayesian skill rating system.
- Minka, T., Cleven, R., & Zaykov, Y. (2018). TrueSkill 2: An improved Bayesian skill rating system.
- Ising, E. (1925). Beitrag zur Theorie des Ferromagnetismus. Zeitschrift für Physik.
"""
module SamplingPlots

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

# ============================================================================
# TrueSkill Model - Sampling Functions
# ============================================================================

"""
    sample(; n=100000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0) -> Vector{Vector{Float64}}

Generates samples from the TrueSkill graphical model using forward sampling.

This function implements the generative process of the TrueSkill model, sampling
from the joint distribution over skills, performances, and outcomes.

# Generative Model

For each sample:
1. s₁ ~ N(μ₁, σ₁²)  # Player 1 skill
2. s₂ ~ N(μ₂, σ₂²)  # Player 2 skill
3. p₁ ~ N(s₁, β²)   # Player 1 performance
4. p₂ ~ N(s₂, β²)   # Player 2 performance
5. y = sign(p₁ - p₂) # Outcome (+1 if player 1 wins, -1 if player 2 wins)

# Keywords
- `n::Int=100000`: Number of samples to generate
- `μ1::Float64=0.0`: Prior mean skill for player 1
- `σ1::Float64=1.0`: Prior skill standard deviation for player 1
- `μ2::Float64=0.0`: Prior mean skill for player 2
- `σ2::Float64=1.0`: Prior skill standard deviation for player 2
- `β::Float64=1.0`: Performance noise standard deviation

# Returns
- `Vector{Vector{Float64}}`: Vector of samples, where each sample is [s₁, s₂, p₁, p₂, y]

# Examples
```julia
# Sample from symmetric model (equal prior skills)
samples = sample(n=10000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0)

# Sample from asymmetric model (player 1 weaker than player 2)
samples = sample(n=10000, μ1=-1.5, σ1=1.0, μ2=1.5, σ2=1.0, β=1.0)

# Extract player 1 skills
s1_samples = [s[1] for s in samples]

# Filter for games where player 1 won
wins = filter(s -> s[5] == 1.0, samples)
```

# Model Interpretation

- **Larger |μ₁ - μ₂|**: Greater skill difference between players
- **Larger σᵢ**: More uncertainty about player i's true skill
- **Larger β**: More randomness in game outcomes (luck vs. skill)
- **β → 0**: Outcome determined purely by skill (deterministic)
- **β → ∞**: Outcome approaches random coin flip

# Notes
- Samples are independent and identically distributed (i.i.d.)
- The outcome y is deterministic given performances p₁ and p₂
- Can be used for rejection sampling by filtering on y
- Computational cost is O(n) in the number of samples
"""
function sample(; n = 100000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0)
    # Pre-allocate vector for efficiency
    samples = Vector{Vector{Float64}}(undef, n)
    
    for i in 1:n
        # Sample latent skill levels from prior distributions
        s1 = rand(Normal(μ1, σ1))  # Player 1 skill
        s2 = rand(Normal(μ2, σ2))  # Player 2 skill
        
        # Sample performance values (skill + noise)
        p1 = rand(Normal(s1, β))   # Player 1 performance this game
        p2 = rand(Normal(s2, β))   # Player 2 performance this game
        
        # Determine game outcome (sign of performance difference)
        y = p1 > p2 ? 1.0 : -1.0   # +1: player 1 wins, -1: player 2 wins
        
        # Store complete sample [s₁, s₂, p₁, p₂, y]
        samples[i] = [s1, s2, p1, p2, y]
    end
    
    return samples
end

# ============================================================================
# TrueSkill Model - Visualization Functions
# ============================================================================

"""
    plot_histogram(xss; ylabel="Frequency", xlabel="x", xlim=(-5, 5), bins=100) -> Nothing

Plots overlaid histograms for one or more sets of continuous samples.

Creates normalized probability density histograms, useful for comparing
prior and posterior distributions or marginal distributions under different
conditions.

# Arguments
- `xss`: Vector of vectors, where each inner vector contains samples to plot

# Keywords
- `ylabel::String="Frequency"`: Label for the y-axis
- `xlabel::String="x"`: Label for the x-axis
- `xlim::Tuple{Real,Real}=(-5, 5)`: Range for x-axis
- `bins::Int=100`: Number of histogram bins

# Examples
```julia
# Plot single distribution
samples = sample(n=10000)
s1_values = [s[1] for s in samples]
plot_histogram([s1_values], xlabel=L"s_1", ylabel=L"\\hat{p}(s_1)")

# Compare prior vs. posterior
all_s1 = [s[1] for s in samples]
win_s1 = [s[1] for s in filter(s -> s[5] == 1.0, samples)]
plot_histogram([all_s1, win_s1], xlabel=L"s_1")
```

# Notes
- All histograms are normalized to probability densities (normalize=:pdf)
- Multiple distributions are overlaid with transparency (alpha=0.5)
- Colors cycle through the default palette
- Automatically displays the plot
- Useful for visualizing rejection sampling results
"""
function plot_histogram(xss; ylabel = "Frequency", xlabel = "x", xlim = (-5, 5), bins = 100)
    # Initialize plot with consistent styling
    p = plot(
        legend=false,
        label=false,
        color=:blue,
        xlim=xlim,
        xtickfontsize=18,
        ytickfontsize=18,
        xguidefontsize=20,
        yguidefontsize=20,
        legendfontsize=20,
    )
    
    # Add histogram for each dataset with transparency for overlay
    for xs in xss
        histogram!(xs, label=false, bins=bins, normalize=:pdf, alpha=0.5)
    end
    
    # Add axis labels
    ylabel!(ylabel)
    xlabel!(xlabel)
    
    display(p)
end

"""
    plot_bars(xs; ylabel="Frequency", xlabel="x") -> Nothing

Plots a bar chart showing the empirical probability mass function for discrete outcomes.

Specifically designed for visualizing the distribution of binary outcomes y ∈ {-1, +1}
in the TrueSkill model, showing the empirical win probabilities.

# Arguments
- `xs`: Vector of outcome values (should contain -1.0 and/or +1.0)

# Keywords
- `ylabel::String="Frequency"`: Label for the y-axis (typically probability)
- `xlabel::String="x"`: Label for the x-axis (typically outcome variable)

# Examples
```julia
# Plot outcome distribution
samples = sample(n=10000)
outcomes = [s[5] for s in samples]
plot_bars(outcomes, ylabel=L"\\hat{p}(y)", xlabel=L"y")

# Check for balanced outcomes (should be ~0.5 each for symmetric model)
```

# Returns
- Nothing (displays plot as side effect)

# Notes
- Automatically computes empirical frequencies
- Bars are centered at -1 and +1 with width 0.75
- For symmetric models (μ₁=μ₂, σ₁=σ₂), should show ~50% for each outcome
- For asymmetric models, shows the effect of skill difference on win probability
- Uses semi-transparent bars (alpha=0.5) for aesthetic consistency
"""
function plot_bars(xs; ylabel = "Frequency", xlabel = "x")
    # Compute empirical probabilities for each outcome
    y_minus_1_frac = length(filter(x -> x == -1.0, xs)) / length(xs)
    y_plus_1_frac = length(filter(x -> x == +1.0, xs)) / length(xs)
    
    # Create bar chart
    p = plot(
        bar([-1, 1], [y_minus_1_frac, y_plus_1_frac], alpha=0.5, bar_width = 0.75),
        legend=false,
        label=false,
        color=:blue,
        xtickfontsize=18,
        ytickfontsize=18,
        xguidefontsize=20,
        yguidefontsize=20,
        legendfontsize=20,
    )
    
    # Add axis labels
    ylabel!(ylabel)
    xlabel!(xlabel)
    
    display(p)
end

# ============================================================================
# Ising Model - Data Structures and Core Functions
# ============================================================================

"""
    IsingModel

A mutable struct representing a 2D Ising model on a square lattice with periodic boundary conditions.

The Ising model is a mathematical model of ferromagnetism in statistical mechanics,
where binary spins (±1) interact with their nearest neighbors and an external magnetic field.

# Fields
- `L::Int`: Linear size of the square lattice (total size is L×L)
- `h::Float64`: External magnetic field strength (logit of P(spin = +1))
- `J::Float64`: Nearest-neighbor interaction strength (positive = ferromagnetic)
- `x::Matrix{Int8}`: Current state of the lattice, spins ∈ {-1, +1}

# Energy Function
The Hamiltonian (energy function) is:
```
E = -h∑ᵢ xᵢ - J∑⟨i,j⟩ xᵢxⱼ
```
where the second sum runs over nearest-neighbor pairs ⟨i,j⟩.

# Physical Interpretation
- **h > 0**: External field favors +1 spins (up magnetization)
- **h < 0**: External field favors -1 spins (down magnetization) 
- **h = 0**: No external bias, symmetric model
- **J > 0**: Ferromagnetic coupling (spins prefer to align)
- **J < 0**: Antiferromagnetic coupling (spins prefer to anti-align)
- **J = 0**: No interactions, independent spins

# Phase Behavior
- **Low J/T**: Paramagnetic phase (random spins)
- **High J/T**: Ordered phase (aligned domains)
- **Critical point**: J_c/T ≈ 0.44 for 2D square lattice

# Examples
```julia
# Create random 10×10 Ising model
model = IsingModel(10, h=0.0, J=1.0)

# Ferromagnetic model with bias toward +1
model = IsingModel(20, h=0.5, J=1.0)

# Independent spins (no interactions)
model = IsingModel(50, h=0.0, J=0.0)
```

# See Also
- [`IsingModel(L::Int; h::Float64, J::Float64)`](@ref): Constructor
- [`gibbs_update!(model, i, j)`](@ref): Single-site update
- [`gibbs_sample`](@ref): Complete sampling demonstration
"""
struct IsingModel
    L::Int               # Linear size of the lattice
    h::Float64           # logit of being +1
    J::Float64           # Interaction strength
    x::Matrix{Int8}      # lattice state (+1 or -1)
end

"""
    IsingModel(L::Int; h::Float64=0.0, J::Float64=1.0) -> IsingModel

Constructs an Ising model with random initial configuration.

# Arguments
- `L::Int`: Linear size of the square lattice (creates L×L grid)

# Keywords
- `h::Float64=0.0`: External magnetic field strength
- `J::Float64=1.0`: Nearest-neighbor interaction strength

# Returns
- `IsingModel`: Initialized model with random ±1 spins

# Examples
```julia
# Default ferromagnetic model
model = IsingModel(100)

# Strong external field
model = IsingModel(50, h=2.0, J=1.0)

# Antiferromagnetic interactions  
model = IsingModel(30, h=0.0, J=-0.5)
```

# Implementation Notes
- Initial spins are sampled uniformly from {-1, +1}
- Uses `Int8` for memory efficiency (2 bytes vs 8 bytes per spin)
- Periodic boundary conditions assumed for neighbor calculations
"""
function IsingModel(L::Int; h::Float64=0.0, J::Float64=1.0)
    x = rand([-1, 1], L, L)
    return IsingModel(L, h, J, x)
end

"""
    gibbs_update!(model::IsingModel, i::Int, j::Int) -> IsingModel

Performs a single Gibbs sampling update at lattice site (i,j).

Updates the spin at position (i,j) by sampling from its conditional distribution
given all other spins, implementing the Gibbs sampling algorithm for the Ising model.

# Arguments
- `model::IsingModel`: The Ising model to update (modified in-place)
- `i::Int`: Row index (1 ≤ i ≤ L)  
- `j::Int`: Column index (1 ≤ j ≤ L)

# Returns
- `IsingModel`: The updated model (same object, modified in-place)

# Algorithm
1. Compute local magnetic field: `h_local = h + J × (sum of 4 neighbors)`
2. Calculate probability: `P(x_{i,j} = +1) = σ(2 × h_local)` where σ is sigmoid
3. Sample new spin value from Bernoulli distribution
4. Update `model.x[i,j]` with sampled value

# Conditional Distribution
The conditional probability is derived from the Boltzmann distribution:
```
P(x_{i,j} = +1 | x_{-i,j}) = 1 / (1 + exp(-2 × h_local))
```
where `h_local = h + J × (x_{i+1,j} + x_{i-1,j} + x_{i,j+1} + x_{i,j-1})`

# Boundary Conditions
Uses periodic boundary conditions (toroidal topology):
- Position (L+1, j) wraps to (1, j)
- Position (0, j) wraps to (L, j)
- Similarly for column indices

# Examples
```julia
model = IsingModel(10)
gibbs_update!(model, 5, 5)  # Update center spin
gibbs_update!(model, 1, 1)  # Update corner (wraps to neighbors)

# Full lattice sweep
for i in 1:model.L, j in 1:model.L
    gibbs_update!(model, i, j)
end
```

# Performance Notes
- O(1) time complexity per update
- Modifies model in-place for efficiency
- Uses `mod1` for periodic boundary wrapping
- Vectorized neighbor sum for speed
"""
function gibbs_update!(model::IsingModel, i::Int, j::Int)
    L, h, J = model.L, model.h, model.J
    
    # Calculate the local magnetic field (external + neighbor interactions)
    local_field = h + J * (
        model.x[mod1(i+1, L), j] +      # Right neighbor  
        model.x[mod1(i-1, L), j] +      # Left neighbor
        model.x[i, mod1(j+1, L)] +      # Top neighbor
        model.x[i, mod1(j-1, L)]        # Bottom neighbor
    )

    # Compute conditional probability P(x_{i,j} = +1 | neighbors)
    p_plus = 1 / (1 + exp(-2 * local_field))

    # Sample new spin value
    model.x[i, j] = rand() < p_plus ? 1 : -1

    return model
end

"""
    draw_ising(model::IsingModel; up=:white, down=:black, grid=true) -> Nothing

Visualizes an Ising model configuration as a heatmap with customizable colors.

Creates a publication-quality visualization of the lattice state, useful for
observing phase transitions, domain formation, and correlation structures.

# Arguments
- `model::IsingModel`: The Ising model to visualize

# Keywords
- `up=:white`: Color for +1 spins (can be any Colors.jl color)
- `down=:black`: Color for -1 spins  
- `grid=true`: Whether to show lattice grid lines

# Display Features
- High DPI (200) for crisp images
- Equal aspect ratio (square lattice)
- No axes or labels for clean presentation
- Optional grid overlay for structure visualization
- Automatic y-axis flip (matrix indexing convention)

# Color Schemes
```julia
# Classic black/white (magnetic domains)
draw_ising(model)

# Heat map style  
draw_ising(model, up=:red, down=:blue)

# Subtle gray scale
draw_ising(model, up=:lightgray, down=:darkgray, grid=false)
```

# Physical Interpretation
- **Clustered regions**: Indicate ferromagnetic correlations (J > 0)
- **Checkerboard patterns**: Suggest antiferromagnetic correlations (J < 0)  
- **Random appearance**: Typical of high-temperature or J ≈ 0 regimes
- **Large domains**: Evidence of phase separation or strong coupling

# Examples
```julia
# Visualize random configuration
model = IsingModel(50)
draw_ising(model)

# Show evolution during sampling
for step in 1:100
    gibbs_update!(model, rand(1:50), rand(1:50))
    if step % 20 == 0
        draw_ising(model, grid=false)
        sleep(0.1)  # Animation effect
    end
end
```

# Technical Notes
- Converts ±1 spins to 0/1 for color mapping
- Uses `heatmap` with categorical color gradient
- Matrix transpose for correct orientation
- Grid lines help identify lattice structure
- Framestyle `:none` removes plot borders
"""
function draw_ising(model::IsingModel; up=:white, down=:black, grid=true)
    # Convert {-1,+1} to {0,1} for color mapping
    A = (model.x .+ 1) ./ 2
    pal = cgrad([down, up], 2; categorical=true)

    plt = heatmap(
        A';                                   # Transpose for correct orientation
        c = pal,
        colorbar = false,
        aspect_ratio = :equal,
        xaxis = nothing,
        yaxis = nothing, 
        framestyle = :none,
        dpi = 200,                           # High resolution
        margin = 2Plots.mm,
    )
    yflip!(true)                            # Matrix convention: top-left is (1,1)

    # Optional grid overlay
    if grid
        vline!(0.5:1:(model.L + 0.5), lw=0.5, lc=:gray80)
        hline!(0.5:1:(model.L + 0.5), lw=0.5, lc=:gray80)
    end

    display(plt)
end

"""
    gibbs_sample(; L=200, h=0.0, J=1.0, n_samples=1, skip=100) -> Nothing

Demonstrates Gibbs sampling for the Ising model with visualization of equilibrium configurations.

Performs MCMC sampling using Gibbs updates and displays the resulting lattice
configurations. Useful for exploring phase behavior and sampling convergence.

# Keywords
- `L::Int=200`: Linear lattice size (creates L×L system)
- `h::Float64=0.0`: External magnetic field strength  
- `J::Float64=1.0`: Nearest-neighbor interaction strength
- `n_samples::Int=1`: Number of independent samples to generate
- `skip::Int=100`: Number of full lattice sweeps between samples (burn-in)

# Algorithm
1. Initialize random lattice configuration
2. Perform `skip` full lattice sweeps (burn-in period)
3. Visualize equilibrium configuration  
4. Repeat for `n_samples` independent realizations

# Burn-in Period
The `skip` parameter ensures samples are drawn from equilibrium:
- **Too small**: Samples may be correlated with initial state
- **Too large**: Unnecessary computation
- **Rule of thumb**: 10-1000 sweeps depending on correlation length

# Parameter Regimes
- **J = 0**: Independent spins, random patterns
- **0 < J < 0.5**: Weak correlations, small domains
- **J ≥ 1.0**: Strong ferromagnetic coupling, large domains
- **J < 0**: Antiferromagnetic, checkerboard-like patterns

# Examples
```julia
# Random independent spins
gibbs_sample(L=100, h=0.0, J=0.0, n_samples=3)

# Weak ferromagnetic coupling  
gibbs_sample(L=150, h=0.0, J=0.3, n_samples=2, skip=50)

# Strong coupling with external field
gibbs_sample(L=200, h=1.0, J=1.5, n_samples=1, skip=200)

# Generate multiple realizations
gibbs_sample(L=80, h=0.0, J=1.0, n_samples=5, skip=100)
```

# Computational Complexity
- Time per sweep: O(L²) 
- Total time: O(n_samples × skip × L²)
- Memory: O(L²) for lattice storage

# Convergence Diagnostics
Visual indicators of equilibration:
- **Structured patterns**: Suggest equilibrated ferromagnetic state
- **Persistent randomness**: May indicate insufficient burn-in
- **Correlation length**: Should stabilize after adequate burn-in

# See Also
- [`IsingModel`](@ref): Model structure and physics
- [`gibbs_update!`](@ref): Single-site update algorithm  
- [`draw_ising`](@ref): Visualization options
"""
function gibbs_sample(; L = 200, h = 0.0, J = 1.0, n_samples = 1, skip=100)  
    for sample_idx in 1:n_samples
        # Initialize random configuration
        model = IsingModel(L; h=h, J=J)
        
        # Burn-in: perform full lattice sweeps to reach equilibrium
        for sweep in 1:skip
            for i in 1:L, j in 1:L
                gibbs_update!(model, i, j)
            end
        end
        
        # Visualize equilibrium sample
        draw_ising(model, grid=false)
    end
end

# ============================================================================
# Main Demonstration Function
# ============================================================================

"""
    main() -> Nothing

Comprehensive demonstration of both TrueSkill and Ising model sampling with visualization.

Executes a complete educational workflow covering:

## TrueSkill Model Demonstrations
1. **Symmetric case**: Equal prior skills (μ₁=μ₂=0, σ₁=σ₂=1) 
2. **Asymmetric case**: Skill imbalance (μ₁=-1.5, μ₂=1.5)

For each case:
- Marginal distributions: p(s₁), p(s₂), p(p₁), p(p₂), p(y)
- Conditional inference: p(sᵢ|y=1), p(pᵢ|y=1) via rejection sampling
- Prior vs. posterior overlays showing Bayesian updating

## Ising Model Demonstrations  
1. **Independent spins**: J=0 (random patterns)
2. **Weak coupling**: J=0.5 (small correlated domains)
3. **Strong coupling**: J=1.0 (large ferromagnetic domains)

# Generated Files

All output saved to `~/Downloads/`:

**TrueSkill - Symmetric Model:**
- `s1.svg`, `s2.svg`: Prior skill distributions  
- `p1.svg`, `p2.svg`: Performance distributions
- `y.svg`: Outcome probabilities
- `s1_y1=1.svg`, `s2_y1=1.svg`: Posterior skills given player 1 wins
- `p1_y1=1.svg`, `p2_y1=1.svg`: Posterior performances given player 1 wins

**TrueSkill - Asymmetric Model:**  
- `*_asymmetric.svg`: Same plots with skill imbalance

**Ising Model:**
- `ising_random.png`: J=0 configuration (no correlations)
- `ising_weak.png`: J=0.5 configuration (weak correlations) 
- `ising_strong.png`: J=1.0 configuration (strong correlations)

# Console Output

**TrueSkill Statistics:**
- Sample sizes and model parameters
- Rejection sampling efficiency (fraction of samples retained)
- Expected: ~50% for symmetric, <50% for asymmetric (stronger player wins more)

**Ising Model Visualizations:**
- Real-time display of equilibrium configurations
- Demonstrates phase transitions and correlation emergence

# Key Learning Outcomes

## Probabilistic Concepts
- **Forward sampling**: Generating data from generative models
- **Conditional distributions**: Effect of evidence on beliefs  
- **Rejection sampling**: Simple Bayesian inference method
- **Prior/posterior**: How observations update probability distributions

## Statistical Physics
- **Phase transitions**: Emergence of order from local interactions
- **MCMC sampling**: Gibbs sampling for complex distributions
- **Correlation functions**: Spatial structure in equilibrium states
- **Critical phenomena**: Behavior near phase boundaries

# Usage Examples

```julia
# Run complete demonstration
SamplingPlots.main()

# Custom TrueSkill analysis
samples = SamplingPlots.sample(n=50000, μ1=0.5, μ2=-0.5) 
wins = filter(s -> s[5] == 1.0, samples)
println("Win rate: ", length(wins)/length(samples))

# Custom Ising exploration  
SamplingPlots.gibbs_sample(L=100, h=0.5, J=0.8, n_samples=3)
```

# Computational Requirements
- **Memory**: ~100-500 MB for large sample sets
- **Time**: 10-60 seconds depending on system
- **Dependencies**: Plots.jl, Distributions.jl, LaTeXStrings.jl
- **File access**: Write permissions to ~/Downloads/

# Reproducibility
- Random seed set to deterministic values
- Large sample sizes (1M) ensure statistical accuracy
- Fixed parameters allow consistent comparison across runs

# Educational Applications
Perfect for courses in:
- Probabilistic machine learning
- Bayesian statistics  
- Statistical mechanics
- MCMC methods
- Graphical models
"""
function main()
    # Set random seed for reproducibility across TrueSkill demonstrations
    Random.seed!(2025)

    # ========================================================================
    # TrueSkill Model: Symmetric Case (Equal Prior Skills)
    # ========================================================================
    
    println("Generating samples for symmetric TrueSkill case...")
    data = sample(n = 1000000)
    
    # Plot marginal distributions (no evidence)
    plot_histogram([[x[1] for x in data]], ylabel=L"\hat{p}\left(s_1\right)", xlabel=L"s_1")
    savefig("~/Downloads/s1.svg")
    plot_histogram([[x[2] for x in data]], ylabel=L"\hat{p}\left(s_2\right)", xlabel=L"s_2")
    savefig("~/Downloads/s2.svg")
    plot_histogram([[x[3] for x in data]], ylabel=L"\hat{p}\left(p_1\right)", xlabel=L"p_1")
    savefig("~/Downloads/p1.svg")
    plot_histogram([[x[4] for x in data]], ylabel=L"\hat{p}\left(p_2\right)", xlabel=L"p_2")
    savefig("~/Downloads/p2.svg")
    plot_bars([x[5] for x in data], ylabel=L"\hat{p}\left(y\right)", xlabel=L"y")
    savefig("~/Downloads/y.svg")

    # Plot conditional distributions given player 1 wins (y=1)
    # Shows prior (light) vs. posterior (dark) overlaid
    plot_histogram([[x[1] for x in data], [x[1] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(s_1\right)", xlabel=L"s_1")
    savefig("~/Downloads/s1_y1=1.svg")
    plot_histogram([[x[2] for x in data], [x[2] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(s_2\right)", xlabel=L"s_2")
    savefig("~/Downloads/s2_y1=1.svg")
    plot_histogram([[x[3] for x in data], [x[3] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(p_1\right)", xlabel=L"p_1")
    savefig("~/Downloads/p1_y1=1.svg")
    plot_histogram([[x[4] for x in data], [x[4] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(p_2\right)", xlabel=L"p_2")
    savefig("~/Downloads/p2_y1=1.svg")

    # Report rejection sampling efficiency
    println("Fraction of samples kept after evidence: ", length(filter(x -> x[5] == 1.0, data)) / length(data))

    # ========================================================================
    # TrueSkill Model: Asymmetric Case (Player 1 Weaker)
    # ========================================================================
    
    println("\nGenerating samples for asymmetric TrueSkill case...")
    data = sample(n = 1000000, μ1=-1.5, σ1=1.0, μ2=1.5, σ2=1.0, β=1.0)
    
    # Plot marginal distributions (no evidence)
    plot_histogram([[x[1] for x in data]], ylabel=L"\hat{p}\left(s_1\right)", xlabel=L"s_1")
    savefig("~/Downloads/s1_asymmetric.svg")
    plot_histogram([[x[2] for x in data]], ylabel=L"\hat{p}\left(s_2\right)", xlabel=L"s_2")
    savefig("~/Downloads/s2_asymmetric.svg")
    plot_histogram([[x[3] for x in data]], ylabel=L"\hat{p}\left(p_1\right)", xlabel=L"p_1")
    savefig("~/Downloads/p1_asymmetric.svg")
    plot_histogram([[x[4] for x in data]], ylabel=L"\hat{p}\left(p_2\right)", xlabel=L"p_2")
    savefig("~/Downloads/p2_asymmetric.svg")
    plot_bars([x[5] for x in data], ylabel=L"\hat{p}\left(y\right)", xlabel=L"y")
    savefig("~/Downloads/y_asymmetric.svg")

    # Plot conditional distributions given player 1 wins (surprising outcome)
    # Posterior should shift more dramatically than in symmetric case
    plot_histogram([[x[1] for x in data], [x[1] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(s_1\right)", xlabel=L"s_1")
    savefig("~/Downloads/s1_y1=1_asymmetric.svg")
    plot_histogram([[x[2] for x in data], [x[2] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(s_2\right)", xlabel=L"s_2")
    savefig("~/Downloads/s2_y1=1_asymmetric.svg")
    plot_histogram([[x[3] for x in data], [x[3] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(p_1\right)", xlabel=L"p_1")
    savefig("~/Downloads/p1_y1=1_asymmetric.svg")
    plot_histogram([[x[4] for x in data], [x[4] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(p_2\right)", xlabel=L"p_2")
    savefig("~/Downloads/p2_y1=1_asymmetric.svg")

    # Report rejection sampling efficiency (should be < 50% since player 2 is stronger)
    println("Fraction of samples kept after evidence: ", length(filter(x -> x[5] == 1.0, data)) / length(data))

    # ========================================================================
    # Ising Model: Phase Behavior Demonstration  
    # ========================================================================
    
    println("\nGenerating Ising model samples across different coupling strengths...")
    
    # Set separate seed for Ising model demonstrations
    Random.seed!(42)
    
    # Independent spins (no correlations)
    println("Sampling independent spins (J=0.0)...")
    gibbs_sample(L=200, h=0.0, J=0.0, n_samples=1, skip=100)
    savefig("~/Downloads/ising_random.png")
    
    # Weak ferromagnetic coupling
    println("Sampling weak coupling (J=0.5)...")  
    gibbs_sample(L=200, h=0.0, J=0.5, n_samples=1, skip=100)
    savefig("~/Downloads/ising_weak.png")
    
    # Strong ferromagnetic coupling
    println("Sampling strong coupling (J=1.0)...")
    gibbs_sample(L=200, h=0.0, J=1.0, n_samples=1, skip=100)
    savefig("~/Downloads/ising_strong.png")
    
    println("\nDemonstration complete! Check ~/Downloads/ for output files.")
end

end