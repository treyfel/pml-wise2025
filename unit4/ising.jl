"""
    ConditionalIsingPlots

A module for demonstrating conditional Ising models in image processing applications,
showcasing Bayesian inference, image denoising, and marginal probability computation.

# Overview

This module implements a conditional Ising model framework for binary image analysis,
where the model conditions on observed data to perform tasks like:
- **Image denoising**: Removing noise while preserving structure
- **Image inpainting**: Filling missing regions based on context
- **Marginal inference**: Computing pixel-wise probabilities
- **Bayesian image processing**: Principled uncertainty quantification

# Conditional Ising Model

Unlike the standard Ising model, the conditional version incorporates observed data `y`
that influences the local field at each site. This creates a powerful framework for
incorporating prior knowledge and observations into spatial models.

## Mathematical Formulation

The energy function includes both data fidelity and spatial smoothness terms:
```
E(x|y) = -∑ᵢⱼ [h·yᵢⱼ·xᵢⱼ + J·xᵢⱼ·∑neighbors(xₖₗ)]
```

Where:
- **x**: Current state (what we're sampling)
- **y**: Observed/reference data (conditioning variable)
- **h**: Data fidelity strength (how much to trust observations)
- **J**: Spatial coupling strength (smoothness prior)

## Key Applications

### Image Denoising
- **Observation y**: Noisy image
- **Prior**: Spatial smoothness (neighboring pixels should be similar)
- **Posterior**: Clean image balancing data fidelity and smoothness

### Marginal Computation
- **Forward sampling**: Generate multiple realizations
- **Importance sampling**: Weight samples for marginal estimation
- **Uncertainty maps**: Pixel-wise confidence in reconstruction

# Algorithmic Components

1. **Gibbs Sampling**: MCMC for sampling from conditional distributions
2. **Importance Sampling**: Weighted sampling for marginal computation
3. **Energy-Based Models**: Principled probabilistic framework
4. **Visualization**: Heatmaps for states and probability distributions

# Educational Value

Perfect for demonstrating:
- Conditional probability in spatial models
- Bayesian image processing principles
- MCMC sampling in computer vision
- Energy-based probabilistic models
- Importance sampling techniques

# Usage Examples

```julia
using ConditionalIsingPlots

# Load and process test image
img = Matrix{Int8}(map(c -> c.val > 0.5 ? 1 : -1, testimage("cameraman")))

# Create conditional model for denoising
model = ConditionalIsingModel(img, h=1.0, J=2.0)

# Sample denoised version
state = ConditionalIsingModelState(model)
gibbs_sample(state, skip=100)
plot_ising_state(state)

# Compute marginal probabilities
marginals = ConditionalIsingModelMarginals(model, n_iters=1000)
plot_ising_marginals(marginals)
```

# Author
Ralf Herbrich, Hasso-Plattner Institute, 2025

# References
- Geman, S. & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images.
- Besag, J. (1986). On the statistical analysis of dirty pictures.
- Li, S.Z. (2009). Markov Random Field Modeling in Image Analysis.
"""
module ConditionalIsingPlots

using Plots
using Random
using Statistics
using TestImages
using ProgressMeter

# ============================================================================
# Data Structures
# ============================================================================

"""
    ConditionalIsingModel

Represents a conditional Ising model for binary image processing tasks.

This model extends the classical Ising model by incorporating observed data `y`
that influences the energy function. The model is particularly useful for image
processing applications where we want to balance data fidelity with spatial smoothness.

# Fields
- `h::Float64`: Data fidelity strength (coupling to observations)
- `J::Float64`: Spatial interaction strength (smoothness prior)
- `y::Matrix{Int8}`: Observed/reference data matrix with values ∈ {-1, +1}

# Energy Function
The conditional energy is:
```
E(x|y) = -h∑ᵢⱼ yᵢⱼxᵢⱼ - J∑⟨i,j⟩ xᵢⱼxₖₗ
```

Where:
- First sum: Data fidelity term (agreement with observations)
- Second sum: Spatial smoothness term (neighbor agreement)
- ⟨i,j⟩: Denotes nearest neighbor pairs

# Physical Interpretation
- **h > 0**: Encourages agreement with observations y
- **h → ∞**: Perfect reconstruction of observations
- **h = 0**: Ignores observations (standard Ising model)
- **J > 0**: Ferromagnetic coupling (spatial smoothness)
- **J = 0**: No spatial correlations (independent pixels)

# Applications
- **Image denoising**: y = noisy image, sample clean image x
- **Image inpainting**: y = partial image, infer missing regions
- **Image segmentation**: y = features, sample segmentation x

# Examples
```julia
# Denoising model with strong data fidelity
model = ConditionalIsingModel(noisy_img, h=2.0, J=1.0)

# Smoothing model with weak data constraint
model = ConditionalIsingModel(rough_img, h=0.5, J=3.0)

# Inpainting model (missing regions have h=0 locally)
model = ConditionalIsingModel(partial_img, h=1.0, J=2.0)
```

# See Also
- [`ConditionalIsingModelState`](@ref): Sampling state container
- [`ConditionalIsingModelMarginals`](@ref): Marginal probability computation
"""
struct ConditionalIsingModel
    h::Float64           # Data fidelity strength (logit of agreement with observations)
    J::Float64           # Spatial interaction strength 
    y::Matrix{Int8}      # Observed/reference data (+1 or -1) for conditioning
end

"""
    ConditionalIsingModelState

Container for the current sampling state of a conditional Ising model.

Maintains both the model parameters and the current configuration during
MCMC sampling. This mutable container allows efficient in-place updates
during Gibbs sampling iterations.

# Fields
- `model::ConditionalIsingModel`: The underlying conditional model
- `x::Matrix{Int8}`: Current state configuration with values ∈ {-1, +1}

# State Evolution
The state `x` evolves through Gibbs sampling updates, where each site is
updated according to its conditional distribution given all other sites
and the observed data `y`.

# Memory Management
- Initial state is a deep copy of observations to avoid aliasing
- In-place updates minimize memory allocation during sampling
- State can be saved/restored for multiple sampling runs

# Examples
```julia
# Initialize state from model
state = ConditionalIsingModelState(model)

# Perform MCMC sampling
gibbs_sample(state, skip=1000)

# Access current configuration
current_image = state.x

# Visualize current state
plot_ising_state(state)
```

# See Also
- [`gibbs_update!`](@ref): Single-site update function
- [`gibbs_sample`](@ref): Full sampling procedure
"""
struct ConditionalIsingModelState
    model::ConditionalIsingModel
    x::Matrix{Int8}      # Current state (+1 or -1) being sampled
end

"""
    ConditionalIsingModelMarginals

Container for marginal probability estimates computed via importance sampling.

Stores pixel-wise marginal probabilities P(xᵢⱼ = +1) estimated from weighted
samples. These marginals provide uncertainty quantification and can be used
for decision-making in image processing tasks.

# Fields
- `model::ConditionalIsingModel`: The underlying conditional model
- `p::Matrix{Float64}`: Marginal probabilities P(xᵢⱼ = +1) ∈ [0,1]

# Interpretation
Each entry `p[i,j]` represents:
- **p ≈ 1.0**: High confidence that pixel should be +1 (white)
- **p ≈ 0.0**: High confidence that pixel should be -1 (black)
- **p ≈ 0.5**: Maximum uncertainty between states
- **Sharp values**: Strong evidence from data and neighbors
- **Soft values**: Conflicting information or weak coupling

# Applications
- **Uncertainty visualization**: Show confidence in reconstruction
- **Thresholding**: Convert probabilities to binary decisions
- **Risk assessment**: Identify uncertain regions for manual review
- **Active learning**: Query uncertain pixels for additional information

# Computation Method
Uses importance sampling with energy-based weights:
1. Generate samples from proposal distribution
2. Compute importance weights based on energy difference
3. Estimate marginals as weighted average over samples

# Examples
```julia
# Compute marginals with default settings
marginals = ConditionalIsingModelMarginals(model)

# High-precision marginals with more samples
marginals = ConditionalIsingModelMarginals(model, n_iters=5000)

# Visualize uncertainty
plot_ising_marginals(marginals)

# Extract high-confidence pixels
confident_pixels = (marginals.p .> 0.9) .| (marginals.p .< 0.1)
```

# See Also
- [`plot_ising_marginals`](@ref): Visualization function
- Importance sampling theory in Monte Carlo methods
"""
struct ConditionalIsingModelMarginals
    model::ConditionalIsingModel
    p::Matrix{Float64}  # Marginal probabilities P(xᵢⱼ = +1)
end

# ============================================================================
# Constructors
# ============================================================================

"""
    ConditionalIsingModel(img; h::Float64=0.0, J::Float64=1.0) -> ConditionalIsingModel

Convenience constructor for conditional Ising model from image data.

# Arguments
- `img`: Image matrix with values ∈ {-1, +1} to use as observations

# Keywords
- `h::Float64=0.0`: Data fidelity strength
- `J::Float64=1.0`: Spatial interaction strength

# Examples
```julia
# Create model from binary image
model = ConditionalIsingModel(binary_img, h=1.5, J=2.0)
```
"""
function ConditionalIsingModel(img; h::Float64=0.0, J::Float64=1.0)
    return ConditionalIsingModel(h, J, img)
end

"""
    ConditionalIsingModelState(model::ConditionalIsingModel) -> ConditionalIsingModelState

Creates sampling state initialized with a copy of the observed data.

# Arguments
- `model::ConditionalIsingModel`: The conditional model to sample from

# Returns
- `ConditionalIsingModelState`: State initialized with x = copy(y)

# Notes
- Uses `deepcopy` to avoid aliasing with observations
- Initial state matches observations exactly (maximum data fidelity)
- State will evolve through Gibbs sampling to balance data and smoothness

# Examples
```julia
state = ConditionalIsingModelState(model)
@assert state.x == model.y  # Initially matches observations
```
"""
function ConditionalIsingModelState(model::ConditionalIsingModel)
    x = deepcopy(model.y)  # Avoid aliasing with original observations
    return ConditionalIsingModelState(model, x)
end

# ============================================================================
# Core Sampling Functions
# ============================================================================

"""
    gibbs_update!(state::ConditionalIsingModelState, i::Int, j::Int) -> ConditionalIsingModelState

Performs single-site Gibbs sampling update at lattice position (i,j).

Updates the spin at position (i,j) by sampling from its conditional distribution
given all neighboring spins and the observed data. This is the core MCMC update
for conditional Ising model sampling.

# Arguments
- `state::ConditionalIsingModelState`: Model state to update (modified in-place)
- `i::Int`: Row index (1 ≤ i ≤ size(x,1))
- `j::Int`: Column index (1 ≤ j ≤ size(x,2))

# Returns
- `ConditionalIsingModelState`: Updated state (same object)

# Algorithm
1. Compute local field including data term and neighbor interactions
2. Calculate conditional probability P(xᵢⱼ = +1 | x₋ᵢⱼ, y)
3. Sample new value from Bernoulli distribution
4. Update state in-place

# Conditional Distribution
The conditional probability is:
```
P(xᵢⱼ = +1 | x₋ᵢⱼ, y) = σ(2 × local_field)
```

Where:
```
local_field = h × yᵢⱼ + J × ∑neighbors(xₖₗ)
```

And σ(z) = 1/(1 + exp(-z)) is the sigmoid function.

# Boundary Conditions
Uses periodic boundary conditions (toroidal topology):
- Neighbors wrap around at boundaries
- Ensures translation invariance
- Avoids boundary artifacts in sampling

# Examples
```julia
# Update single site
gibbs_update!(state, 10, 15)

# Full lattice sweep
for i in 1:size(state.x, 1), j in 1:size(state.x, 2)
    gibbs_update!(state, i, j)
end
```

# Performance Notes
- O(1) time complexity per update
- In-place modification for memory efficiency
- Vectorized neighbor access for speed
"""
function gibbs_update!(state::ConditionalIsingModelState, i::Int, j::Int)
    (L_x, L_y), h, J = size(state.x), state.model.h, state.model.J
    
    # Calculate local magnetic field: data fidelity + spatial interactions
    local_field = h * state.model.y[i, j] + J * (
        state.x[mod1(i+1, L_x), j] +      # Right neighbor  
        state.x[mod1(i-1, L_x), j] +      # Left neighbor
        state.x[i, mod1(j+1, L_y)] +      # Top neighbor
        state.x[i, mod1(j-1, L_y)]        # Bottom neighbor
    )

    # Compute conditional probability P(xᵢⱼ = +1 | neighbors, observations)
    p_plus = 1 / (1 + exp(-2 * local_field))

    # Sample new spin value from Bernoulli distribution
    state.x[i, j] = rand() < p_plus ? 1 : -1

    return state
end

"""
    gibbs_sample(state; skip=100) -> ConditionalIsingModelState

Performs MCMC burn-in sampling to reach equilibrium distribution.

Executes multiple full lattice sweeps to ensure the sampled state represents
a draw from the equilibrium distribution rather than the initial condition.

# Arguments  
- `state`: Model state to update (modified in-place)

# Keywords
- `skip::Int=100`: Number of full lattice sweeps for burn-in

# Returns
- `ConditionalIsingModelState`: Equilibrated state (same object)

# Burn-in Process
1. Performs `skip` complete sweeps through the lattice
2. Each sweep updates every pixel once in raster order
3. Final state should be approximately from equilibrium distribution

# Convergence Considerations
- **Too few sweeps**: State may retain initial condition bias
- **Too many sweeps**: Unnecessary computation
- **Rule of thumb**: 10-1000 sweeps depending on system size and coupling

# Sweep Order
Uses deterministic raster scan order (row-major):
- Systematic coverage of all sites
- Predictable memory access patterns
- Good mixing for most parameter regimes

# Examples
```julia
# Standard burn-in
gibbs_sample(state, skip=100)

# Quick sampling (may not be equilibrated)
gibbs_sample(state, skip=10)

# High-quality sampling
gibbs_sample(state, skip=500)
```

# Diagnostic Tips
- Visual inspection: Look for structured patterns vs noise
- Multiple runs: Check consistency across independent samples
- Correlation analysis: Measure spatial correlations vs parameters
"""
function gibbs_sample!(state; skip=100)  
    # Burn-in: perform full lattice sweeps to reach equilibrium
    for sweep in 1:skip
        for i in 1:size(state.model.y, 1), j in 1:size(state.model.y, 2)
            gibbs_update!(state, i, j)
        end
    end

    return state
end

# ============================================================================
# Marginal Computation via Importance Sampling
# ============================================================================

"""
    ConditionalIsingModelMarginals(model::ConditionalIsingModel; n_iters=1000) -> ConditionalIsingModelMarginals

Computes marginal probabilities P(xᵢⱼ = +1) using importance sampling.

Estimates pixel-wise marginal probabilities by generating weighted samples
and computing empirical averages. This provides uncertainty quantification
for each pixel in the reconstructed image.

# Arguments
- `model::ConditionalIsingModel`: Model for which to compute marginals

# Keywords  
- `n_iters::Int=1000`: Number of importance samples to generate

# Returns
- `ConditionalIsingModelMarginals`: Container with marginal probabilities

# Algorithm
1. **Proposal sampling**: Generate samples from reference distribution
2. **Importance weighting**: Compute energy-based importance weights
3. **Marginal estimation**: Weighted average over samples
4. **Numerical stability**: Log-space computation to avoid overflow

# Importance Sampling Details
Uses the model itself as the proposal distribution and reweights based on
energy differences. This is particularly effective when the model parameters
provide a good approximation to the target distribution.

# Weight Computation
For each sample x:
```
w ∝ exp(E_proposal(x) - E_target(x))
```

Where energies are computed using the respective model parameters.

# Convergence
- **More samples**: Better marginal estimates but higher computational cost
- **Effective sample size**: Monitor weight concentration
- **Diagnostic**: Check that weights are not dominated by few samples

# Examples
```julia
# Standard marginal computation
marginals = ConditionalIsingModelMarginals(model)

# High-precision estimates
marginals = ConditionalIsingModelMarginals(model, n_iters=5000)

# Extract pixel probabilities
prob_white = marginals.p[10, 15]  # P(x₁₀,₁₅ = +1)
prob_black = 1 - prob_white       # P(x₁₀,₁₅ = -1)
```

# Computational Complexity
- Time: O(n_iters × L² × sweeps_per_sample)
- Memory: O(n_iters × L²) for sample storage
- Parallelizable: Samples can be generated independently

# Numerical Considerations
- Uses log-space arithmetic for numerical stability
- Handles extreme weights gracefully
- Normalizes probabilities to [0,1] range
"""
function ConditionalIsingModelMarginals(model::ConditionalIsingModel; n_iters=1000)
    """
    Computes energy of configuration x under given model parameters.
    
    # Arguments
    - `model::ConditionalIsingModel`: Model defining energy function
    - `x::Matrix{Int8}`: Configuration to evaluate
    
    # Returns
    - `Float64`: Energy value (lower = more probable)
    """
    function energy(model::ConditionalIsingModel, x::Matrix{Int8})
        E = 0.0
        L_x, L_y = size(x)
        
        # Data fidelity term: agreement with observations
        for i in 1:L_x, j in 1:L_y
            E += model.h * model.y[i, j] * x[i, j]
        end
        
        # Spatial interaction term: neighbor agreement
        for i in 1:L_x, j in 1:L_y
            E += model.J * x[i, j] * (
                x[mod1(i+1, L_x), j] +      # Right neighbor  
                x[i, mod1(j+1, L_y)] +      # Top neighbor
                x[mod1(i-1, L_x), j] +      # Left neighbor
                x[i, mod1(j-1, L_y)]        # Bottom neighbor
            )
        end
        
        return E
    end

    # Pre-allocate storage for samples and weights
    log_weights = zeros(Float64, n_iters)
    samples = Vector{Matrix{Int8}}(undef, n_iters)
    
    # Use model as proposal distribution (self-importance sampling)
    proposal_model = ConditionalIsingModel(model.y, h=model.h, J=model.J)

    # Generate importance samples
    @showprogress for i in eachindex(log_weights)        # Generate sample from proposal distribution
        state = ConditionalIsingModelState(proposal_model)
        gibbs_sample!(state, skip=10)  # Equilibrate sample
        
        # Store sample converted to {0,1} for averaging
        samples[i] = (state.x .+ 1) ./ 2
        
        # Compute importance weight (energy difference)
        log_weights[i] = energy(model, state.x) - energy(proposal_model, state.x)
    end

    # Compute normalized importance weights (numerically stable)
    max_logw = maximum(log_weights)
    weights = exp.(log_weights .- max_logw)
    weights ./= sum(weights)  # Normalize to sum to 1

    println("Weights: ", weights)

    # Compute marginal probabilities as weighted averages
    marginals = zeros(Float64, size(model.y))
    for i in eachindex(weights)
        marginals .+= weights[i] .* samples[i]
    end

    return ConditionalIsingModelMarginals(model, marginals)
end

# ============================================================================
# Visualization Functions
# ============================================================================

"""
    plot_ising_state(state::ConditionalIsingModelState; up=:white, down=:black) -> Nothing

Visualizes the current state of a conditional Ising model as a binary heatmap.

Creates a high-quality visualization of the lattice configuration, suitable for
observing spatial patterns, domain formation, and sampling evolution.

# Arguments
- `state::ConditionalIsingModelState`: State to visualize

# Keywords
- `up=:white`: Color for +1 spins (any Colors.jl compatible color)
- `down=:black`: Color for -1 spins

# Display Features
- High resolution (200 DPI) for crisp images
- Equal aspect ratio preserving spatial relationships
- Clean presentation without axes or labels
- Automatic y-axis flip for matrix convention

# Usage Patterns
```julia
# Standard black/white visualization
plot_ising_state(state)

# Custom color scheme
plot_ising_state(state, up=:red, down=:blue)

# Animation loop
for i in 1:100
    gibbs_update!(state, rand(1:L), rand(1:L))
    if i % 10 == 0
        plot_ising_state(state)
        sleep(0.1)
    end
end
```

# See Also
- [`plot_ising_marginals`](@ref): For probability visualizations
- [`gibbs_sample`](@ref): For generating states to visualize
"""
function plot_ising_state(state::ConditionalIsingModelState; up=:white, down=:black)
    # Convert binary {-1,+1} to {0,1} for color mapping
    A = (state.x .+ 1) ./ 2
    pal = cgrad([down, up], 2; categorical=true)

    plt = heatmap(
        A';                                   # Transpose for correct orientation
        c = pal,
        colorbar = false,
        aspect_ratio = :equal,
        xaxis = nothing,
        yaxis = nothing, 
        framestyle = :none,
        dpi = 200,                           # High resolution for crisp output
        margin = 2Plots.mm,
    )
    yflip!(true)                            # Matrix convention: top-left is (1,1)

    display(plt)
end

"""
    plot_ising_marginals(marginals::ConditionalIsingModelMarginals; up=:white, down=:black) -> Nothing

Visualizes marginal probabilities as a continuous grayscale heatmap.

Creates a probability visualization where intensity represents confidence:
bright regions have high P(x = +1), dark regions have high P(x = -1),
and intermediate values show uncertainty.

# Arguments
- `marginals::ConditionalIsingModelMarginals`: Marginal probabilities to visualize

# Keywords
- `up=:white`: Color for probability 1.0 (certain +1)
- `down=:black`: Color for probability 0.0 (certain -1)

# Interpretation
- **Bright pixels**: High confidence for +1 state
- **Dark pixels**: High confidence for -1 state  
- **Gray pixels**: Uncertain regions (probability ≈ 0.5)
- **Sharp boundaries**: Strong evidence for transitions
- **Soft boundaries**: Uncertain transitions

# Applications
- **Uncertainty visualization**: Identify ambiguous regions
- **Quality assessment**: Evaluate reconstruction confidence
- **Decision support**: Focus attention on uncertain areas
- **Method comparison**: Compare uncertainty across algorithms

# Examples
```julia
# Standard uncertainty visualization
plot_ising_marginals(marginals)

# Custom color scheme for uncertainty
plot_ising_marginals(marginals, up=:yellow, down=:purple)

# Save uncertainty map
plot_ising_marginals(marginals)
savefig("uncertainty_map.png")
```

# See Also
- [`ConditionalIsingModelMarginals`](@ref): Marginal computation
- [`plot_ising_state`](@ref): For binary state visualization
"""
function plot_ising_marginals(marginals::ConditionalIsingModelMarginals; up=:white, down=:black)
    # Use continuous color gradient for probability values
    pal = cgrad([down, up], 256; categorical=false)

    plt = heatmap(
        marginals.p';                        # Transpose for correct orientation
        c = pal,
        colorbar = false,                    # Could add colorbar for probability scale
        aspect_ratio = :equal,
        xaxis = nothing,
        yaxis = nothing, 
        framestyle = :none,
        dpi = 200,                           # High resolution
        margin = 2Plots.mm,
    )
    yflip!(true)                            # Matrix convention: top-left is (1,1)

    display(plt)
end

# ============================================================================
# Demonstration Functions
# ============================================================================

"""
    plot_conditional_ising_model_samples(img; h_1=0.5, J_1=0.0, h_2=0.5, J_2=2.0) -> Nothing

Demonstrates image processing pipeline using conditional Ising models.

Shows the progression from original image through noise addition to smoothed
reconstruction, illustrating the balance between data fidelity and spatial priors.

# Arguments
- `img`: Input binary image matrix ∈ {-1, +1}

# Keywords
- `h_1::Float64=0.5`: Data fidelity for noise model
- `J_1::Float64=0.0`: Spatial coupling for noise model (typically 0)
- `h_2::Float64=0.5`: Data fidelity for smoothing model  
- `J_2::Float64=2.0`: Spatial coupling for smoothing model

# Pipeline Stages
1. **Original**: Perfect reconstruction (h=10.0, J=0.0)
2. **Noisy**: Add noise by reducing spatial coupling
3. **Smoothed**: Apply spatial prior for denoising

# Parameter Effects
- **High h, Low J**: Faithful to data, minimal smoothing
- **Low h, High J**: Strong smoothing, less data fidelity
- **Balanced h, J**: Compromise between fidelity and smoothness

# Output Files
- `original_image.png`: Reference image
- `noisy_image.png`: Image with added noise
- `smoothed_image.png`: Denoised result

# Examples
```julia
# Standard denoising demonstration
plot_conditional_ising_model_samples(img)

# Strong denoising
plot_conditional_ising_model_samples(img, h_2=0.3, J_2=3.0)

# Minimal denoising
plot_conditional_ising_model_samples(img, h_2=2.0, J_2=0.5)
```
"""
function plot_conditional_ising_model_samples(img; h_1=0.5, J_1=0.0, h_2=0.5, J_2=2.0)
    # Stage 1: Perfect reconstruction (reference)
    original = gibbs_sample!(ConditionalIsingModelState(ConditionalIsingModel(img, h=10.0, J=0.0)))
    plot_ising_state(original)
    savefig("~/Downloads/original_image.png")

    # Stage 2: Add noise by reducing spatial correlations
    img_with_noise = gibbs_sample!(ConditionalIsingModelState(ConditionalIsingModel(img, h=h_1, J=J_1)))
    plot_ising_state(img_with_noise)
    savefig("~/Downloads/noisy_image.png")

    # Stage 3: Apply spatial smoothing for denoising
    smoothed_img = gibbs_sample!(ConditionalIsingModelState(ConditionalIsingModel(img, h=h_2, J=J_2)))
    plot_ising_state(smoothed_img)
    savefig("~/Downloads/smoothed_image.png")
end

"""
    plot_conditional_ising_model_marginals(img; h=0.5, J=2.0) -> Nothing

Demonstrates marginal probability computation for uncertainty quantification.

Shows the progression from original image through noise addition to marginal
probability estimation, highlighting uncertain regions in the reconstruction.

# Arguments
- `img`: Input binary image matrix ∈ {-1, +1}

# Keywords
- `h::Float64=0.5`: Data fidelity strength for marginal computation
- `J::Float64=2.0`: Spatial coupling strength for marginal computation

# Process
1. **Original**: Reference image for comparison
2. **Noisy**: Create noisy version for reconstruction
3. **Marginals**: Compute pixel-wise uncertainty

# Output Files
- `original_image_marginals.png`: Reference image
- `noisy_image_marginals.png`: Noisy observations
- `marginals_image.png`: Uncertainty heatmap

# Interpretation
The marginal image reveals:
- **Confident regions**: Sharp black/white corresponding to strong evidence
- **Uncertain regions**: Gray areas where model is unsure
- **Boundary effects**: Often uncertain due to conflicting neighbors

# Examples
```julia
# Standard uncertainty analysis
plot_conditional_ising_model_marginals(img)

# High-smoothing regime
plot_conditional_ising_model_marginals(img, h=0.2, J=3.0)

# Low-smoothing regime  
plot_conditional_ising_model_marginals(img, h=1.5, J=0.5)
```
"""
function plot_conditional_ising_model_marginals(img; h=0.5, J=2.0)
    # Stage 1: Reference image
    original = gibbs_sample!(ConditionalIsingModelState(ConditionalIsingModel(img, h=10.0, J=0.0)))
    plot_ising_state(original)
    savefig("~/Downloads/original_image_marginals.png")

    # Stage 2: Create noisy observations
    img_with_noise = gibbs_sample!(ConditionalIsingModelState(ConditionalIsingModel(img, h=h, J=0.0)))
    plot_ising_state(img_with_noise)
    savefig("~/Downloads/noisy_image_marginals.png")

    # Stage 3: Compute and visualize marginal probabilities
    marginals = ConditionalIsingModelMarginals(ConditionalIsingModel(img_with_noise.x, h=h, J=J), n_iters=1000)
    plot_ising_marginals(marginals)
    savefig("~/Downloads/marginals_image.png")
end

# ============================================================================
# Main Demonstration Function
# ============================================================================

"""
    main() -> Nothing

Comprehensive demonstration of conditional Ising models for image processing.

Executes a complete workflow showing:
1. **Image loading**: Convert test image to binary format
2. **Sampling demonstration**: Show denoising pipeline  
3. **Marginal computation**: Uncertainty quantification
4. **File output**: Save results for analysis

# Test Images
Uses TestImages.jl test cases:
- `"cameraman"`: Classic computer vision test image
- `"woman_blonde"`: Portrait with varied textures
- `"lake_gray"`: Natural scene with smooth regions

# Workflow
1. Load and binarize test image (threshold at 0.5)
2. Transpose for correct orientation
3. Run sampling demonstration with balanced parameters
4. Run marginal demonstration with smoothing parameters
5. Save all outputs to ~/Downloads/

# Parameter Selection
- **Sampling demo**: h=1.0, J=1.0 (balanced fidelity/smoothness)
- **Marginal demo**: h=1.0, J=2.0 (emphasize smoothness for uncertainty)

# Output Files
Generated in ~/Downloads/:
- `original_image.png`: Reference image
- `noisy_image.png`: Simulated noisy observations  
- `smoothed_image.png`: Denoised result
- `original_image_marginals.png`: Reference for marginals
- `noisy_image_marginals.png`: Noisy observations
- `marginals_image.png`: Pixel-wise uncertainty map

# Usage
```julia
# Run complete demonstration
ConditionalIsingPlots.main()

# Or run specific parts
using ConditionalIsingPlots
ConditionalIsingPlots.main()
```

# Educational Value
This demonstration illustrates:
- **Bayesian image processing**: Principled uncertainty quantification
- **Model parameters**: Effect of h and J on reconstruction quality
- **MCMC sampling**: Practical application to computer vision
- **Importance sampling**: Advanced inference techniques
"""
function main()
    # Load and preprocess test image
    println("Loading and preprocessing test image...")
    
    # Convert grayscale test image to binary {-1, +1}
    # Threshold at 0.5: bright pixels → +1, dark pixels → -1
    # bw_img = Matrix{Int8}(map(c -> c.val > 0.5 ? 1 : -1, testimage("cameraman")))
    # bw_img = Matrix{Int8}(map(c -> c.val > 0.5 ? 1 : -1, testimage("woman_blonde")))
    bw_img = Matrix{Int8}(map(c -> c.val > 0.5 ? 1 : -1, testimage("lake_gray")))

    println("Running sampling demonstration...")
    # Demonstrate image processing pipeline
    # Parameters chosen to show clear denoising effect
    plot_conditional_ising_model_samples(bw_img'; h_1=1.0, J_1=0.0, h_2=1.0, J_2=1.0)

    println("Computing marginal probabilities...")
    # Demonstrate uncertainty quantification
    # Higher J emphasizes spatial smoothness for clearer uncertainty patterns
    plot_conditional_ising_model_marginals(bw_img'; h=1.0, J=1.0)
    
    println("Demonstration complete! Check ~/Downloads/ for output images.")
    println("Files generated:")
    println("  - original_image.png: Reference image")
    println("  - noisy_image.png: Simulated noisy version")  
    println("  - smoothed_image.png: Denoised result")
    println("  - original_image_marginals.png: Reference for marginals")
    println("  - noisy_image_marginals.png: Noisy observations")
    println("  - marginals_image.png: Pixel-wise uncertainty map")
end

end