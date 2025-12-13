"""
# Evidence Lower BOund (ELBO) Analysis Module

A comprehensive Julia module for computing and visualizing the Evidence Lower BOund (ELBO)
and KL divergence decomposition in variational inference. This module demonstrates the
fundamental trade-off between likelihood maximization and posterior approximation quality
in Bayesian machine learning.

## Theoretical Background

### Variational Inference Decomposition
For a latent variable model with observations V and hidden variables H:
```
log P(V) = ELBO(q) + KL(q(H) || p(H|V))
```

Where:
- **ELBO(q)**: Evidence Lower BOund = E_q[log P(V,H)] - E_q[log q(H)]
- **KL divergence**: Measures how well q(H) approximates the true posterior p(H|V)
- **log P(V)**: Log marginal likelihood (constant for fixed data)

### Model Specification
This module analyzes the hierarchical Gaussian model:
- **Prior**: H ~ N(μ₀, σ₀²) (latent variable)
- **Likelihood**: Vᵢ ~ N(H, β²) for i = 1,...,n (observations)
- **Variational family**: q(H) = N(μ̂, σ̂²) (approximate posterior)

## Key Features

- **ELBO computation**: Exact computation using Gaussian conjugacy
- **KL divergence analysis**: Decomposition of approximation quality
- **Interactive visualization**: Plots showing ELBO/KL trade-offs
- **Parameter sensitivity**: Analysis across different parameter settings
- **Educational demonstrations**: Clear illustration of variational principles

## Applications

- **Variational inference education**: Understanding the ELBO objective
- **Algorithm analysis**: Comparing different variational approximations
- **Hyperparameter tuning**: Visualizing sensitivity to approximation parameters
- **Method development**: Prototyping new variational algorithms

## Usage Examples

```julia
using .ELBO

# Basic ELBO analysis
Random.seed!(42)
ELBO.plot_elbo_and_kl_divergence(σ_hat=1.0, β=1.0, n=10)

# Compare different approximation qualities
ELBO.plot_elbo_and_kl_divergence(σ_hat=0.1, β=1.0, n=10)  # Tight approximation
ELBO.plot_elbo_and_kl_divergence(σ_hat=2.0, β=1.0, n=10)  # Loose approximation

# Analyze effect of observation noise
ELBO.plot_elbo_and_kl_divergence(σ_hat=1.0, β=0.5, n=10)  # Low noise
ELBO.plot_elbo_and_kl_divergence(σ_hat=1.0, β=2.0, n=10)  # High noise
```

---
© 2025 by Ralf Herbrich  
Hasso-Plattner Institute
"""
module ELBO

# === IMPORTS ===
include("../unit1/gaussian.jl")

using Plots
using Distributions
using LaTeXStrings
using Random

using .Gaussian

# === EXPORTS ===
export compute_elbo_and_kl_divergence, plot_elbo_and_kl_divergence, 
       analyze_approximation_quality, compare_approximations, main

# === CORE COMPUTATIONAL FUNCTIONS ===

"""
    compute_elbo_and_kl_divergence(q::Gaussian1D, vs; β=1, prior::Gaussian1D=Gaussian1D(0,1)) -> (Float64, Float64)

Compute the Evidence Lower BOund (ELBO) and KL divergence for a hierarchical Gaussian model.

This function performs exact computation of the ELBO and KL divergence for the model:
- **Prior**: H ~ N(μ₀, σ₀²)
- **Likelihood**: Vᵢ ~ N(H, β²) for observations vs
- **Variational posterior**: q(H) = N(μ̂, σ̂²)

# Arguments
- `q::Gaussian1D`: Variational posterior approximation q(H)
- `vs::AbstractVector{<:Real}`: Observed data points V₁, V₂, ..., Vₙ
- `β::Real=1`: Observation noise standard deviation (must be positive)
- `prior::Gaussian1D=Gaussian1D(0,1)`: Prior distribution p(H)

# Returns
- `Tuple{Float64, Float64}`: (ELBO value, KL divergence)

# Mathematical Details

## ELBO Computation
The ELBO decomposes as:
```
ELBO(q) = E_q[log p(V,H)] - E_q[log q(H)]
        = E_q[log p(H)] + E_q[log p(V|H)] - E_q[log q(H)]
        = -KL(q(H)||p(H)) + Σᵢ E_q[log p(Vᵢ|H)]
```

## Exact Gaussian Computation
Using Gaussian conjugacy, the true posterior is:
```
p(H|V) = N(μ_post, σ²_post)
where:
μ_post = (μ₀/σ₀² + Σᵢ vᵢ/β²) / (1/σ₀² + n/β²)
σ²_post = 1 / (1/σ₀² + n/β²)
```

The KL divergence and ELBO can then be computed analytically.

# Algorithm Steps
1. **Construct joint distribution**: Combine prior and likelihood terms
2. **Compute log marginal likelihood**: log P(V) using normalization
3. **Compute KL divergence**: KL(q(H) || p(H|V))
4. **Extract ELBO**: ELBO = log P(V) - KL

# Examples
```julia
# Generate synthetic data
Random.seed!(42)
h_true = 2.0
vs = rand(Normal(h_true, 1.0), 10)

# Test different approximations
q_good = Gaussian1DFromMeanVariance(2.0, 0.1)  # Close to truth
q_bad = Gaussian1DFromMeanVariance(0.0, 5.0)   # Far from truth

elbo_good, kl_good = compute_elbo_and_kl_divergence(q_good, vs)
elbo_bad, kl_bad = compute_elbo_and_kl_divergence(q_bad, vs)

println("Good approximation - ELBO: \$elbo_good, KL: \$kl_good")
println("Bad approximation - ELBO: \$elbo_bad, KL: \$kl_bad")
```

# Performance Notes
- **Complexity**: O(n) in number of observations
- **Memory**: O(1) additional storage (uses streaming computation)
- **Numerical stability**: Uses precision parameterization internally

# See Also
- [`plot_elbo_and_kl_divergence`](@ref): Visualization of ELBO landscape
- [`analyze_approximation_quality`](@ref): Detailed approximation analysis
- Variational inference literature for theoretical background
"""
function compute_elbo_and_kl_divergence(
    q::Gaussian.Gaussian1D, 
    vs::AbstractVector{<:Real}; 
    β::Real = 1, 
    prior::Gaussian.Gaussian1D = Gaussian.Gaussian1D(0, 1)
)
    # Validate inputs
    β > 0 || throw(ArgumentError("Observation noise β must be positive, got β=$β"))
    !isempty(vs) || throw(ArgumentError("Observations vector cannot be empty"))
    
    # Construct true posterior by combining prior and likelihood
    p = Gaussian.NonNormalizedGaussian1D(prior)
    for v in vs
        p *= Gaussian.Gaussian1DFromMeanVariance(v, β^2)
    end
    
    # Extract results
    log_P_V = p.log_norm
    kl_divergence = Gaussian.KL_divergence(Gaussian.NonNormalizedGaussian1D(q), p)
    elbo = log_P_V - kl_divergence
    
    return elbo, kl_divergence
end

# === VISUALIZATION FUNCTIONS ===

"""
    plot_elbo_and_kl_divergence(; β=1, σ_hat=1, h_true=1, prior=Gaussian1D(0,1), n=10, title="ELBO and KL Divergence Analysis") -> Nothing

Plot the ELBO and KL divergence as functions of the variational mean μ̂.

This function creates an interactive visualization showing how the ELBO and KL divergence
change as the variational mean parameter μ̂ varies around the true latent value. The plot
demonstrates the fundamental trade-off in variational inference between likelihood 
maximization and posterior approximation quality.

# Arguments
- `β::Real=1`: Observation noise standard deviation (must be positive)
- `σ_hat::Real=1`: Variational posterior standard deviation (must be positive)  
- `h_true::Real=1`: True latent variable value used to generate synthetic data
- `prior::Gaussian1D=Gaussian1D(0,1)`: Prior distribution p(H)
- `n::Int=10`: Number of synthetic observations to generate (must be positive)
- `title::String="ELBO and KL Divergence Analysis"`: Plot title
- `save_path::Union{String,Nothing}=nothing`: Optional path to save the plot

# Returns
- `Nothing`: Displays the plot directly

# Plot Elements
The visualization shows three curves:
- **ELBO (blue)**: Evidence Lower BOund as a function of μ̂
- **KL Divergence (red)**: KL(q(H) || p(H|V)) as a function of μ̂
- **Log Marginal Likelihood (black dashed)**: log P(V) - constant across μ̂

# Mathematical Interpretation
- **ELBO maximum**: Optimal variational approximation for given σ̂
- **KL minimum**: Best posterior approximation for given σ̂
- **Constant sum**: ELBO + KL = log P(V) demonstrates the decomposition

# Visual Features
- Professional styling with LaTeX labels
- Clear legend and appropriate colors
- Reasonable μ̂ range centered around true value
- High resolution for publication quality

# Examples
```julia
# Basic usage with default parameters
plot_elbo_and_kl_divergence()

# High precision approximation
plot_elbo_and_kl_divergence(σ_hat=0.1, title="Tight Variational Approximation")

# Low precision approximation  
plot_elbo_and_kl_divergence(σ_hat=3.0, title="Loose Variational Approximation")

# High noise observations
plot_elbo_and_kl_divergence(β=2.0, title="High Noise Regime")

# Many observations
plot_elbo_and_kl_divergence(n=100, title="Large Sample Analysis")

# Save to file
plot_elbo_and_kl_divergence(save_path="elbo_analysis.png")
```

# Performance Notes
- **Resolution**: Uses 300 points for smooth curves
- **Memory**: O(n) for data generation, O(1) for each ELBO computation
- **Display**: Automatically displays the plot; can be saved if path provided

# Educational Value
This visualization is particularly useful for:
- Understanding the ELBO objective in variational inference
- Demonstrating the effect of approximation quality (σ̂)
- Showing the impact of observation noise (β)
- Illustrating the data dependency (n)

# See Also
- [`compute_elbo_and_kl_divergence`](@ref): Core computation function
- [`analyze_approximation_quality`](@ref): Quantitative analysis
- [`compare_approximations`](@ref): Side-by-side comparison
"""
function plot_elbo_and_kl_divergence(; 
    β::Real = 1, 
    σ_hat::Real = 1, 
    h_true::Real = 1, 
    prior::Gaussian.Gaussian1D = Gaussian.Gaussian1D(0, 1), 
    n::Int = 10,
    title::String = "ELBO and KL Divergence Analysis",
    save_path::Union{String,Nothing} = nothing
)
    # Input validation
    β > 0 || throw(ArgumentError("Observation noise β must be positive, got β=$β"))
    σ_hat > 0 || throw(ArgumentError("Variational std σ_hat must be positive, got σ_hat=$σ_hat"))
    n > 0 || throw(ArgumentError("Number of observations n must be positive, got n=$n"))
    
    # Generate synthetic observations
    vs = rand(Normal(h_true, β), n)
    
    # Create range of variational means centered around true value
    μ_hat_range = range(h_true - 3 * β, h_true + 3 * β, length=300)
    
    # Compute ELBO and KL for each μ̂ value
    results = map(μ_hat_range) do μ_hat
        q = Gaussian.Gaussian1DFromMeanVariance(μ_hat, σ_hat^2)
        compute_elbo_and_kl_divergence(q, vs; β=β, prior=prior)
    end
    
    # Extract ELBO and KL divergence values
    elbo_values = [r[1] for r in results]
    kl_values = [r[2] for r in results]
    log_marginal_likelihood = elbo_values .+ kl_values  # Should be constant
    
    # Create the plot with professional styling
    p = plot(
        μ_hat_range, 
        elbo_values,
        linewidth = 3,
        color = :green,
        # label = "ELBO",
        label = false,
        xlabel = L"\hat{\mu}",
        ylabel = "Value",
        title = title,
        xtickfontsize = 12,
        ytickfontsize = 12,
        xguidefontsize = 14,
        yguidefontsize = 14,
        legendfontsize = 12,
        titlefontsize = 16,
        grid = true,
        gridwidth = 1,
        gridcolor = :lightgray,
        size = (800, 600),
        dpi = 300
    )
    
    # Add KL divergence curve
    plot!(p,
        μ_hat_range, 
        kl_values,
        linewidth = 3,
        # label = "KL Divergence",
        label = false,
        color = :blue
    )
    
    # Add log marginal likelihood (should be constant)
    plot!(p,
        μ_hat_range, 
        log_marginal_likelihood,
        linewidth = 3,
        # label = L"\log P(V)",
        label = false,
        color = :red,
        # linestyle = :dash
    )
    
    # Add vertical line at true value for reference
    # vline!(p, [h_true], 
    #     color = :green, 
    #     linestyle = :dot, 
    #     linewidth = 2, 
    #     label = L"H_{true}",
    #     alpha = 0.7
    # )
    
    # Save if requested
    if save_path !== nothing
        savefig(p, save_path)
        @info "Plot saved to $save_path"
    end
    
    # Display the plot
    display(p)
    
    return nothing
end

# === ANALYSIS FUNCTIONS ===

"""
    analyze_approximation_quality(q::Gaussian1D, vs; β=1, prior::Gaussian1D=Gaussian1D(0,1)) -> NamedTuple

Analyze the quality of a variational approximation with detailed diagnostics.

This function provides comprehensive analysis of how well a variational approximation
q(H) captures the true posterior p(H|V), including decomposition of the ELBO and
comparison with the optimal posterior.

# Arguments
- `q::Gaussian1D`: Variational posterior approximation q(H)
- `vs::AbstractVector{<:Real}`: Observed data points V₁, V₂, ..., Vₙ
- `β::Real=1`: Observation noise standard deviation (must be positive)
- `prior::Gaussian1D=Gaussian1D(0,1)`: Prior distribution p(H)

# Returns
- `NamedTuple`: Comprehensive analysis results containing:
  - `elbo`: Evidence Lower BOund value
  - `kl_divergence`: KL divergence from true posterior
  - `log_marginal_likelihood`: Log probability of observations
  - `optimal_posterior`: True posterior p(H|V)
  - `approximation_gap`: Distance from optimal approximation
  - `effective_sample_size`: Estimated effective sample size
  - `diagnostics`: Additional diagnostic information

# Mathematical Analysis

## Decomposition
The analysis breaks down the approximation quality into components:
- **Likelihood term**: How well q(H) explains observations
- **Prior term**: How close q(H) is to the prior
- **Entropy term**: Information content of the approximation

## Optimality Check
Compares the provided approximation with the analytical optimal solution:
```
p*(H|V) = N(μ*, σ²*) where:
μ* = (μ₀/σ₀² + Σᵢ vᵢ/β²) / (1/σ₀² + n/β²)
σ²* = 1 / (1/σ₀² + n/β²)
```

# Examples
```julia
# Generate data and test approximation
Random.seed!(42)
h_true = 2.0
vs = rand(Normal(h_true, 1.0), 10)
q = Gaussian1DFromMeanVariance(1.8, 0.5)

# Analyze approximation quality
analysis = analyze_approximation_quality(q, vs)

println("ELBO: ", analysis.elbo)
println("KL divergence: ", analysis.kl_divergence)
println("Approximation gap: ", analysis.approximation_gap)
```

# Diagnostic Interpretation
- **High ELBO**: Good overall approximation
- **Low KL divergence**: Close to true posterior
- **Small approximation gap**: Near-optimal within variational family

# See Also
- [`compute_elbo_and_kl_divergence`](@ref): Core computation
- [`compare_approximations`](@ref): Compare multiple approximations
"""
function analyze_approximation_quality(
    q::Gaussian.Gaussian1D, 
    vs::AbstractVector{<:Real}; 
    β::Real = 1, 
    prior::Gaussian.Gaussian1D = Gaussian.Gaussian1D(0, 1)
)
    # Validate inputs
    β > 0 || throw(ArgumentError("Observation noise β must be positive"))
    !isempty(vs) || throw(ArgumentError("Observations vector cannot be empty"))
    
    # Compute basic metrics
    elbo, kl_divergence = compute_elbo_and_kl_divergence(q, vs; β=β, prior=prior)
    log_marginal_likelihood = elbo + kl_divergence
    
    # Compute optimal posterior analytically
    optimal_posterior = Gaussian.NonNormalizedGaussian1D(prior)
    for v in vs
        optimal_posterior *= Gaussian.Gaussian1DFromMeanVariance(v, β^2)
    end
    
    # Compute approximation gap (KL divergence from optimal)
    q_non_normalized = Gaussian.NonNormalizedGaussian1D(q)
    approximation_gap = Gaussian.KL_divergence(q_non_normalized, optimal_posterior)
    
    # Estimate effective sample size
    prior_precision = 1.0 / Gaussian.variance(prior)
    likelihood_precision = length(vs) / β^2
    effective_sample_size = likelihood_precision / (prior_precision + likelihood_precision)
    
    # Additional diagnostics
    q_mean = Gaussian.mean(q)
    q_var = Gaussian.variance(q)
    optimal_mean = Gaussian.mean(optimal_posterior)
    optimal_var = Gaussian.variance(optimal_posterior)
    
    diagnostics = (
        mean_error = abs(q_mean - optimal_mean),
        variance_ratio = q_var / optimal_var,
        relative_elbo = elbo / log_marginal_likelihood,
        data_fit = elbo + kl_divergence - Gaussian.KL_divergence(q_non_normalized, 
                                                                 Gaussian.NonNormalizedGaussian1D(prior))
    )
    
    return (
        elbo = elbo,
        kl_divergence = kl_divergence,
        log_marginal_likelihood = log_marginal_likelihood,
        optimal_posterior = optimal_posterior,
        approximation_gap = approximation_gap,
        effective_sample_size = effective_sample_size,
        diagnostics = diagnostics
    )
end

"""
    compare_approximations(approximations::Vector{Gaussian1D}, vs; β=1, prior::Gaussian1D=Gaussian1D(0,1), labels=nothing) -> Nothing

Compare multiple variational approximations side-by-side with visualization.

This function provides comparative analysis of different variational approximations,
helping to understand which approaches work better under different conditions.

# Arguments
- `approximations::Vector{Gaussian1D}`: List of variational approximations to compare
- `vs::AbstractVector{<:Real}`: Observed data points V₁, V₂, ..., Vₙ
- `β::Real=1`: Observation noise standard deviation (must be positive)
- `prior::Gaussian1D=Gaussian1D(0,1)`: Prior distribution p(H)
- `labels::Union{Vector{String},Nothing}=nothing`: Optional labels for approximations

# Returns
- `Nothing`: Displays comparison table and visualization

# Visualization Elements
- **Comparison table**: ELBO, KL divergence, and approximation gap for each method
- **Distribution overlay**: Visual comparison of approximations with true posterior
- **Performance ranking**: Sorted by approximation quality

# Examples
```julia
# Generate data
Random.seed!(42)
vs = rand(Normal(2.0, 1.0), 10)

# Define different approximations
tight = Gaussian1DFromMeanVariance(2.0, 0.1)    # Tight approximation
loose = Gaussian1DFromMeanVariance(2.0, 2.0)    # Loose approximation
biased = Gaussian1DFromMeanVariance(1.5, 0.5)   # Biased approximation

approximations = [tight, loose, biased]
labels = ["Tight", "Loose", "Biased"]

compare_approximations(approximations, vs; labels=labels)
```

# Performance Metrics
- **ELBO**: Higher is better (maximizes lower bound)
- **KL divergence**: Lower is better (closer to true posterior)
- **Approximation gap**: Lower is better (less approximation error)

# See Also
- [`analyze_approximation_quality`](@ref): Detailed analysis of single approximation
- [`plot_elbo_and_kl_divergence`](@ref): Visualization of ELBO landscape
"""
function compare_approximations(
    approximations::Vector{Gaussian.Gaussian1D}, 
    vs::AbstractVector{<:Real}; 
    β::Real = 1, 
    prior::Gaussian.Gaussian1D = Gaussian.Gaussian1D(0, 1),
    labels::Union{Vector{String},Nothing} = nothing
)
    # Validate inputs
    !isempty(approximations) || throw(ArgumentError("Must provide at least one approximation"))
    β > 0 || throw(ArgumentError("Observation noise β must be positive"))
    !isempty(vs) || throw(ArgumentError("Observations vector cannot be empty"))
    
    # Generate default labels if not provided
    if labels === nothing
        labels = ["Approximation $i" for i in 1:length(approximations)]
    else
        length(labels) == length(approximations) || 
            throw(ArgumentError("Number of labels must match number of approximations"))
    end
    
    # Analyze each approximation
    analyses = [analyze_approximation_quality(q, vs; β=β, prior=prior) for q in approximations]
    
    # Print comparison table
    println("="^80)
    println("VARIATIONAL APPROXIMATION COMPARISON")
    println("="^80)
    println(rpad("Method", 20), rpad("ELBO", 15), rpad("KL Divergence", 15), rpad("Approx Gap", 15), "Rank")
    println("-"^80)
    
    # Sort by ELBO (higher is better)
    sorted_indices = sortperm([a.elbo for a in analyses], rev=true)
    
    for (rank, idx) in enumerate(sorted_indices)
        analysis = analyses[idx]
        label = labels[idx]
        println(
            rpad(label, 20),
            rpad(string(round(analysis.elbo, digits=4)), 15),
            rpad(string(round(analysis.kl_divergence, digits=4)), 15),
            rpad(string(round(analysis.approximation_gap, digits=4)), 15),
            rank
        )
    end
    println("-"^80)
    
    # Create visualization comparing distributions
    optimal_posterior = analyses[1].optimal_posterior
    x_range = range(
        Gaussian.mean(optimal_posterior) - 3*sqrt(Gaussian.variance(optimal_posterior)),
        Gaussian.mean(optimal_posterior) + 3*sqrt(Gaussian.variance(optimal_posterior)),
        length=200
    )
    
    # Plot true posterior
    p = plot(
        x_range,
        [pdf(Normal(Gaussian.mean(optimal_posterior), sqrt(Gaussian.variance(optimal_posterior))), x) for x in x_range],
        linewidth=3,
        color=:black,
        label="True Posterior",
        xlabel="H",
        ylabel="Density",
        title="Comparison of Variational Approximations",
        legend=:topright,
        size=(800, 600)
    )
    
    # Plot each approximation
    colors = [:blue, :red, :green, :orange, :purple, :brown]
    for (i, (q, label)) in enumerate(zip(approximations, labels))
        color = colors[mod1(i, length(colors))]
        plot!(p,
            x_range,
            [pdf(Normal(Gaussian.mean(q), sqrt(Gaussian.variance(q))), x) for x in x_range],
            linewidth=2,
            color=color,
            label=label,
            linestyle=:dash
        )
    end
    
    display(p)
    
    return nothing
end

# === UTILITY FUNCTIONS ===

"""
    generate_synthetic_data(h_true::Real, n::Int; β=1, seed=nothing) -> Vector{Float64}

Generate synthetic observations from the hierarchical Gaussian model.

This utility function creates synthetic data for testing and demonstration purposes,
following the model specification: Vᵢ ~ N(h_true, β²) for i = 1,...,n.

# Arguments
- `h_true::Real`: True latent variable value
- `n::Int`: Number of observations to generate (must be positive)
- `β::Real=1`: Observation noise standard deviation (must be positive)
- `seed::Union{Int,Nothing}=nothing`: Optional random seed for reproducibility

# Returns
- `Vector{Float64}`: Generated observations

# Examples
```julia
# Generate data with default noise
vs = generate_synthetic_data(2.0, 10)

# Generate data with custom noise and seed
vs = generate_synthetic_data(2.0, 20; β=0.5, seed=42)
```

# See Also
- [`compute_elbo_and_kl_divergence`](@ref): Analyze generated data
- [`plot_elbo_and_kl_divergence`](@ref): Visualize with generated data
"""
function generate_synthetic_data(h_true::Real, n::Int; β::Real=1, seed::Union{Int,Nothing}=nothing)
    # Validate inputs
    n > 0 || throw(ArgumentError("Number of observations n must be positive"))
    β > 0 || throw(ArgumentError("Observation noise β must be positive"))
    
    # Set seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end
    
    # Generate observations
    return rand(Normal(h_true, β), n)
end

# === MAIN FUNCTION ===

"""
    main()

Demonstrate the ELBO analysis with default parameters.

This function provides a comprehensive demonstration of the ELBO analysis capabilities
with carefully chosen parameters that illustrate the key concepts of variational inference.

# Usage
```julia
# Run the main demonstration
ELBO.main()

# Or include in larger analysis
include("elbo.jl")
using .ELBO
main()
```

# Educational Value
The chosen parameters show:
- Clear ELBO curve with visible maximum
- Meaningful KL divergence variation
- Reasonable computational time
- Publication-quality visualization

# See Also
- [`plot_elbo_and_kl_divergence`](@ref): The main visualization function
- [`compute_elbo_and_kl_divergence`](@ref): Core computation
"""
function main()
    Random.seed!(1582)
    plot_elbo_and_kl_divergence(σ_hat = sqrt(1/11), prior=Gaussian.Gaussian1DFromMeanVariance(0,1), β = 1, n = 10, save_path = "~/Downloads/elbo_correct_variance.svg")
    Random.seed!(1582)
    plot_elbo_and_kl_divergence(σ_hat = sqrt(1/100), prior=Gaussian.Gaussian1DFromMeanVariance(0,1), β = 1, n = 10, save_path = "~/Downloads/elbo_small_variance.svg")
    Random.seed!(1582)
    plot_elbo_and_kl_divergence(σ_hat = sqrt(1), prior=Gaussian.Gaussian1DFromMeanVariance(0,1), β = 1, n = 10, save_path = "~/Downloads/elbo_large_variance.svg")
end

end  # module ELBO