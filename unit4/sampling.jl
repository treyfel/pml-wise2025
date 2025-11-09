"""
    SamplingPlots

A module demonstrating importance sampling for probabilistic inference,
with applications to the TrueSkill graphical model.

# Overview

This module implements importance sampling, a Monte Carlo technique for
approximating expectations and distributions when direct sampling is difficult
or inefficient.

# Importance Sampling

## Core Idea

To compute expectations under a target distribution p(x):
```
E_p[f(x)] = ∫ f(x)p(x)dx
```

When we can't sample from p(x) directly, we sample from a proposal q(x):
```
E_p[f(x)] = ∫ f(x) [p(x)/q(x)] q(x)dx ≈ (1/N) Σᵢ f(xᵢ) w(xᵢ)
```

where xᵢ ~ q(x) and w(xᵢ) = p(xᵢ)/q(xᵢ) are importance weights.

## Key Properties

- **Unbiased**: E[weighted estimate] = true expectation
- **Variance**: Depends on mismatch between p and q
- **Effective Sample Size**: Reduced when weights are unequal
- **Proposal Choice**: Critical for efficiency

## Good Proposal Distributions

1. **High overlap**: q(x) > 0 whenever p(x) > 0
2. **Heavy tails**: q should have heavier tails than p
3. **Similar shape**: q ≈ p reduces variance
4. **Easy to sample**: Must be able to draw samples from q

# TrueSkill Model

Hierarchical model for player skill estimation:
```
s₁, s₂ ~ N(μ, σ²)           # Prior skills
p₁ ~ N(s₁, β²)              # Performance given skill
p₂ ~ N(s₂, β²)
y = I(p₁ > p₂)              # Outcome (1 if player 1 wins)
```

## Inference Challenge

Computing p(s₁|y=1) requires integrating over performances:
```
p(s₁|y=1) ∝ ∫∫ p(s₁)p(p₁|s₁)p(s₂)p(p₂|s₂)I(p₁>p₂) dp₁dp₂ds₂
```

This integral is intractable analytically, motivating sampling approaches.

# Implementations

## Basic Importance Sampling
- Generic importance sampling for continuous distributions
- Visualization of weights and effective sample size
- Comparison of different proposal distributions

## TrueSkill Importance Sampling
- Samples from full joint distribution p(s₁,s₂,p₁,p₂,y)
- Computes marginals via weighted samples
- Handles discrete outcome y with indicator function

## Conditional Sampling
- Samples from p(s₁,s₂,p₁,p₂|y=1)
- Uses outcome constraint in importance weights
- Demonstrates inference given observed game results

# Author
Ralf Herbrich, Hasso-Plattner Institute, 2025

# References
- Liu, J. S. (2008). Monte Carlo Strategies in Scientific Computing.
- Herbrich, R., Minka, T., & Graepel, T. (2006). TrueSkill™: A Bayesian Skill Rating System.
"""
module SamplingPlots

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

# ============================================================================
# Generic Importance Sampling Visualization
# ============================================================================

"""
    plot_importance_sampling(dn, proposal; n=100000, bins=100, xlim=(-4, 4)) -> Nothing

Visualizes importance sampling by plotting weighted histogram vs. true density.

Demonstrates how samples from a proposal distribution q(x), when weighted by
w(x) = p(x)/q(x), approximate the target distribution p(x).

# Arguments
- `dn`: Target distribution (must support `pdf` and ideally `rand`)
- `proposal`: Proposal distribution (must support `pdf` and `rand`)

# Keywords
- `n::Int=100000`: Number of samples to draw
- `bins::Int=100`: Number of histogram bins
- `xlim::Tuple=(-4, 4)`: Range for x-axis

# Algorithm
1. Draw samples: xᵢ ~ q(x) for i=1,...,n
2. Compute weights: wᵢ = p(xᵢ)/q(xᵢ)
3. Plot weighted histogram of samples
4. Overlay true density p(x) for comparison

# Examples
```julia
# Good proposal: q ≈ p
plot_importance_sampling(Normal(0, 1), Normal(0, 1))

# Suboptimal proposal: q shifted from p
plot_importance_sampling(Normal(0, 1), Normal(1.5, 1))

# Poor proposal: uniform vs. Normal
plot_importance_sampling(Normal(0, 1), Uniform(-2.5, 2.5))
```

# Interpretation
- Blue histogram: Weighted empirical distribution
- Red curve: True target distribution
- Close match indicates good proposal and sufficient samples
- Large discrepancy suggests proposal mismatch or insufficient samples

# Notes
- Histogram is normalized to probability density (integrates to 1)
- Works best when proposal has support everywhere target does
- Extreme weights indicate poor proposal choice
"""
function plot_importance_sampling(dn, proposal; n=100000, bins = 100, xlim=(-4, 4))
    # Draw samples from proposal distribution
    xs = rand(proposal, n)
    
    # Compute importance weights: w(x) = p(x)/q(x)
    ws = pdf(dn, xs) ./ pdf(proposal, xs)
    
    # Create weighted histogram
    p = histogram(
        xs,
        weights=ws,
        color=:blue,
        normalize = :pdf,
        legend=false,
        bins = bins,
        xlim = xlim,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16
    )
    
    # Overlay true density for comparison
    xs = range(xlim[1], xlim[2], length=1000)
    plot!(xs, pdf(dn, xs), color=:red, linewidth=2)
    
    xlabel!(L"x")
    ylabel!(L"\hat{p}(x)")
    display(p)
end

"""
    plot_importance_weights(dn, proposal; xlim=(-4, 4)) -> Nothing

Plots the importance weight function w(x) = p(x)/q(x).

Visualizes how importance weights vary across the domain, helping diagnose
proposal quality and potential numerical issues.

# Arguments
- `dn`: Target distribution
- `proposal`: Proposal distribution

# Keywords
- `xlim::Tuple=(-4, 4)`: Range for x-axis

# Examples
```julia
# Well-matched distributions → weights near 1
plot_importance_weights(Normal(0, 1), Normal(0, 1))

# Shifted proposal → varying weights
plot_importance_weights(Normal(0, 1), Normal(1.5, 1))

# Heavy-tailed proposal → bounded weights
plot_importance_weights(Normal(0, 1), Uniform(-2.5, 2.5))
```

# Interpretation

**Good Proposal:**
- Weights roughly constant (near 1)
- No extreme values
- Low variance in weights

**Poor Proposal:**
- Highly varying weights
- Extreme values (>> 1 or ≈ 0)
- High variance → high estimation variance

**Warning Signs:**
- Weights → ∞: proposal has zero density where target doesn't
- Weights → 0: wasting samples in low-importance regions
- Highly peaked: few samples dominate the estimate

# Theory
Variance of importance sampling estimate:
```
Var[estimate] ∝ Var[w(x)]
```

Lower weight variance → better proposal → lower estimation variance.
"""
function plot_importance_weights(dn, proposal; xlim=(-4, 4))
    xs = range(xlim[1], xlim[2], length=1000)
    
    # Compute weight function: w(x) = p(x)/q(x)
    p = plot(
        xs,
        pdf(dn, xs) ./ pdf(proposal, xs),
        color=:red,
        linewidth=3,
        legend=false,
        xlim = xlim,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16
    )
    
    xlabel!(L"x")
    ylabel!(L"\frac{p(x)}{q(x)}")
    display(p)
end

# ============================================================================
# TrueSkill Importance Sampling
# ============================================================================

"""
    sample(; n=1000000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0, ...) -> (samples, weights)

Generates weighted samples from the TrueSkill joint distribution using importance sampling.

Samples from the full generative model and computes importance weights for
inference. Allows flexible choice of proposal distributions for each variable.

# TrueSkill Model

**Generative Process:**
```
s₁ ~ N(μ₁, σ₁²)           # Player 1 skill
s₂ ~ N(μ₂, σ₂²)           # Player 2 skill
p₁ ~ N(s₁, β²)            # Player 1 performance
p₂ ~ N(s₂, β²)            # Player 2 performance
y = I(p₁ > p₂)            # Outcome (1 if p1 wins, 0 otherwise)
```

# Keywords

**Model Parameters:**
- `n::Int=1000000`: Number of samples
- `μ1::Float64=0.0`: Prior mean skill for player 1
- `σ1::Float64=1.0`: Prior skill std dev for player 1
- `μ2::Float64=0.0`: Prior mean skill for player 2
- `σ2::Float64=1.0`: Prior skill std dev for player 2
- `β::Float64=1.0`: Performance noise std dev

**Proposal Distributions:**
- `proposal_s1`: Proposal for s₁ (default: prior N(μ₁,σ₁²))
- `proposal_s2`: Proposal for s₂ (default: prior N(μ₂,σ₂²))
- `proposal_p1`: Proposal for p₁ (default: N(μ₁,σ₁²+β²))
- `proposal_p2`: Proposal for p₂ (default: N(μ₂,σ₂²+β²))
- `proposal_y`: Proposal for y (default: Bernoulli(0.5))

# Returns
- `samples::Vector{Vector{Float64}}`: Each sample is [s₁, s₂, p₁, p₂, y]
- `weights::Vector{Float64}`: Importance weights for each sample

# Importance Weight Computation

For each sample, the weight is:
```
w = [p(s₁)p(s₂)p(p₁|s₁)p(p₂|s₂)p(y|p₁,p₂)] / [q(s₁)q(s₂)q(p₁)q(p₂)q(y)]
```

where p(y|p₁,p₂) = I(y matches sign(p₁-p₂)).

# Examples
```julia
# Default: proposal = prior (optimal for unconditioned inference)
samples, weights = sample(n=100000)

# Compute marginal for s₁
s1_values = [s[1] for s in samples]
histogram(s1_values, weights=weights, normalize=:pdf)

# Custom proposals for targeted inference
samples, weights = sample(
    n=100000,
    proposal_s1 = Normal(1.0, 0.5),  # Shifted proposal
    proposal_y = Bernoulli(0.7)      # Biased outcome proposal
)
```

# Default Proposal Choice

The default proposals are:
- Skills: Use priors (optimal when no conditioning)
- Performances: Use marginals N(μᵢ, σᵢ²+β²)
- Outcome: Uniform Bernoulli(0.5)

These are optimal for unconditional inference but can be tuned for
conditional queries (e.g., given y=1).

# Use Cases
- Approximating marginal distributions
- Computing expectations under the model
- Comparing with exact inference (belief propagation)
- Studying effect of model parameters
"""
function sample(; n = 1000000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0,
    proposal_s1 = Normal(μ1, σ1), proposal_s2 = Normal(μ2, σ2),
    proposal_p1 = Normal(μ1, sqrt(σ1^2 + β^2)), proposal_p2 = Normal(μ2, sqrt(σ2^2 + β^2)),
    proposal_y = Bernoulli(0.5))

    # Pre-allocate storage
    samples = Vector{Vector{Float64}}(undef, n)
    weights = Vector{Float64}(undef, n)
    
    # Draw all samples from proposals (vectorized for efficiency)
    s1 = rand(proposal_s1, n)
    s2 = rand(proposal_s2, n)
    p1 = rand(proposal_p1, n)
    p2 = rand(proposal_p2, n)
    y = rand(proposal_y, n)

    # Compute importance weights for each sample
    for i in 1:n
        # Target distribution factors
        f1 = pdf(Normal(μ1, σ1), s1[i])        # p(s₁)
        f2 = pdf(Normal(μ2, σ2), s2[i])        # p(s₂)
        f3 = pdf(Normal(s1[i], β), p1[i])      # p(p₁|s₁)
        f4 = pdf(Normal(s2[i], β), p2[i])      # p(p₂|s₂)
        # Indicator function: p(y|p₁,p₂)
        f5 = if (y[i] == 1 && p1[i] > p2[i]) || (y[i] == 0 && p1[i] < p2[i]) 
            1.0 
        else 
            0.0 
        end 

        # Proposal distribution factors
        g1 = pdf(proposal_s1, s1[i])
        g2 = pdf(proposal_s2, s2[i])
        g3 = pdf(proposal_p1, p1[i])
        g4 = pdf(proposal_p2, p2[i])
        g5 = pdf(proposal_y, y[i])

        # Importance weight: w = p(x) / q(x)
        weights[i] = f1 * f2 * f3 * f4 * f5 / (g1 * g2 * g3 * g4 * g5)
        samples[i] = [s1[i], s2[i], p1[i], p2[i], y[i]]
    end
    
    return samples, weights
end

"""
    sample_with_outcome(; n=1000000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0, ...) -> (samples, weights)

Samples from TrueSkill model conditioned on outcome y=1 (player 1 wins).

Implements importance sampling for conditional inference: p(s₁,s₂,p₁,p₂|y=1).
Useful for updating beliefs about skills after observing a game outcome.

# Conditional Distribution

Computes:
```
p(s₁,s₂,p₁,p₂|y=1) ∝ p(s₁)p(s₂)p(p₁|s₁)p(p₂|s₂)I(p₁ > p₂)
```

# Keywords

**Model Parameters:** (same as `sample`)
- `n::Int=1000000`: Number of samples
- `μ1, σ1, μ2, σ2, β`: Model parameters

**Proposal Distributions:**
- `proposal_s1, proposal_s2`: Skill proposals
- `proposal_p1, proposal_p2`: Performance proposals
- No `proposal_y` (outcome is fixed to 1)

# Returns
- `samples::Vector{Vector{Float64}}`: Each sample is [s₁, s₂, p₁, p₂]
- `weights::Vector{Float64}`: Importance weights

# Importance Weights

```
w = [p(s₁)p(s₂)p(p₁|s₁)p(p₂|s₂)I(p₁>p₂)] / [q(s₁)q(s₂)q(p₁)q(p₂)]
```

Note: No y in denominator since outcome is deterministic given evidence.

# Examples
```julia
# Sample from posterior given player 1 won
samples, weights = sample_with_outcome(n=100000)

# Extract posterior samples for s₁
s1_posterior = [s[1] for s in samples]
histogram(s1_posterior, weights=weights, normalize=:pdf)

# Compare to prior
samples_prior, weights_prior = sample(n=100000)
s1_prior = [s[1] for s in samples_prior]
```

# Effective Sample Size

Many samples will have zero weight (when p₁ < p₂), reducing efficiency.
The effective sample size is approximately:
```
ESS ≈ n × P(p₁ > p₂) = n × Φ((μ₁-μ₂)/√(σ₁²+σ₂²+2β²))
```

where Φ is the standard normal CDF.

# Use Cases
- Skill updates after observed game
- Posterior inference given evidence
- Comparing importance sampling vs. belief propagation
- Studying effect of evidence on uncertainty
"""
function sample_with_outcome(; n = 1000000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0,
    proposal_s1 = Normal(μ1, σ1), proposal_s2 = Normal(μ2, σ2),
    proposal_p1 = Normal(μ1, sqrt(σ1^2 + β^2)), proposal_p2 = Normal(μ2, sqrt(σ2^2 + β^2)))

    # Pre-allocate storage
    samples = Vector{Vector{Float64}}(undef, n)
    weights = Vector{Float64}(undef, n)
    
    # Draw all samples from proposals
    s1 = rand(proposal_s1, n)
    s2 = rand(proposal_s2, n)
    p1 = rand(proposal_p1, n)
    p2 = rand(proposal_p2, n)

    # Compute importance weights for conditional distribution
    for i in 1:n
        # Target distribution factors
        f1 = pdf(Normal(μ1, σ1), s1[i])        # p(s₁)
        f2 = pdf(Normal(μ2, σ2), s2[i])        # p(s₂)
        f3 = pdf(Normal(s1[i], β), p1[i])      # p(p₁|s₁)
        f4 = pdf(Normal(s2[i], β), p2[i])      # p(p₂|s₂)
        # Indicator for outcome y=1: p(y=1|p₁,p₂) = I(p₁>p₂)
        f5 = (p1[i] > p2[i]) ? 1.0 : 0.0

        # Proposal distribution factors
        g1 = pdf(proposal_s1, s1[i])
        g2 = pdf(proposal_s2, s2[i])
        g3 = pdf(proposal_p1, p1[i])
        g4 = pdf(proposal_p2, p2[i])

        # Importance weight for conditional distribution
        weights[i] = f1 * f2 * f3 * f4 * f5 / (g1 * g2 * g3 * g4)
        samples[i] = [s1[i], s2[i], p1[i], p2[i]]
    end
    
    return samples, weights
end

# ============================================================================
# Visualization Functions
# ============================================================================

"""
    plot_histogram(xss, wss; ylabel="Frequency", xlabel="x", xlim=(-5, 5), bins=100) -> Nothing

Plots overlaid weighted histograms for comparing distributions.

Creates normalized probability density histograms from weighted samples,
useful for comparing prior vs. posterior or different inference methods.

# Arguments
- `xss::Vector{Vector}`: Multiple sets of samples to plot
- `wss::Vector{Vector}`: Corresponding importance weights

# Keywords
- `ylabel::String="Frequency"`: Y-axis label
- `xlabel::String="x"`: X-axis label
- `xlim::Tuple=(-5, 5)`: X-axis range
- `bins::Int=100`: Number of histogram bins

# Examples
```julia
# Compare prior and posterior for s₁
samples_prior, w_prior = sample(n=100000)
samples_post, w_post = sample_with_outcome(n=100000)

s1_prior = [[s[1] for s in samples_prior]]
s1_post = [[s[1] for s in samples_post]]

plot_histogram(
    [s1_prior, s1_post], 
    [w_prior, w_post],
    xlabel=L"s_1"
)
```

# Notes
- All histograms normalized to probability densities
- Multiple distributions overlaid with transparency
- Weighted histogram accounts for importance weights
"""
function plot_histogram(xss, wss; ylabel = "Frequency", xlabel = "x", xlim = (-5, 5), bins = 100)
    # Initialize plot
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
    
    # Add weighted histogram for each dataset
    for i in eachindex(xss)
        histogram!(xss[i], weights=wss[i], label=false, bins=bins, normalize=:pdf, alpha=0.5)
    end
    
    ylabel!(ylabel)
    xlabel!(xlabel)
    display(p)
end

"""
    plot_bars(xs, ws; ylabel="Frequency", xlabel="x") -> Nothing

Plots weighted bar chart for discrete outcome distribution.

Computes and displays the empirical probability mass function for binary
outcomes using importance weights.

# Arguments
- `xs::Vector`: Outcome values (0 or 1 for Bernoulli samples)
- `ws::Vector`: Importance weights

# Keywords
- `ylabel::String="Frequency"`: Y-axis label
- `xlabel::String="x"`: X-axis label

# Examples
```julia
samples, weights = sample(n=100000)
outcomes = [s[5] for s in samples]
plot_bars(outcomes, weights, ylabel=L"\\hat{p}(y)", xlabel=L"y")
```

# Notes
- Properly normalizes weights: p(y=k) = Σᵢ wᵢI(yᵢ=k) / Σᵢwᵢ
- Maps Bernoulli {0,1} to outcome labels {-1,+1} for display
"""
function plot_bars(xs, ws; ylabel = "Frequency", xlabel = "x")
    # Normalize weights
    Z = sum(ws)
    y_minus_1_frac = sum(ws[xs .== 0]) / Z  # P(y=0) mapped to -1 bar
    y_plus_1_frac = sum(ws[xs .== 1]) / Z   # P(y=1) mapped to +1 bar
    
    # Create bar plot
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
    
    ylabel!(ylabel)
    xlabel!(xlabel)
    display(p)
end

# ============================================================================
# Main Demonstration Function
# ============================================================================

"""
    main() -> Nothing

Runs comprehensive demonstrations of importance sampling techniques.

Generates visualizations showing:
1. Generic importance sampling with different proposals
2. TrueSkill model marginal distributions
3. Conditional inference given game outcome
4. Comparison of proposals and their importance weights

# Output Files

Generated in ~/Downloads/:

**Generic Importance Sampling:**
- `importance_sampling.svg`: Perfect proposal (q=p)
- `importance_weights.svg`: Weights for perfect proposal
- `importance_sampling2.svg`: Shifted proposal
- `importance_weights2.svg`: Weights for shifted proposal
- `importance_sampling3.svg`: Uniform proposal
- `importance_weights3.svg`: Weights for uniform proposal

**TrueSkill Marginals:**
- `s1_importance.svg`, `s2_importance.svg`: Skill distributions
- `p1_importance.svg`, `p2_importance.svg`: Performance distributions
- `y_importance.svg`: Outcome distribution

**Conditional Inference:**
- `s1_importance_with_outcome.svg`: Prior vs. posterior for s₁
- `s2_importance_with_outcome.svg`: Prior vs. posterior for s₂
- `p1_importance_with_outcome.svg`: Prior vs. posterior for p₁
- `p2_importance_with_outcome.svg`: Prior vs. posterior for p₂

# Pedagogical Value

Demonstrates:
- Effect of proposal choice on importance sampling
- How weight variance affects estimation quality
- Inference in hierarchical graphical models
- Conditional vs. unconditional distributions
- Visual comparison with exact methods

# Notes
- Uses 1,000,000 samples for smooth estimates
- Random seed 2025 for reproducibility
- Generates publication-quality SVG figures
"""
function main()
    Random.seed!(2025)

    println("Generating generic importance sampling examples...")
    
    # Example 1: Perfect proposal (q = p)
    println("  - Perfect proposal (q = p)")
    plot_importance_sampling(Normal(0, 1), Normal(0, 1), n=1000000)
    savefig("~/Downloads/importance_sampling.svg")
    plot_importance_weights(Normal(0, 1), Normal(0, 1))
    savefig("~/Downloads/importance_weights.svg")
    
    # Example 2: Suboptimal proposal (shifted mean)
    println("  - Shifted proposal")
    plot_importance_sampling(Normal(0, 1), Normal(1.5, 1), n=1000000)
    savefig("~/Downloads/importance_sampling2.svg")
    plot_importance_weights(Normal(0, 1), Normal(1.5, 1))
    savefig("~/Downloads/importance_weights2.svg")
    
    # Example 3: Poor proposal (uniform)
    println("  - Uniform proposal")
    plot_importance_sampling(Normal(0, 1), Uniform(-2.5, 2.5), n=1000000)
    savefig("~/Downloads/importance_sampling3.svg")
    plot_importance_weights(Normal(0, 1), Uniform(-2.5, 2.5))
    savefig("~/Downloads/importance_weights3.svg")

    println("\nGenerating TrueSkill importance sampling examples...")
    
    # Sample from full TrueSkill model
    println("  - Sampling from joint distribution...")
    samples, weights = sample()
    
    # Sample with outcome constraint
    println("  - Sampling with outcome y=1...")
    samples_with_outcome, weights_with_outcome = sample_with_outcome()
    
    # Plot marginal distributions (unconditional)
    println("  - Plotting marginal distributions...")
    plot_histogram([[x[1] for x in samples]], [weights], ylabel=L"\hat{p}\left(s_1\right)", xlabel=L"s_1")
    savefig("~/Downloads/s1_importance.svg")
    plot_histogram([[x[2] for x in samples]], [weights], ylabel=L"\hat{p}\left(s_2\right)", xlabel=L"s_2")
    savefig("~/Downloads/s2_importance.svg")
    plot_histogram([[x[3] for x in samples]], [weights], ylabel=L"\hat{p}\left(p_1\right)", xlabel=L"p_1")
    savefig("~/Downloads/p1_importance.svg")
    plot_histogram([[x[4] for x in samples]], [weights], ylabel=L"\hat{p}\left(p_2\right)", xlabel=L"p_2")
    savefig("~/Downloads/p2_importance.svg")
    plot_bars([x[5] for x in samples], weights, ylabel=L"\hat{p}\left(y\right)", xlabel=L"y")
    savefig("~/Downloads/y_importance.svg")

    # Plot conditional distributions (prior vs. posterior overlaid)
    println("  - Plotting conditional distributions (prior vs posterior)...")
    plot_histogram([[x[1] for x in samples], [x[1] for x in samples_with_outcome]], [weights, weights_with_outcome], ylabel=L"\hat{p}\left(s_1\right)", xlabel=L"s_1")
    savefig("~/Downloads/s1_importance_with_outcome.svg")
    plot_histogram([[x[2] for x in samples], [x[2] for x in samples_with_outcome]], [weights, weights_with_outcome], ylabel=L"\hat{p}\left(s_2\right)", xlabel=L"s_2")
    savefig("~/Downloads/s2_importance_with_outcome.svg")
    plot_histogram([[x[3] for x in samples], [x[3] for x in samples_with_outcome]], [weights, weights_with_outcome], ylabel=L"\hat{p}\left(p_1\right)", xlabel=L"p_1")
    savefig("~/Downloads/p1_importance_with_outcome.svg")
    plot_histogram([[x[4] for x in samples], [x[4] for x in samples_with_outcome]], [weights, weights_with_outcome], ylabel=L"\hat{p}\left(p_2\right)", xlabel=L"p_2")
    savefig("~/Downloads/p2_importance_with_outcome.svg")
    
    println("\nAll visualizations generated successfully!")
    println("Files saved to ~/Downloads/")
end

end