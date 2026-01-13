"""
# Direct Message Approximation (DMA) — KL Divergence Analysis and Visualization

A teaching module that visualizes and analyzes the behavior of direct message approximation (DMA)
in simple factor graphs. It focuses on comparing exact vs approximate messages and marginals for
product factors, and explores KL divergences between Normal and LogNormal distributions under
moment matching and parameter sweeps.

## What This Module Demonstrates
- DMA-derived messages to variables vs exact messages from integration
- Approximate vs exact marginals and their normalization
- KL( Normal || LogNormal ) under moment-matched LogNormal for varying `μ` and `σ`
- KL( LogNormal || Normal ) under moment-matched Normal for varying `μ` and `σ`

## Outputs
- Side-by-side plots for messages and marginals (SVG)
- Heatmaps and contours for KL divergences (SVG)

## Dependencies
- Distributions, Plots, LaTeXStrings, QuadGK, Random, Statistics, LinearAlgebra

2026 by Ralf Herbrich
Hasso-Plattner Institute
"""
module DMA

include("product.jl")
include("relu.jl")

using Distributions
using Plots
using Random
using Statistics
using StatsFuns
using LinearAlgebra
using LaTeXStrings
using QuadGK
using .ProductFactorModule
using .ReLUFactorModule

"""
    Base.:*(d1::Normal{Float64}, d2::Normal{Float64})

Multiply two Gaussian distributions using precision-weighted combination.

This operator override implements Gaussian multiplication (product of two Gaussian PDFs),
which is a fundamental operation in probabilistic inference and message passing algorithms.
The product of two Gaussians is proportional to another Gaussian.

Given two Gaussians N(μ₁, σ₁²) and N(μ₂, σ₂²), their product is:
p(x) ∝ N(x; μ₁, σ₁²) × N(x; μ₂, σ₂²) = N(x; μ, σ²)

where the precision-weighted parameters are:
- ρ₁ = 1/σ₁², ρ₂ = 1/σ₂² (precisions)
- μ = (μ₁ρ₁ + μ₂ρ₂)/(ρ₁ + ρ₂) (precision-weighted mean)
- σ² = 1/(ρ₁ + ρ₂) (combined precision)

This operation is used extensively in:
- Belief propagation and message passing
- Bayesian inference (combining prior and likelihood)
- Factor graph computations

# Arguments
- `d1::Normal{Float64}`: First Gaussian distribution
- `d2::Normal{Float64}`: Second Gaussian distribution

# Returns
- `Normal{Float64}`: Product distribution (normalized Gaussian)

# Example
```julia
prior = Normal(0.0, 2.0)
likelihood = Normal(1.0, 1.0)
posterior = prior * likelihood  # N(0.67, 0.89)
```

# Mathematical Note
The product is computed in precision (inverse variance) space for numerical stability
and mathematical elegance. The precision of the product is the sum of the individual
precisions: ρ = ρ₁ + ρ₂.
"""
function Base.:*(d1::Normal{Float64}, d2::Normal{Float64})
    μ1, σ1 = mean(d1), std(d1)
    μ2, σ2 = mean(d2), std(d2)

    precision1 = 1.0 / σ1^2
    precision2 = 1.0 / σ2^2

    μ = (μ1 * precision1 + μ2 * precision2) / (precision1 + precision2)
    σ2 = 1.0 / (precision1 + precision2)

    return Normal(μ, sqrt(σ2))
end

"""
    plot_α_divergences(; prior_X, prior_Y, prior_Z, approximate_msg_to_Y, base_path)

Constructs a simple factor graph with three variables (X, Y, Z) connected by a product factor
`Z = X * Y` and their Gaussian priors. Computes both exact and approximate messages/marginals
for `Y`, then visualizes the comparison.

# Arguments
- `prior_X::Normal{Float64}`: Prior for `X` (default: 𝒩(2.0, 1.0))
- `prior_Y::Normal{Float64}`: Prior for `Y` (default: 𝒩(3.0, 1.0))
- `prior_Z::Normal{Float64}`: Prior for `Z` (default: 𝒩(10.0, 1.0))
- `approximate_msg_to_Y::Union{Nothing, Normal{Float64}}`: Optional approximate message to `Y`. When `nothing`, uses DMA-derived message (default: `nothing`)
- `base_path::Union{String, Nothing}`: Base path for saving plots. When `nothing`, plots are not saved (default: "~/Downloads/division_")

# Returns
Nothing. Displays plots and prints diagnostic information to the console.

# Method
1. Build factor graph with priors and product factor
2. Run DMA to update messages to all variables
3. Compute exact message and marginal via numerical integration and normalization
4. Compare normalized exact curves with approximate curves
5. Save figures if `base_path` is provided

# Examples
```julia
# Compare exact vs approximate DMA marginalization
plot_α_divergences(
    prior_X = Normal(2.0, sqrt(0.5)),
    prior_Y = Normal(3.0, sqrt(0.5)),
    prior_Z = Normal(6.0, 1.0),
    base_path = "~/Downloads/exact_",
)

# Test with a poor approximation
plot_α_divergences(
    prior_X = Normal(2.0, 1.0),
    prior_Y = Normal(3.0, 1.0),
    prior_Z = Normal(10.0, 1.0),
    approximate_msg_to_Y = Normal(-6.0, 1.0),
    base_path = "~/Downloads/approx_",
)
```
"""
function plot_α_divergences(;
    prior_X = Normal(2.0, 1.0),
    prior_Y = Normal(3.0, 1.0),
    prior_Z = Normal(10.0, 1.0),
    approximate_msg_to_Y = nothing,
    base_path = "~/Downloads/division_",
)
    # Extract the parameters of the priors
    μ_X, σ2_X = mean(prior_X), var(prior_X)
    μ_Y, σ2_Y = mean(prior_Y), var(prior_Y)
    μ_Z, σ2_Z = mean(prior_Z), var(prior_Z)

    # Define approximate marginal and message functions
    # If no approximate message is provided, use DMA from the product factor
    approximate_msg_to_Y = isnothing(approximate_msg_to_Y) ? approximate_product_factor_msg_to_y(prior_Z, prior_X) : approximate_msg_to_Y
    approximate_marginal_Y = prior_Y * approximate_msg_to_Y

    println("Approximate message to Y: ", approximate_msg_to_Y)

    # Define exact marginal and message functions
    # Exact marginal: p(y) ∝ p(z | x, y) * p(x) * p(y) integrated over x
    real_marginal_Y = y_val -> pdf(Normal(μ_Z, y_val^2 * σ2_X + σ2_Z), μ_X * y_val) * pdf(Normal(μ_Y, sqrt(σ2_Y)), y_val)

    # Exact message: p(z | x, y) * p(x) integrated over x
    real_msg_to_Y = y_val -> pdf(Normal(μ_Z, y_val^2 * σ2_X + σ2_Z), μ_X * y_val)

    # Determine integration range for numerical normalization of messages
    y_min = mean(approximate_msg_to_Y) - 16 * std(approximate_msg_to_Y)
    y_max = mean(approximate_msg_to_Y) + 16 * std(approximate_msg_to_Y)
    y_range = range(y_min, y_max, length=10000)

    # Numerically compute the normalization constant for the exact message
    Z_real_message = sum(real_msg_to_Y.(y_range)) * step(y_range)

    # Plot comparison of exact and approximate messages to Y
    plt = plot(
        y_range,
        y -> real_msg_to_Y(y) / Z_real_message,
        linewidth=3,
        color = :blue,
        label = L"m_{f \to X}(x)",
        xlabel = L"x",
        ylabel = L"m_{f \to X}(x)",
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        legendfontsize = 12,
    )
    plot!(
        y_range,
        y_val -> pdf(approximate_msg_to_Y, y_val),
        linewidth=3,
        color = :red,
        label = L"\hat{m}_{f \to X}(x)",
    )
    display(plt)
    if !isnothing(base_path)
        savefig(plt, base_path * "messages.svg")
    end

    # Determine integration range for marginals (narrower range: ±4 standard deviations)
    y_min = mean(approximate_marginal_Y) - 4 * std(approximate_marginal_Y)
    y_max = mean(approximate_marginal_Y) + 4 * std(approximate_marginal_Y)
    y_min = min(y_min, μ_Y - 4.0 * sqrt(σ2_Y))  # ensure we cover negative values as well
    y_max = max(y_max, μ_Y + 4.0 * sqrt(σ2_Y))  # ensure we cover negative values as well
    y_range = range(y_min, y_max, length=10000)
    
    # Compute normalization constant for exact marginal via numerical integration
    Z = sum(real_marginal_Y.(y_range)) * step(y_range)
    println("Normalization constant of real marginal of y: ", Z)

    plt = plot(
        y_range,
        y -> real_marginal_Y(y) / Z,
        linewidth=3,
        color = :blue,
        label = L"m_{f \to X}(x) \cdot m_{X \to f}(x)",
        xlabel = L"x",
        ylabel = L"p(x)",
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        legendfontsize = 12,
    )
    plot!(
        y_range,
        y_val -> pdf(approximate_marginal_Y, y_val),
        linewidth=3,
        color = :red,
        label = L"\hat{m}_{f \to X}(x) \cdot m_{X \to f}(x)",
    )
    display(plt)
    if !isnothing(base_path)
        savefig(plt, base_path * "marginals.svg")
    end
end

"""
    plot_kl_divergence_normal_lognormal(; μ_range, σ_range)

Visualize KL divergence KL( Normal(μ, σ²) || LogNormal(m, s²) ) where the LogNormal
parameters `(m, s²)` are chosen by moment matching to the Normal `(μ, σ)` restricted
to positive support.

# Description
- For each `(μ, σ)` in the provided ranges, compute a moment-matched LogNormal and
    evaluate KL divergence from the (truncated) Normal to that LogNormal.
- Render a heatmap with optional contours to highlight KL levels.

# Arguments
- `μ_range`: Range of Normal means to sweep (μ > 0)
- `σ_range`: Range of Normal standard deviations to sweep (σ ≥ 0)

# Returns
Nothing. Displays a heatmap of KL values.

# Output
Use `savefig("~/Downloads/kl_normal_lognormal.svg")` after calling this function
to persist the visualization.
"""
function plot_kl_divergence_normal_lognormal(;
    μ_range = range(0.1, stop = 10.0, length = 200),
    σ_range = range(0.1, stop = 10.0, length = 200),
)
    function lognormal_params_from_meanvar(μ::Real, σ::Real)
        μ > 0 || throw(ArgumentError("LogNormal moment-matching requires μ > 0. Got μ=$μ"))
        σ ≥ 0 || throw(ArgumentError("σ must be ≥ 0. Got σ=$σ"))

        # s^2 = log(1 + σ^2 / μ^2) computed stably
        r = (σ/μ)^2
        s2 = log1p(r)
        m  = log(μ) - 0.5*s2
        return m, s2
    end

    function kl_normal_lognormal_momentmatched(μ, σ; rtol=1e-10, atol=0.0)
        # Degenerate case: σ == 0 => X is a point mass at μ > 0; KL is finite and easy.
        if σ == 0
            m, s2 = lognormal_params_from_meanvar(μ, 0.0)
            q = LogNormal(m, sqrt(s2))
            # KL(delta_μ || q) = -log q(μ) - H(delta) ; treat H(delta)=0 convention here
            return -logpdf(q, μ)
        end

        # Truncated Normal on (0,∞): p_T(x) = p_N(x) / Z for x>0, with Z = P(X>0)
        m, s2 = lognormal_params_from_meanvar(μ, σ)
        q = LogNormal(m, sqrt(s2))
        Nstd = Normal()                 # standard normal for z
        a = (0.0 - μ) / σ               # truncation in z-space: z > a
        logZ = logccdf(Nstd, a)         # log P(Z > a) computed stably

        # Expectation under truncated normal:
        # E_T[g(X)] = (1/Z) ∫_{a}^{∞} g(μ+σ z) φ(z) dz
        #
        # KL = E_T[ log p_T(X) - log q(X) ]
        # log p_T(x) = log p_N(x) - logZ
        N = Normal(μ, σ)

        integrand(z) = begin
            x = μ + σ*z
            # x should be > 0 for z>a, but guard against roundoff:
            if x <= 0
                return 0.0
            end
            logpT = logpdf(N, x) - logZ
            logq  = logpdf(q, x)
            # weight φ(z) already in pdf(Nstd,z)
            return (logpT - logq) * pdf(Nstd, z)
        end

        val, err = quadgk(integrand, a, Inf; rtol=rtol, atol=atol)
        # divide by Z = exp(logZ) stably:
        return val * exp(-logZ)
    end

    KL_values = zeros(length(σ_range), length(μ_range))
    for (i, μ) in enumerate(μ_range)
        for (j, σ) in enumerate(σ_range)
            KL_values[j, i] = kl_normal_lognormal_momentmatched(μ, σ) + 1e-8
        end
    end

    plt = heatmap(
        μ_range,
        σ_range,
        KL_values,
        xlabel = L"\mu_X",
        ylabel = L"\sigma_X",
        zlabel = L"\mathrm{KL}\left(\mathcal{N}(\cdot;\mu_X, \sigma_X^2) \;||\; \mathrm{LN}(\cdot)\right)",
        xticks = range(minimum(μ_range), stop = maximum(μ_range), step = 2.0),
        yticks = range(minimum(σ_range), stop = maximum(σ_range), step = 2.0),
        xtickfontsize = 12,
        ytickfontsize = 12,
        xguidefontsize = 14,
        yguidefontsize = 14,
        zguidefontsize = 14,
        color = :viridis,
        # colorbar_scale = :log2,
        # colorbar_ticks = 2.0 .^ (-9:0),
        colorbar_formatter = :scientific,
    )
    contour!(
        μ_range,
        σ_range,
        KL_values,
        levels = 2.0 .^ (-5:2),
        linewidth = 1.5,
        linecolor = :white,
        label = false,
    )
    display(plt)
end

"""
    plot_kl_divergence_lognormal_normal(; μ_range=range(-3.0, 3.0, length=200), σ_range=range(0.1, 1.0, length=200))

Visualize KL divergence KL( LogNormal(μ, σ²) || Normal ) for a grid of LogNormal parameters.

This function computes and visualizes the Kullback-Leibler divergence from a LogNormal
distribution to its moment-matched Normal approximation. The KL divergence measures how
much information is lost when approximating the LogNormal with a Gaussian.

The closed-form KL divergence used is:
KL(LN(μ,σ²) || N) = ½(σ² + log(exp(σ²)-1) - log(σ²))

This formula is independent of μ (the location parameter), which is reflected in the
heatmap showing constant KL values along horizontal slices. The divergence increases
with σ, as larger scale parameters make the LogNormal more skewed and harder to
approximate with a symmetric Gaussian.

# Arguments
- `μ_range::AbstractRange=range(-3.0, 3.0, length=200)`: Range of LogNormal location parameters μ
- `σ_range::AbstractRange=range(0.1, 1.0, length=200)`: Range of LogNormal scale parameters σ (σ > 0)

# Returns
Nothing. Displays a heatmap with contour lines showing constant KL levels.

# Visualization Details
- Heatmap uses viridis colormap
- Contour lines at levels 2^(-5) through 2^2
- White contour lines for visibility
- Scientific notation for colorbar

# Output
Use `savefig("~/Downloads/kl_lognormal_normal.svg")` after calling to save the plot.

# Mathematical Background
The KL divergence KL(p||q) = ∫ p(x) log(p(x)/q(x)) dx measures the expected
logarithmic difference between distributions p and q. For LogNormal→Normal
approximation, it quantifies the approximation error in expectation-based inference.

# Example
```julia
plot_kl_divergence_lognormal_normal(
    μ_range=range(-5.0, 5.0, length=300),
    σ_range=range(0.05, 2.0, length=300)
)
savefig("~/Downloads/kl_lognormal_normal_extended.svg")
```
"""
function plot_kl_divergence_lognormal_normal(;
    μ_range = range(-3.0, stop = 3.0, length = 200),
    σ_range = range(0.1, stop = 1.0, length = 200),
)
    KL_values = zeros(length(σ_range), length(μ_range))
    for (i, μ) in enumerate(μ_range)
        for (j, σ) in enumerate(σ_range)
            KL_values[j, i] = 1/2 * (σ^2 + log(expm1(σ^2)) - log(σ^2))
        end
    end

    plt = heatmap(
        μ_range,
        σ_range,
        KL_values,
        xlabel = L"\mu_Y",
        ylabel = L"\sigma_Y",
        zlabel = L"\mathrm{KL}\left(\mathrm{LN}(\cdot;\mu_Y, \sigma_Y^2) \;||\; \mathcal{N}(\cdot)\right)",
        xticks = range(minimum(μ_range), stop = maximum(μ_range), step = 2.0),
        yticks = range(minimum(σ_range), stop = maximum(σ_range), step = 0.3),
        xtickfontsize = 12,
        ytickfontsize = 12,
        xguidefontsize = 14,
        yguidefontsize = 14,
        zguidefontsize = 14,
        color = :viridis,
        colorbar_formatter = :scientific,
    )
    contour!(
        μ_range,
        σ_range,
        KL_values,
        levels = 2.0 .^ (-5:2),
        linewidth = 1.5,
        linecolor = :white,
        label = false,
    )
    display(plt)
end

"""
    plot_kl_divergence_ReLU_normal(; μ_range=range(-3.0, 3.0, length=200), σ_range=range(1e-4, 1.0, length=200), α=0.1)

Visualize KL divergence between exact ReLU output distribution and its moment-matched Gaussian approximation.

This function computes and visualizes the Kullback-Leibler divergence from the true
output distribution of a leaky ReLU transformation y = ReLU_α(x) to its moment-matched
Gaussian approximation, where x ~ N(μ, σ²).

The KL divergence is computed using the closed-form expression:
KL = ½ log(σ²_out/σ²_in) - q·log(α)

where:
- σ²_out is the variance of the approximate output Gaussian
- σ²_in is the variance of the input Gaussian
- q = P(X < 0) is the probability mass in the negative region
- α is the leakage parameter

When α = 1 (identity function), KL = 0 exactly. As α → 0 (standard ReLU), the
approximation error increases, especially when significant mass lies in x < 0.

# Arguments
- `μ_range::AbstractRange=range(-3.0, 3.0, length=200)`: Range of input Gaussian means μ
- `σ_range::AbstractRange=range(1e-4, 1.0, length=200)`: Range of input Gaussian std deviations σ
- `α::Float64=0.1`: Leakage parameter for ReLU_α (0 ≤ α ≤ 1)
  - α = 0: Standard ReLU (highest approximation error)
  - α = 1: Identity (zero approximation error)
  - 0 < α < 1: Leaky ReLU (intermediate error)

# Returns
Nothing. Displays a heatmap with contours showing KL divergence levels.

# Visualization Details
- Heatmap uses viridis colormap
- Contour lines at power-of-2 levels: 2^(-5), 2^(-4), ..., 2^2
- Highest KL values typically occur when μ < 0 (most mass gets scaled by α)

# Mathematical Insight
The KL divergence captures how much information is lost when replacing the true
(mixed/truncated) ReLU output distribution with a Gaussian. This is crucial for
understanding the approximation quality in neural network inference with
probabilistic message passing.

# Example
```julia
# Standard ReLU approximation error
plot_kl_divergence_ReLU_normal(α=0.0)
savefig("~/Downloads/kl_relu_standard.svg")

# Leaky ReLU with α=0.2
plot_kl_divergence_ReLU_normal(
    μ_range=range(-5.0, 5.0, length=300),
    σ_range=range(0.1, 2.0, length=300),
    α=0.2
)
savefig("~/Downloads/kl_relu_leaky_0.2.svg")
```
"""
function plot_kl_divergence_ReLU_normal(;
    μ_range = range(-3.0, stop = 3.0, length = 200),
    σ_range = range(1e-4, stop = 1.0, length = 200),
    α = 0.1,
    xlabel = L"\mu_X",
    ylabel = L"\sigma_X",
)
    function kl_divergence_of_moment_matched_messages(incoming_message::Normal; α=0.1)
        approx_outgoing_message = moment_match_relu(incoming_message, α=α)
        μ_incoming, σ_incoming = mean(incoming_message), std(incoming_message)
        σ_outgoing = std(approx_outgoing_message)

        z = μ_incoming / σ_incoming
        q = normccdf(z)  # P(X<0) stably

        # If α==1, KL should be exactly 0 (up to rounding)
        if α == 1
            return 0.0
        end

        # Main closed-form KL
        KL = 0.5 * log((σ_outgoing^2) / (σ_incoming^2)) - q * log(α)
        KL = max(KL, 0.0)  # optional: enforce non-negativity numerically

        return KL
    end

    KL_values = zeros(length(σ_range), length(μ_range))
    for (i, μ) in enumerate(μ_range)
        for (j, σ) in enumerate(σ_range)
            KL_values[j, i] = kl_divergence_of_moment_matched_messages(Normal(μ, σ); α=α)
        end
    end

    plt = heatmap(
        μ_range,
        σ_range,
        KL_values,
        xlabel = xlabel,
        ylabel = ylabel,
        zlabel = L"\mathrm{KL}\left(\mathrm{ReLU}_\alpha(\cdot;\mu_X, \sigma_X^2) \;||\; \mathcal{N}(\cdot)\right)",
        xticks = range(minimum(μ_range), stop = maximum(μ_range), step = 2.0),
        yticks = range(minimum(σ_range), stop = maximum(σ_range), step = 0.3),
        xtickfontsize = 12,
        ytickfontsize = 12,
        xguidefontsize = 14,
        yguidefontsize = 14,
        zguidefontsize = 14,
        color = :viridis,
        colorbar_formatter = :scientific,
    )
    contour!(
        μ_range,
        σ_range,
        KL_values,
        levels = 2.0 .^ (-5:2),
        linewidth = 1.5,
        linecolor = :white,
        label = false,
    )
    display(plt)
end

"""
    main()

Main entry point demonstrating direct message approximation (DMA) analysis with comprehensive visualizations.

This function runs a complete suite of demonstrations showing:

1. **Good Approximation Scenario**:
   - Priors: X ~ N(2.0, 0.5), Y ~ N(3.0, 0.5), Z ~ N(6.0, 1.0)
   - Product constraint: Z = X × Y
   - DMA with moment-matched LogNormal approximation
   - Shows accurate message and marginal approximations

2. **Bad Approximation Scenario**:
   - Priors: X ~ N(2.0, 1.0), Y ~ N(3.0, 1.0), Z ~ N(10.0, 1.0)
   - Deliberately poor message: N(-6.0, 1.0) to Y
   - Demonstrates failure mode when approximation is far from truth
   - Highlights importance of accurate message approximation

3. **KL Divergence Analysis**:
   - KL(Normal || LogNormal): Shows when Gaussian-to-LogNormal approximation fails
   - KL(LogNormal || Normal): Shows when LogNormal-to-Gaussian approximation fails
   - KL for ReLU factors with α=0.1 and α=10.0: Quantifies ReLU approximation error

# Generated Output Files
All files are saved to ~/Downloads/:

**Message and Marginal Comparisons:**
- `good_division_approximation_messages.svg` - Exact vs approximate messages (good case)
- `good_division_approximation_marginals.svg` - Exact vs approximate marginals (good case)
- `bad_division_approximation_messages.svg` - Exact vs approximate messages (bad case)
- `bad_division_approximation_marginals.svg` - Exact vs approximate marginals (bad case)

**KL Divergence Heatmaps:**
- `kl_normal_lognormal.svg` - KL(N || LN) as function of (μ, σ)
- `kl_lognormal_normal.svg` - KL(LN || N) as function of (μ, σ)
- `kl_relu_normal_0.1.svg` - KL for leaky ReLU with α=0.1
- `kl_relu_normal_10.0.svg` - KL for inverse leaky ReLU with α=10.0

# Pedagogical Value
These demonstrations illustrate:
- The accuracy of DMA for product factors in factor graphs
- When Gaussian approximations work well vs. when they fail
- The relationship between distribution parameters and approximation quality
- How KL divergence quantifies approximation error
- The importance of moment matching for non-linear transformations

# Usage
```julia
# Run all demonstrations
julia> include("dma.jl")
julia> DMA.main()

# Or call specific functions
julia> using .DMA
julia> DMA.plot_α_divergences(
    prior_X = Normal(1.0, 0.5),
    prior_Y = Normal(2.0, 0.5),
    prior_Z = Normal(2.0, 1.0)
)
```

# Performance Note
The KL divergence computations involve numerical integration (quadgk) over infinite
ranges, which may take several seconds to complete for fine-grained grids.
"""
function main()
    println("\n" * "="^80)
    println("Direct Message Approximation (DMA) - KL Divergence Analysis")
    println("="^80)
    
    # Scenario 1: Good approximation with DMA
    println("\n📊 SCENARIO 1: Good Approximation with DMA")
    println("-"^80)
    println("Setting up factor graph with product constraint Z = X × Y")
    println("  • Prior X ~ N(2.0, 0.5)")
    println("  • Prior Y ~ N(3.0, 0.5)")
    println("  • Prior Z ~ N(6.0, 1.0)  [Note: 6.0 ≈ 2.0 × 3.0]")
    println("\nComputing messages using DMA with moment-matched LogNormal approximation...")
    
    plot_α_divergences(
        prior_X = Normal(2.0, sqrt(0.5)),
        prior_Y = Normal(3.0, sqrt(0.5)),
        prior_Z = Normal(6.0, sqrt(1.0)),
        approximate_msg_to_Y = nothing,
        base_path = "~/Downloads/good_division_approximation_",
    )
    println("✓ Saved: good_division_approximation_messages.svg")
    println("✓ Saved: good_division_approximation_marginals.svg")
    
    # Scenario 2: Poor approximation with deliberately wrong message
    println("\n📊 SCENARIO 2: Poor Approximation (Deliberately Wrong)")
    println("-"^80)
    println("Setting up factor graph with product constraint Z = X × Y")
    println("  • Prior X ~ N(2.0, 1.0)")
    println("  • Prior Y ~ N(3.0, 1.0)")
    println("  • Prior Z ~ N(10.0, 1.0)")
    println("  • Using deliberately poor approximation: msg_to_Y ~ N(-6.0, 1.0)")
    println("\nComputing messages with intentionally bad approximation...")
    
    plot_α_divergences(
        prior_X = Normal(2.0, sqrt(1.0)),
        prior_Y = Normal(3.0, sqrt(1.0)),
        prior_Z = Normal(10.0, sqrt(1.0)),
        approximate_msg_to_Y = Normal(-6.0, sqrt(1.0)),
        base_path = "~/Downloads/bad_division_approximation_",
    )
    println("✓ Saved: bad_division_approximation_messages.svg")
    println("✓ Saved: bad_division_approximation_marginals.svg")

    # KL Divergence Analyses
    println("\n📈 KL Divergence Analysis: Normal → LogNormal")
    println("-"^80)
    println("Computing KL(N(μ,σ²) || LN(m,s²)) with moment-matched LogNormal...")
    println("Sweeping μ ∈ [0.1, 10.0] and σ ∈ [0.1, 10.0]")
    
    plot_kl_divergence_normal_lognormal()
    savefig("~/Downloads/kl_normal_lognormal.svg")
    println("✓ Saved: kl_normal_lognormal.svg")

    println("\n📈 KL Divergence Analysis: LogNormal → Normal")
    println("-"^80)
    println("Computing KL(LN(μ,σ²) || N) with moment-matched Normal...")
    println("Sweeping μ ∈ [-3.0, 3.0] and σ ∈ [0.1, 1.0]")
    
    plot_kl_divergence_lognormal_normal()
    savefig("~/Downloads/kl_lognormal_normal.svg")
    println("✓ Saved: kl_lognormal_normal.svg")

    println("\n📈 KL Divergence Analysis: ReLU Factor (α = 0.1)")
    println("-"^80)
    println("Computing KL between exact ReLU output and Gaussian approximation...")
    println("Leaky ReLU with α = 0.1 (allows 10% leakage for negative values)")
    println("Sweeping μ ∈ [-3.0, 3.0] and σ ∈ [1e-4, 1.0]")
    
    plot_kl_divergence_ReLU_normal(α = 0.1, xlabel = L"\mu_X", ylabel = L"\sigma_X")
    savefig("~/Downloads/kl_relu_normal_0.1.svg")
    println("✓ Saved: kl_relu_normal_0.1.svg")

    println("\n📈 KL Divergence Analysis: ReLU Factor (α = 10.0)")
    println("-"^80)
    println("Computing KL between exact ReLU output and Gaussian approximation...")
    println("Inverse leaky ReLU with α = 10.0 (amplifies negative values)")
    println("Sweeping μ ∈ [-3.0, 3.0] and σ ∈ [1e-4, 1.0]")
    
    plot_kl_divergence_ReLU_normal(α = 10.0, xlabel = L"\mu_Y", ylabel = L"\sigma_Y")
    savefig("~/Downloads/kl_relu_normal_10.0.svg")
    println("✓ Saved: kl_relu_normal_10.0.svg")
    
    println("\n" * "="^80)
    println("✅ All demonstrations completed successfully!")
    println("="^80)
    println("\n📁 Output Location: ~/Downloads/")
    println("\nGenerated Files:")
    println("  Messages & Marginals:")
    println("    • good_division_approximation_{messages,marginals}.svg")
    println("    • bad_division_approximation_{messages,marginals}.svg")
    println("  KL Divergence Heatmaps:")
    println("    • kl_normal_lognormal.svg")
    println("    • kl_lognormal_normal.svg")
    println("    • kl_relu_normal_{0.1,10.0}.svg")
    println("\n" * "="^80 * "\n")
end

end