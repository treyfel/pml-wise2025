# Plots for approximate inference for a 1D Gaussian
#
# This module implements moment matching for truncated Gaussian distributions,
# demonstrating expectation propagation (EP) approximations. It visualizes how
# Gaussian approximations to truncated distributions work and computes the
# correction factors v(z) and w(z) for mean and variance adjustments.
#
# 2025 by Ralf Herbrich
# Hasso-Plattner Institute

module ApproximationPlots

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

"""
    DoublyTruncatedGaussian

Represents a Gaussian distribution truncated at both lower and upper bounds.

The probability density function is:
p(x) = N(x; μ, σ²) / Z  for lower < x < upper
p(x) = 0                otherwise

where Z = Φ((upper-μ)/σ) - Φ((lower-μ)/σ) is the normalization constant.

# Fields
- `lower::Float64`: Lower truncation bound
- `upper::Float64`: Upper truncation bound
- `normal::Normal`: Underlying Gaussian distribution before truncation

# Example
```julia
dt = DoublyTruncatedGaussian(-1.5, 1.0, Normal(0, 1))
# Creates N(0,1) truncated to interval [-1.5, 1.0]
```
"""
struct DoublyTruncatedGaussian
    lower::Float64
    upper::Float64
    normal::Normal
    
    # Constructor with validation
    function DoublyTruncatedGaussian(lower::Float64, upper::Float64, normal::Normal)
        @assert lower < upper "Lower bound must be less than upper bound"
        new(lower, upper, normal)
    end
end

"""
    plot_approximation(μ=0, σ=1)

Generates a plot comparing the true and approximate posterior for a 1D Gaussian
truncated at zero (i.e., restricted to x > 0).

This demonstrates expectation propagation (EP) approximation:
1. Prior: N(μ, σ²)
2. Likelihood (factor): I(x > 0)
3. True posterior: N(μ, σ²) truncated at 0
4. EP approximation: Gaussian moment matching to the truncated distribution

The plot shows:
- Green: Prior distribution
- Red solid: True likelihood (indicator function)
- Black solid: True posterior (normalized truncated Gaussian)
- Black dashed: Approximate posterior (Gaussian with matched moments)
- Red dashed: Approximate likelihood (message from factor to variable)

# Arguments
- `μ::Float64`: Mean of the prior Gaussian (default: 0)
- `σ::Float64`: Standard deviation of the prior Gaussian (default: 1)

# Mathematical Details
The approximation uses:
- v(z) = φ(z)/Φ(z) where z = μ/σ (additive correction for mean)
- μ_new = μ + σ·v(z)
- σ²_new = σ²(1 - v(z)(v(z) + z))

# Example
```julia
plot_approximation(0, 0.8)  # Prior N(0, 0.64) with x > 0 constraint
savefig("approximation.svg")
```
"""
function plot_approximation(μ=0, σ=1)
    # Helper function to add a plot line
    function plot_function(f; color=:blue, style=:solid)
        plot!(xs, f, linewidth=3, color=color, style=style)
    end

    # Compute the mean and variance of the best Gaussian approximation
    # using moment matching for a truncated Gaussian
    z = μ / σ  # Standardized truncation point
    v = pdf(Normal(), z) / cdf(Normal(), z)  # Correction factor v(z)
    μ_approx_posterior = μ + σ * v
    σ_approx_posterior = sqrt(σ^2 * (1 - (v * (v + z))))

    # Convert to natural parameters (precision form)
    τ_prior = μ / (σ^2)
    ρ_prior = 1 / (σ^2)
    τ_approx_posterior = μ_approx_posterior / (σ_approx_posterior^2)
    ρ_approx_posterior = 1 / (σ_approx_posterior^2)
    
    # Compute approximate likelihood message (difference of natural parameters)
    τ_approx_likelihood = τ_approx_posterior - τ_prior
    ρ_approx_likelihood = ρ_approx_posterior - ρ_prior
    μ_approx_likelihood = τ_approx_likelihood / ρ_approx_likelihood
    σ_approx_likelihood = sqrt(1 / ρ_approx_likelihood)

    println("Variance Analysis:")
    println("  σ² (prior)     = ", round(σ^2, digits=4))
    println("  σ² (likelihood) = ", round(σ_approx_likelihood^2, digits=4))
    println("  σ² (posterior)  = ", round(σ_approx_posterior^2, digits=4))

    # Create visualization
    d = Normal(μ, σ)
    xs = range(start=μ - 3 * σ, stop=μ + 3 * σ, length=1000)
    
    p = plot(
        legend=false,
        color=:blue,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
    )
    xlabel!(L"x")
    ylabel!(L"p(x)")

    # Plot the prior PDF
    plot_function(x -> pdf(d, x); color=:green)
    
    # Plot the factor function (true likelihood: indicator for x > 0)
    plot_function(x -> (x > 0) ? 1 : 0; color=:red)
    
    # Plot the true posterior PDF (truncated Gaussian)
    Z = 1 - cdf(d, 0)  # Normalization constant
    plot_function(x -> (x > 0) ? (1 / Z * pdf(d, x)) : 0; color=:black)

    # Plot the approximate posterior PDF (Gaussian approximation)
    plot_function(
        x -> pdf(Normal(μ_approx_posterior, σ_approx_posterior), x);
        color=:black,
        style=:dash,
    )
    
    # Plot the approximate likelihood message
    plot_function(
        x -> pdf(Normal(μ_approx_likelihood, σ_approx_likelihood), x);
        color=:red,
        style=:dash,
    )

    display(p)
end

"""
    plot_double_truncated(dt::DoublyTruncatedGaussian)

Visualizes the probability density function of a doubly truncated Gaussian distribution.

The plot shows:
- Blue solid line: Truncated distribution p(x)
- Black dashed line: Original Gaussian before truncation
- Red shaded area: Region of truncation (integration region)

# Arguments
- `dt::DoublyTruncatedGaussian`: The doubly truncated Gaussian to visualize

# Example
```julia
dt = DoublyTruncatedGaussian(-1.5, 1.0, Normal(0, 1))
plot_double_truncated(dt)
savefig("double_truncated.svg")
```
"""
function plot_double_truncated(dt::DoublyTruncatedGaussian)
    # PDF of the doubly truncated Gaussian
    function double_truncated_pdf(x)
        if x < dt.lower || x > dt.upper
            return 0.0
        else
            Z = cdf(dt.normal, dt.upper) - cdf(dt.normal, dt.lower)
            return pdf(dt.normal, x) / Z
        end
    end

    # Set plot range with some padding
    x_min = dt.lower - 0.5 * (dt.upper - dt.lower)
    x_max = dt.upper + 0.5 * (dt.upper - dt.lower)
    xs = range(start = x_min, stop = x_max, length = 1000)

    # Create filled region showing the truncation interval
    pts = [(dt.lower, 0.0)]
    for x in range(start = dt.lower, stop = dt.upper, length = 500)
        push!(pts, (x, pdf(dt.normal, x)))
    end
    for x in range(start = dt.upper, stop = dt.lower, length = 500)
        push!(pts, (x, 0.0))
    end
    push!(pts, (dt.lower, 0.0))

    p = plot(
        legend = false,
        xlabel = L"x",
        ylabel = L"p_X(x)",
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    # Shaded region under the curve
    plot!(Shape(pts), fillcolor = :red, fillalpha = 0.2, linewidth = 0.5)
    # Original Gaussian (before truncation)
    plot!(xs, x -> pdf(dt.normal, x), linewidth = 1, color = :black, style=:dash)
    # Truncated distribution
    plot!(xs, x -> double_truncated_pdf(x), linewidth = 3, color = :blue)
    display(p)
end

"""
    plot_v(; lower=0, upper=100)

Plots the additive correction factor v(z) for the mean of a doubly-truncated Gaussian.

The correction factor is:
v(z) = (φ(a-z) - φ(b-z)) / (Φ(b-z) - Φ(a-z))

where:
- z = μ/σ (standardized mean)
- a = lower bound (standardized)
- b = upper bound (standardized)
- φ is the standard normal PDF
- Φ is the standard normal CDF

The updated mean is: μ_new = μ + σ·v(z)

# Arguments
- `lower::Float64`: Lower truncation bound (default: 0)
- `upper::Float64`: Upper truncation bound (default: 100)

# Visualization
- Blue line: v(z) function
- Red dashed lines: Truncation boundaries
- Red shaded area: Valid region between bounds

# Example
```julia
plot_v(lower = -3, upper = 3)  # Symmetric truncation
savefig("v_3_3.svg")
```
"""
function plot_v(; lower=0, upper=100)
    function v(z)
        d = Normal(0, 1)
        numerator = pdf(d, lower - z) - pdf(d, upper - z)
        denominator = cdf(d, upper - z) - cdf(d, lower - z)
        return numerator / denominator
    end

    zs = range(start = -6, stop = 6, length = 1000)
    vs = v.(zs)
    l = max(lower, minimum(zs))
    u = min(upper, maximum(zs))

    p = plot(
        legend = false,
        xlabel = L"z",
        ylabel = L"v(z)",
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    # Vertical lines at truncation bounds
    plot!([l, l], [minimum(vs), maximum(vs)], linewidth = 1, color = :red, style = :dash)
    plot!([u, u], [minimum(vs), maximum(vs)], linewidth = 1, color = :red, style = :dash)
    # Shaded region between bounds
    plot!(Shape([(l, minimum(vs)), (l, maximum(vs)), (u, maximum(vs)), (u, minimum(vs))]), 
          fillcolor = :red, fillalpha = 0.2, linewidth = 0)
    # Main function
    plot!(zs, vs, linewidth = 5, color = :blue)
    display(p)
end

"""
    plot_w(; lower=0, upper=100)

Plots the multiplicative correction factor w(z) for the variance of a doubly-truncated Gaussian.

The correction factor is:
w(z) = ((z+b)φ(b-z) - (z+a)φ(a-z)) / (Φ(b-z) - Φ(a-z)) + v(z)(2z + v(z))

where:
- z = μ/σ (standardized mean)
- a = lower bound (standardized)
- b = upper bound (standardized)
- v(z) is the mean correction factor
- φ is the standard normal PDF
- Φ is the standard normal CDF

The updated variance is: σ²_new = σ²(1 - w(z))

# Arguments
- `lower::Float64`: Lower truncation bound (default: 0)
- `upper::Float64`: Upper truncation bound (default: 100)

# Visualization
- Blue line: w(z) function
- Red dashed lines: Truncation boundaries
- Red shaded area: Valid region between bounds

# Example
```julia
plot_w(lower = -5, upper = 0)  # Left-truncated
savefig("w_5_0.svg")
```
"""
function plot_w(; lower=0, upper=100)
    function v(z)
        d = Normal(0, 1)
        numerator = pdf(d, lower - z) - pdf(d, upper - z)
        denominator = cdf(d, upper - z) - cdf(d, lower - z)
        return numerator / denominator
    end

    function w(z)
        d = Normal(0, 1)
        denominator = cdf(d, upper - z) - cdf(d, lower - z)
        term1 = ((z + upper) * pdf(d, upper - z) - (z + lower) * pdf(d, lower - z)) / denominator
        term2 = v(z) * (2*z + v(z))
        return term1 + term2
    end

    zs = range(start = -6, stop = 6, length = 1000)
    ws = w.(zs)
    l = max(lower, minimum(zs))
    u = min(upper, maximum(zs))

    p = plot(
        legend = false,
        xlabel = L"z",
        ylabel = L"w(z)",
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    # Vertical lines at truncation bounds
    plot!([l, l], [minimum(ws), maximum(ws)], linewidth = 1, color = :red, style = :dash)
    plot!([u, u], [minimum(ws), maximum(ws)], linewidth = 1, color = :red, style = :dash)
    # Shaded region between bounds
    plot!(Shape([(l, minimum(ws)), (l, maximum(ws)), (u, maximum(ws)), (u, minimum(ws))]), 
          fillcolor = :red, fillalpha = 0.2, linewidth = 0)
    # Main function
    plot!(zs, ws, linewidth = 5, color = :blue)
    display(p)
end

"""
    main()

Main demonstration function that generates all visualization plots for approximate
inference with truncated Gaussians.

Generates:
1. EP approximation plot for N(0, 0.8²) truncated at x > 0
2. Doubly truncated Gaussian visualization
3. Mean correction factor v(z) plots for various truncation scenarios:
   - Left truncation: (-∞, 0]
   - Symmetric truncation: [-3, 3]
   - Right truncation: [0, ∞)
4. Variance correction factor w(z) plots for the same scenarios

All plots are saved as SVG files to ~/Downloads/
"""
function main()
    # EP approximation example
    plot_approximation(0, 0.8)
    savefig("~/Downloads/approximation.svg")

    # Doubly truncated Gaussian example
    plot_double_truncated(DoublyTruncatedGaussian(-1.5, 1.0, Normal(0, 1)))
    savefig("~/Downloads/double_truncated.svg")

    # Mean correction factor v(z) for different truncation scenarios
    plot_v(lower = -100, upper = 0)  # Effective left-tail truncation
    savefig("~/Downloads/v_100_0.svg")
    plot_v(lower = -5, upper = 0)
    savefig("~/Downloads/v_5_0.svg")
    plot_v(lower = -3, upper = 3)  # Symmetric truncation
    savefig("~/Downloads/v_3_3.svg")
    plot_v(lower = 0, upper = 100)  # Effective right-tail truncation
    savefig("~/Downloads/v_0_100.svg")

    # Variance correction factor w(z) for different truncation scenarios
    plot_w(lower = -100, upper = 0)
    savefig("~/Downloads/w_100_0.svg")
    plot_w(lower = -5, upper = 0)
    savefig("~/Downloads/w_5_0.svg")
    plot_w(lower = -3, upper = 3)
    savefig("~/Downloads/w_3_3.svg")
    plot_w(lower = 0, upper = 100)
    savefig("~/Downloads/w_0_100.svg")
end

end