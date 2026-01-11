"""
    ReLUFactorModule

Module for understanding and working with ReLU (Rectified Linear Unit) factors in probabilistic graphical models.

This module provides functions to:
- Compute moment-matched Gaussian approximations for ReLU factors
- Visualize ReLU functions and their smooth approximations (including GELU)
- Compare exact vs. approximate message passing through ReLU factors
- Demonstrate message approximations for different parameter configurations

The ReLU factor represents the constraint y = ReLU_α(x) = max(αx, x), where α ∈ [0,1] is a 
leakage parameter. When α = 0, this is the standard ReLU. When α > 0, it's a leaky ReLU that
allows small negative values to pass through.

Key concepts:
- ReLU is a non-linear transformation commonly used in neural networks
- Message passing through ReLU factors requires approximation since the output is non-Gaussian
- Moment matching is used to approximate the output distribution with a Gaussian
- Smooth approximations like softplus and GELU can approximate ReLU behavior

2025 by Ralf Herbrich
Hasso-Plattner Institute
"""
module ReLUFactorModule

using Distributions
using Plots
using Random
using Statistics
using StatsFuns
using LinearAlgebra
using LaTeXStrings

"""
    moment_match_relu(d; α=0.1)

Compute a moment-matched Gaussian approximation for the output of a leaky ReLU factor.

Given an input distribution `d` (typically Gaussian) and a leaky ReLU transformation
y = ReLU_α(x) = max(αx, x), this function computes a Gaussian approximation of the
output distribution by matching the first two moments (mean and variance).

The approximation uses the following formulas:
- P = Φ(μ_x/σ_x) where Φ is the standard normal CDF
- ϕ = φ(μ_x/σ_x) where φ is the standard normal PDF
- A = α + (1-α)P
- B = α² + (1-α²)P
- μ_y = μ_x·A + σ_x·ϕ·(1-α)
- σ²_y = (μ_x² + σ_x²)·B + μ_x·σ_x·ϕ·(1-α²) - (μ_x·A + σ_x·ϕ·(1-α))²

This approximation is exact when α = 1, and provides a good approximation
for intermediate values.

# Arguments
- `d::Distribution`: Input distribution (typically Normal), representing the belief about x
- `α::Float64=0.1`: Leakage parameter for leaky ReLU (0 ≤ α ≤ 1)
  - α = 0: Standard ReLU (returns max(0, x))
  - α = 1: Identity function (returns x)
  - 0 < α < 1: Leaky ReLU (allows αx for x < 0)

# Returns
- `Normal{Float64}`: Gaussian approximation of the output distribution y = ReLU_α(x)

# Example
```julia
# Standard ReLU approximation
input_dist = Normal(1.0, 2.0)
output_approx = moment_match_relu(input_dist, α=0.0)

# Leaky ReLU approximation
leaky_output = moment_match_relu(input_dist, α=0.1)
```
"""
function moment_match_relu(d; α=0.1)
        μ_x = mean(d)
        σ_x = sqrt(var(d))

        P = cdf(Normal(), μ_x/σ_x)
        ϕ = pdf(Normal(), μ_x/σ_x)
        A = α + (1.0 - α) * P
        B = α^2 + (1.0 - α^2) * P
        μ_y = μ_x * A + σ_x * ϕ * (1.0 - α)
        σ2_y = (μ_x^2 + σ_x^2) * B + μ_x * σ_x * ϕ * (1.0 - α^2) - (μ_x * A + σ_x * ϕ * (1.0 - α))^2

    return Normal(μ_y, sqrt(σ2_y))
end

"""
    plot_approximate_relu(; x_range=range(-5, 5, length=300), σ2=1, α=0.1)

Visualize the mean and uncertainty of the moment-matched ReLU approximation as a function of input mean.

This function creates a plot showing how the ReLU function is approximated when the input
has uncertainty. For each value μ_x along the x-axis (with fixed variance σ²), it:
1. Computes the moment-matched Gaussian approximation of y = ReLU_α(x)
2. Plots the mean E[y] and ± one standard deviation as a ribbon

The plot shows:
- Black line: The deterministic ReLU_α function (for reference)
- Blue line with ribbon: Mean ± std of the approximated output distribution

The ribbon illustrates how uncertainty in the input propagates through the ReLU,
with maximum uncertainty near x = 0 where the ReLU transitions.

# Arguments
- `x_range::AbstractRange=range(-5, 5, length=300)`: Range of input means to evaluate
- `σ2::Float64=1`: Fixed variance of the input distribution
- `α::Float64=0.1`: Leakage parameter for the leaky ReLU

# Returns
Nothing. Displays the plot.

# Example
```julia
# Visualize ReLU with small input uncertainty
plot_approximate_relu(σ2=0.5, α=0.0)

# Visualize leaky ReLU with larger uncertainty
plot_approximate_relu(σ2=2.0, α=0.1, x_range=range(-10, 10, length=500))
```
"""
function plot_approximate_relu(; x_range=range(-5, 5, length=300), σ2=1, α=0.1)
    relu(x) = (x < 0.0) ? α * x : x

    outgoing_message(x_val) = moment_match_relu(Normal(x_val, sqrt(σ2)), α=α)

    preds = outgoing_message.(x_range)

    plt = plot(
        x_range,
        relu,
        linewidth=2,
        color = :black,
        label = false,
        xlabel = L"\mu_x",
        ylabel = L"y = \mathrm{ReLU}_\alpha(x)",
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    plot!(
        x_range,
        [mean(pred) for pred in preds],
        ribbon = [std(pred) for pred in preds],
        color = :blue,
        linewidth=2,
        label = false,
    )
    display(plt)
end

"""
    plot_real_and_approximate_message(; dx=Normal(1,1), y_range=range(-3, 5, length=300), α=0.1, xlabel=L"y", ylabel=L"m_{f \to Y}(y)")

Compare the exact and approximate outgoing messages from a leaky ReLU factor.

This function visualizes the difference between:
1. The exact (analytical) message from a leaky ReLU factor f(x,y) = δ(y - ReLU_α(x))
2. The moment-matched Gaussian approximation of that message

For a Gaussian input message to x, the exact output distribution is:
- For y < 0: A scaled Gaussian (compressed by factor 1/α)
- For y ≥ 0: The original Gaussian

The moment-matched approximation replaces this bi-modal or truncated distribution
with a single Gaussian, which may introduce approximation error.

The plot shows:
- Blue line: Exact message distribution (analytical)
- Red line: Moment-matched Gaussian approximation

The quality of the approximation depends on the input mean, variance, and leakage parameter α.

# Arguments
- `dx::Normal=Normal(1,1)`: Input Gaussian message/belief about x
- `y_range::AbstractRange=range(-3, 5, length=300)`: Range of output values to plot
- `α::Float64=0.1`: Leakage parameter for the leaky ReLU
- `xlabel::LaTeXString=L"y"`: Label for x-axis
- `ylabel::LaTeXString=L"m_{f \to Y}(y)"`: Label for y-axis

# Returns
Nothing. Displays the plot.

# Example
```julia
# Compare messages with negative input mean
plot_real_and_approximate_message(
    dx=Normal(-2, 1), 
    y_range=range(-10, 5, length=1000),
    α=0.5
)

# Compare with positive input mean
plot_real_and_approximate_message(
    dx=Normal(2, 1),
    α=0.1
)
```
"""
function plot_real_and_approximate_message(; 
    dx = Normal(1,1), 
    y_range = range(-3, 5, length=300),
    α = 0.1,
    xlabel = L"y",
    ylabel = L"m_{f \to Y}(y)",
)
    function real_outgoing_message(y_val)
        return (y_val < 0.0) ? pdf(dx, y_val/α) / α : pdf(dx, y_val)
    end

    d_approx = moment_match_relu(dx, α=α)
    approximate_outgoing_message(y_val) = pdf(d_approx, y_val)

    plt = plot(
        y_range,
        real_outgoing_message,
        linewidth=3,
        color = :blue,
        label = false,
        xlabel = xlabel,
        ylabel = ylabel,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    plot!(
        y_range,
        approximate_outgoing_message,
        linewidth=3,
        color = :red,
        label = false,
    )
    display(plt)
end

"""
    plot_relu(; x_range=range(-7, 7, length=300))

Visualize and compare different activation functions related to ReLU.

This function creates a comprehensive plot comparing several activation functions:

1. **Softplus (red)**: log(1 + exp(x)) - A smooth approximation of ReLU
2. **Binary Sum (red scatter)**: Σ_{i=1}^N 1/(1 + exp(x - i + 0.5)) - Discrete approximation
3. **Standard ReLU (green)**: ReLU_0(x) = max(0, x)
4. **Leaky ReLU (blue)**: ReLU_0.1(x) = max(0.1x, x)
5. **GELU (black)**: x·Φ(x) - Gaussian Error Linear Unit, where Φ is the standard normal CDF

The plot illustrates:
- How softplus smoothly approximates ReLU
- How binary sums can approximate softplus
- The difference between standard and leaky ReLU
- How GELU provides a smooth, probabilistic alternative

These visualizations are useful for understanding activation functions in neural networks
and their probabilistic interpretations.

# Arguments
- `x_range::AbstractRange=range(-7, 7, length=300)`: Range of x values to plot

# Returns
Nothing. Displays the plot.

# Implementation Notes
- The binary sum with N=100 approximates softplus through a sum of sigmoids
- GELU is defined as x·Φ(x) where Φ is the cumulative distribution function of N(0,1)
- Softplus is the smooth approximation: softplus(x) = log(1 + exp(x))

# Example
```julia
# Default comparison plot
plot_relu()

# Extended range for better visualization
plot_relu(x_range=range(-10, 10, length=500))
```
"""
function plot_relu(;
    x_range = range(-7, 7, length=300)
)
    
    function binary_sum(x; N = 100)
        t = x .- (1:N) .+ 0.5
        return sum(1 ./(1 .+ exp.(-t)))
    end
    
    relu(x; α = 0.0) = (x < 0.0) ? α * x : x


    plt = plot(
        x_range,
        log.(1 .+ exp.(x_range)),
        linewidth=3,
        color = :red,
        label = false,
        xlabel = L"x",
        ylabel = L"\mathrm{ReLU}_{\alpha}(x)",
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        legendfontsize = 12,
    )
    scatter!(
        x_range[1:20:end],
        binary_sum.(x_range[1:20:end], N=100),
        markersize=6,
        color = :red,
        label = L"\sum_{n=0}^{N-1} (1 + \exp(x - i))^{-1}",
    )
    plot!(
        x_range,
        relu.(x_range),
        linewidth=3,
        color = :green,
        label = L"\mathrm{ReLU}_0(x)",
    )
    plot!(
        x_range,
        x -> relu.(x, α=0.1),
        linewidth=3,
        color = :blue,
        label = L"\mathrm{ReLU}_{0.1}(x)",
    )
    plot!(
        x_range,
        x_range .* cdf.(Normal(0,1), x_range),
        linewidth=3,
        color = :black,
        label = L"\mathrm{GELU}(x)",
    )
    display(plt)
end

"""
    main()

Main entry point for the ReLUFactorModule demonstrations.

This function runs a comprehensive suite of visualizations demonstrating:

1. **Activation Function Comparison**: 
   - Plots ReLU, leaky ReLU, softplus, GELU, and binary sum approximations

2. **ReLU Approximation with Uncertainty**:
   - Shows how moment-matching approximates ReLU when inputs have uncertainty

3. **Message Approximation Quality**:
   - Compares exact vs. approximate messages for various configurations:
     - Different input means: μ_x ∈ {-2, 0, +2}
     - Different leakage parameters: α ∈ {0.5, 2.0}
   - Demonstrates both forward (to Y) and backward (to X) message passing

The demonstrations help understand:
- How well Gaussian approximations work for ReLU factors
- The impact of input distribution parameters on approximation quality
- The difference between leaky (α = 0.5) and inverse leaky (α = 2.0) ReLU

# Output Files
Creates multiple visualization files in ~/Downloads/:
- relu_plot.svg: Comparison of activation functions
- relu_approximation.svg: ReLU approximation with uncertainty
- relu_message_approximation_{mean}_{alpha}.svg: Message approximations
  - For means: -2, 0, +2
  - For alphas: 0.5, 2

# Example
```julia
using .ReLUFactorModule
ReLUFactorModule.main()
```
"""
function main()
    println("\n" * "="^80)
    println("ReLU Factor Approximation - Comprehensive Analysis")
    println("="^80)
    
    # Plot activation functions
    println("\n📊 Activation Functions Comparison")
    println("-"^80)
    println("Comparing different activation functions over x ∈ [-7, 7]:")
    println("  • Softplus: log(1 + exp(x)) - smooth ReLU approximation (red)")
    println("  • Binary Sum: Σ 1/(1+exp(x-i+0.5)) - discrete approximation (red scatter)")
    println("  • Standard ReLU: max(0, x) (green)")
    println("  • Leaky ReLU: max(0.1x, x) with α=0.1 (blue)")
    println("  • GELU: x·Φ(x) - Gaussian Error Linear Unit (black)")
    plot_relu(x_range = range(-7, 7, length=300))
    savefig("~/Downloads/relu_plot.svg")
    println("✓ Saved: relu_plot.svg")

    # ReLU approximation with high uncertainty
    println("\n📊 ReLU Approximation with Uncertainty (High Variance)")
    println("-"^80)
    println("Moment-matched Gaussian approximation of y = ReLU_0.1(x)")
    println("  • Input variance: σ² = 1.0  [σ = 1.0]")
    println("  • Leakage parameter: α = 0.1")
    println("  • Range: μ_x ∈ [-12, 3]")
    println("  • Black line: Deterministic ReLU_α function")
    println("  • Blue ribbon: Mean ± std of approximated output")
    plot_approximate_relu(σ2 = 1, α = 0.1, x_range = range(-12, 3, length=300))
    savefig("~/Downloads/relu_approximation_σ=1.svg")
    println("✓ Saved: relu_approximation_σ=1.svg")

    # ReLU approximation with low uncertainty
    println("\n📊 ReLU Approximation with Uncertainty (Low Variance)")
    println("-"^80)
    println("Moment-matched Gaussian approximation of y = ReLU_0.1(x)")
    println("  • Input variance: σ² = 0.1  [σ ≈ 0.316]")
    println("  • Leakage parameter: α = 0.1")
    println("  • Range: μ_x ∈ [-12, 3]")
    println("  • Narrower ribbon shows less uncertainty propagation")
    plot_approximate_relu(σ2 = 0.1, α = 0.1, x_range = range(-12, 3, length=300))
    savefig("~/Downloads/relu_approximation_σ=0.316.svg")
    println("✓ Saved: relu_approximation_σ=0.316.svg")

    # Message approximations for different configurations
    println("\n📊 Message Approximation: Negative Input Mean (μ = -2)")
    println("-"^80)
    
    println("\n  Configuration 1: Leaky ReLU (α = 0.5)")
    println("    • Input: x ~ N(-2, 1)")
    println("    • Output: y = ReLU_0.5(x)")
    println("    • Blue: Exact message distribution")
    println("    • Red: Moment-matched Gaussian approximation")
    plot_real_and_approximate_message(dx = Normal(-2, 1), y_range = range(-10, 5, length=1000), α = 0.5, xlabel = L"y", ylabel = L"m_{f \to Y}(y)")
    savefig("~/Downloads/relu_message_approximation_-2_0.5.svg")
    println("    ✓ Saved: relu_message_approximation_-2_0.5.svg")
    
    println("\n  Configuration 2: Inverse Leaky ReLU (α = 2.0)")
    println("    • Input: x ~ N(-2, 1)")
    println("    • Output: x = ReLU_2.0^(-1)(y)")
    println("    • Blue: Exact message distribution")
    println("    • Red: Moment-matched Gaussian approximation")
    plot_real_and_approximate_message(dx = Normal(-2, 1), y_range = range(-10, 5, length=1000), α = 2, xlabel = L"x", ylabel = L"m_{f \to X}(x)")
    savefig("~/Downloads/relu_message_approximation_-2_2.svg")
    println("    ✓ Saved: relu_message_approximation_-2_2.svg")

    println("\n📊 Message Approximation: Zero Input Mean (μ = 0)")
    println("-"^80)
    
    println("\n  Configuration 3: Leaky ReLU (α = 0.5)")
    println("    • Input: x ~ N(0, 1)")
    println("    • Output: y = ReLU_0.5(x)")
    println("    • Maximum uncertainty at transition point")
    plot_real_and_approximate_message(dx = Normal(0, 1), y_range = range(-10, 5, length=1000), α = 0.5, xlabel = L"y", ylabel = L"m_{f \to Y}(y)")
    savefig("~/Downloads/relu_message_approximation_0_0.5.svg")
    println("    ✓ Saved: relu_message_approximation_0_0.5.svg")
    
    println("\n  Configuration 4: Inverse Leaky ReLU (α = 2.0)")
    println("    • Input: x ~ N(0, 1)")
    println("    • Output: x = ReLU_2.0^(-1)(y)")
    plot_real_and_approximate_message(dx = Normal(0, 1), y_range = range(-10, 5, length=1000), α = 2, xlabel = L"x", ylabel = L"m_{f \to X}(x)")
    savefig("~/Downloads/relu_message_approximation_0_2.svg")
    println("    ✓ Saved: relu_message_approximation_0_2.svg")

    println("\n📊 Message Approximation: Positive Input Mean (μ = +2)")
    println("-"^80)
    
    println("\n  Configuration 5: Leaky ReLU (α = 0.5)")
    println("    • Input: x ~ N(+2, 1)")
    println("    • Output: y = ReLU_0.5(x)")
    println("    • Most mass in positive region (best approximation)")
    plot_real_and_approximate_message(dx = Normal(+2, 1), y_range = range(-10, 5, length=1000), α = 0.5, xlabel = L"y", ylabel = L"m_{f \to Y}(y)")
    savefig("~/Downloads/relu_message_approximation_2_0.5.svg")
    println("    ✓ Saved: relu_message_approximation_2_0.5.svg")
    
    println("\n  Configuration 6: Inverse Leaky ReLU (α = 2.0)")
    println("    • Input: x ~ N(+2, 1)")
    println("    • Output: x = ReLU_2.0^(-1)(y)")
    plot_real_and_approximate_message(dx = Normal(+2, 1), y_range = range(-10, 5, length=1000), α = 2, xlabel = L"x", ylabel = L"m_{f \to X}(x)")
    savefig("~/Downloads/relu_message_approximation_2_2.svg")
    println("    ✓ Saved: relu_message_approximation_2_2.svg")

    println("\n" * "="^80)
    println("✅ All ReLU factor demonstrations completed successfully!")
    println("="^80)
    println("\n📁 Output Location: ~/Downloads/")
    println("\nGenerated Files:")
    println("  Activation Functions:")
    println("    • relu_plot.svg")
    println("  ReLU Approximations with Uncertainty:")
    println("    • relu_approximation_σ=1.svg (high variance)")
    println("    • relu_approximation_σ=0.316.svg (low variance)")
    println("  Message Approximations (6 configurations):")
    println("    • relu_message_approximation_{-2,0,+2}_{0.5,2}.svg")
    println("\n💡 Key Insights:")
    println("  • Approximation quality depends on input mean relative to zero")
    println("  • μ < 0: Most mass gets scaled by α (higher error)")
    println("  • μ = 0: Maximum uncertainty at transition (moderate error)")
    println("  • μ > 0: Most mass passes through unchanged (lower error)")
    println("  • α = 0.5: Leaky ReLU (scales negative values down)")
    println("  • α = 2.0: Inverse leaky (amplifies negative values)")
    println("\n" * "="^80 * "\n")
end

export moment_match_relu

end