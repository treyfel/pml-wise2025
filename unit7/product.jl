"""
    ProductFactorModule

Module for understanding and working with approximate product factors in probabilistic graphical models.

This module provides functions to:
- Sample messages from product factors (z = x * y) to all connected variables
- Compute approximate Gaussian messages for product factors
- Visualize and compare exact (sampled) vs. approximate message distributions
- Compare different approximation strategies (log-normal vs. Matchbox)

The module demonstrates message passing in factor graphs with non-linear product constraints,
and the accuracy of Gaussian approximations for these operations. It includes implementations
of both the log-normal approximation and the Matchbox approximation from the 2009 paper.

2025 by Ralf Herbrich
Hasso-Plattner Institute
"""
module ProductFactorModule

using Distributions
using Plots
using Random
using Statistics
using LinearAlgebra
using LaTeXStrings

""" 
    sample_product_factor_msg_to_z(msg_from_x::Normal{Float64}, msg_from_y::Normal{Float64}; n = 100000)

Sample the message of a product factor f(x,y,z) = δ(z - x*y) to variable z.

This function generates samples from the distribution of z = x * y when x and y are
Gaussian random variables. The resulting distribution is typically not Gaussian,
and this sampling provides a ground truth for evaluating Gaussian approximations.

# Arguments
- `msg_from_x::Normal{Float64}`: Incoming Gaussian message/belief about x
- `msg_from_y::Normal{Float64}`: Incoming Gaussian message/belief about y
- `n::Int=100000`: Number of samples to generate

# Returns
- `Vector{Float64}`: Array of n samples from the distribution of z = x * y

# Example
```julia
msg_x = Normal(3.0, 0.5)
msg_y = Normal(5.0, 0.5)
z_samples = sample_product_factor_msg_to_z(msg_x, msg_y, n=10000)
```
"""
function sample_product_factor_msg_to_z(msg_from_x::Normal{Float64}, msg_from_y::Normal{Float64}; n = 100000)
    x_samples = rand(msg_from_x, n)
    y_samples = rand(msg_from_y, n)

    return x_samples .* y_samples
end

""" 
    sample_product_factor_msg_to_x(msg_from_z::Normal{Float64}, msg_from_y::Normal{Float64}; n = 100000)

Sample the message of a product factor f(x,y,z) = δ(z - x*y) to variable x.

This function generates samples from the distribution of x = z / y when z and y are
Gaussian random variables. The division can produce heavy-tailed distributions,
especially when y is close to zero. This sampling provides ground truth for evaluating
Gaussian approximations of the division operation.

# Arguments
- `msg_from_z::Normal{Float64}`: Incoming Gaussian message/belief about z
- `msg_from_y::Normal{Float64}`: Incoming Gaussian message/belief about y
- `n::Int=100000`: Number of samples to generate

# Returns
- `Vector{Float64}`: Array of n samples from the distribution of x = z / y

# Example
```julia
msg_z = Normal(10.0, 1.0)
msg_y = Normal(5.0, 0.5)
x_samples = sample_product_factor_msg_to_x(msg_z, msg_y, n=10000)
```
"""
function sample_product_factor_msg_to_x(msg_from_z::Normal{Float64}, msg_from_y::Normal{Float64}; n = 100000)
    z_samples = rand(msg_from_z, n)
    y_samples = rand(msg_from_y, n)

    return z_samples ./ y_samples
end

""" 
    sample_product_factor_msg_to_y(msg_from_z::Normal{Float64}, msg_from_x::Normal{Float64}; n = 100000)

Sample the message of a product factor f(x,y,z) = δ(z - x*y) to variable y.

This function generates samples from the distribution of y = z / x when z and x are
Gaussian random variables. It is functionally equivalent to `sample_product_factor_msg_to_x`
with swapped arguments, as the product factor is symmetric with respect to x and y.

# Arguments
- `msg_from_z::Normal{Float64}`: Incoming Gaussian message/belief about z
- `msg_from_x::Normal{Float64}`: Incoming Gaussian message/belief about x
- `n::Int=100000`: Number of samples to generate

# Returns
- `Vector{Float64}`: Array of n samples from the distribution of y = z / x

# Example
```julia
msg_z = Normal(10.0, 1.0)
msg_x = Normal(3.0, 0.5)
y_samples = sample_product_factor_msg_to_y(msg_z, msg_x, n=10000)
```
"""
sample_product_factor_msg_to_y(msg_from_z::Normal{Float64}, msg_from_x::Normal{Float64}; n = 100000) = sample_product_factor_msg_to_x(msg_from_z, msg_from_x, n=n)

"""
    approximate_product_factor_msg_to_z(msg_from_x::Normal{Float64}, msg_from_y::Normal{Float64}; matchbox = false)

Compute an approximate Gaussian message from a product factor f(x,y,z) = δ(z - x*y) to variable z.

This function approximates the distribution of z = x * y (where x and y are Gaussian)
with a Gaussian distribution. Two approximation methods are supported:

1. Log-normal approximation (default): Uses the moment-matching approach based on
   log-normal distribution theory. This gives:
   - μ_z = μ_x * μ_y
   - σ²_z = μ_x² * σ_y² + μ_y² * σ_x² + σ_x² * σ_y²

2. Matchbox approximation: Based on the 2009 Matchbox paper, using:
   - μ_z = μ_x * μ_y
   - σ²_z = (μ_x² + σ_x²) * (μ_y² + σ_y²) - μ_z²

# Arguments
- `msg_from_x::Normal{Float64}`: Incoming Gaussian message/belief about x
- `msg_from_y::Normal{Float64}`: Incoming Gaussian message/belief about y
- `matchbox::Bool=false`: If true, use Matchbox approximation; otherwise use log-normal

# Returns
- `Normal{Float64}`: Approximate Gaussian distribution for z

# Example
```julia
msg_x = Normal(3.0, 0.5)
msg_y = Normal(5.0, 0.5)
# Log-normal approximation
approx_z_ln = approximate_product_factor_msg_to_z(msg_x, msg_y)
# Matchbox approximation
approx_z_mb = approximate_product_factor_msg_to_z(msg_x, msg_y, matchbox=true)
```
"""
function approximate_product_factor_msg_to_z(msg_from_x::Normal{Float64}, msg_from_y::Normal{Float64}; matchbox = false)
    if matchbox
        # Matchbox approximation
        μ = msg_from_x.μ * msg_from_y.μ
        σ2 = (msg_from_x.μ^2 + msg_from_x.σ^2) * (msg_from_y.μ^2 + msg_from_y.σ^2) - μ^2
        return Normal(μ, sqrt(σ2))
    else
        # Log-normal approximation
        μ = msg_from_x.μ * msg_from_y.μ
        σ2 = msg_from_x.μ^2 * msg_from_y.σ^2 + msg_from_y.μ^2 * msg_from_x.σ^2 + msg_from_x.σ^2 * msg_from_y.σ^2
        return Normal(μ, sqrt(σ2))
    end
end

"""
    approximate_product_factor_msg_to_x(msg_from_z::Normal{Float64}, msg_from_y::Normal{Float64}; matchbox = false)

Compute an approximate Gaussian message from a product factor f(x,y,z) = δ(z - x*y) to variable x.

This function approximates the distribution of x = z / y (where z and y are Gaussian)
with a Gaussian distribution. Two approximation methods are supported:

1. Log-normal approximation (default): Uses moment-matching with correction factor
   c = 1 + σ_y²/μ_y². The formulas are:
   - μ_x = (μ_z / μ_y) * c
   - σ²_x = c² * (σ_z²/μ_y² + σ_y²*μ_z²/μ_y⁴ + σ_z²*σ_y²/μ_y⁴)
   Note: Throws ArgumentError if μ_y = 0

2. Matchbox approximation: Based on the 2009 Matchbox paper, with c = μ_y² + σ_y²:
   - μ_x = μ_z * μ_y / c
   - σ²_x = σ_z² / c²

# Arguments
- `msg_from_z::Normal{Float64}`: Incoming Gaussian message/belief about z
- `msg_from_y::Normal{Float64}`: Incoming Gaussian message/belief about y
- `matchbox::Bool=false`: If true, use Matchbox approximation; otherwise use log-normal

# Returns
- `Normal{Float64}`: Approximate Gaussian distribution for x

# Throws
- `ArgumentError`: If μ_y = 0 and using log-normal approximation (division by zero)

# Example
```julia
msg_z = Normal(10.0, 1.0)
msg_y = Normal(5.0, 0.5)
# Log-normal approximation
approx_x_ln = approximate_product_factor_msg_to_x(msg_z, msg_y)
# Matchbox approximation
approx_x_mb = approximate_product_factor_msg_to_x(msg_z, msg_y, matchbox=true)
```
"""
function approximate_product_factor_msg_to_x(msg_from_z::Normal{Float64}, msg_from_y::Normal{Float64}; matchbox = false)
    if matchbox
        # Matchbox approximation
        c = msg_from_y.μ^2 + msg_from_y.σ^2
        μ = msg_from_z.μ * msg_from_y.μ / c
        σ2 = msg_from_z.σ^2 / c^2
        return Normal(μ, sqrt(σ2))
    else
        # Log-normal approximation
        if (msg_from_y.μ == 0)
            throw(ArgumentError("Division by zero results in a uniform message."))
        else
            c = 1 + msg_from_y.σ^2 / msg_from_y.μ^2
            μ = msg_from_z.μ / msg_from_y.μ * c
            σ2 = c^2 * (msg_from_z.σ^2/msg_from_y.μ^2 + msg_from_y.σ^2/msg_from_y.μ^2 * msg_from_z.μ^2/msg_from_y.μ^2 + msg_from_z.σ^2/msg_from_y.μ^2 * msg_from_y.σ^2/msg_from_y.μ^2)
            return Normal(μ, sqrt(σ2))
        end
    end
end

"""
    approximate_product_factor_msg_to_y(msg_from_z::Normal{Float64}, msg_from_x::Normal{Float64}; matchbox = false)

Compute an approximate Gaussian message from a product factor f(x,y,z) = δ(z - x*y) to variable y.

This function approximates the distribution of y = z / x (where z and x are Gaussian)
with a Gaussian distribution. It is functionally equivalent to `approximate_product_factor_msg_to_x`
with swapped arguments, as the product factor is symmetric with respect to x and y.

See `approximate_product_factor_msg_to_x` for detailed documentation of the approximation methods.

# Arguments
- `msg_from_z::Normal{Float64}`: Incoming Gaussian message/belief about z
- `msg_from_x::Normal{Float64}`: Incoming Gaussian message/belief about x
- `matchbox::Bool=false`: If true, use Matchbox approximation; otherwise use log-normal

# Returns
- `Normal{Float64}`: Approximate Gaussian distribution for y

# Throws
- `ArgumentError`: If μ_x = 0 and using log-normal approximation (division by zero)

# Example
```julia
msg_z = Normal(10.0, 1.0)
msg_x = Normal(3.0, 0.5)
approx_y = approximate_product_factor_msg_to_y(msg_z, msg_x)
```
"""
approximate_product_factor_msg_to_y(msg_from_z::Normal{Float64}, msg_from_x::Normal{Float64}; matchbox = false) = approximate_product_factor_msg_to_x(msg_from_z, msg_from_x, matchbox = matchbox)

"""
    plot_product_factor(; x=Normal(3.0, 0.5), y=Normal(5.0, 0.5), z=Normal(10.0, 1.0), n=100000, α=0.01, base="~/Downloads/base_", with_matchbox=false)

Visualize and compare exact (sampled) messages vs. approximate Gaussian messages for a product factor.

This function generates histograms comparing sampled messages from a product factor
f(x,y,z) = δ(z - x*y) with their Gaussian approximations for all three variables.
It creates six plots (three for each variable) saved as both PDF and SVG:

1. Message to z: Compares samples of z = x * y with approximations
2. Message to x: Compares samples of x = z / y with approximations
3. Message to y: Compares samples of y = z / x with approximations

Each plot shows:
- Blue histogram: Sampled "true" message distribution
- Red line: Log-normal approximation
- Green line (optional): Matchbox approximation (if with_matchbox=true)

The visualization helps assess the quality of Gaussian approximations for non-linear
operations in factor graphs.

# Arguments
- `x::Normal{Float64}=Normal(3.0, 0.5)`: Gaussian message/prior for variable x
- `y::Normal{Float64}=Normal(5.0, 0.5)`: Gaussian message/prior for variable y
- `z::Normal{Float64}=Normal(10.0, 1.0)`: Gaussian message/prior for variable z
- `n::Int=100000`: Number of samples to generate for each distribution
- `α::Float64=0.01`: Quantile threshold for plot limits (uses [α, 1-α] range)
- `base::String="~/Downloads/base_"`: Base path for output files (appends variable name and extension)
- `with_matchbox::Bool=false`: If true, include Matchbox approximation in plots

# Returns
Nothing. Displays plots and saves them to files.

# Files Created
- `{base}product_factor_msg_to_z.pdf` and `.svg`: Message to z visualization
- `{base}product_factor_msg_to_x.pdf` and `.svg`: Message to x visualization
- `{base}product_factor_msg_to_y.pdf` and `.svg`: Message to y visualization

# Example
```julia
# Compare approximations for a simple case
plot_product_factor(x=Normal(3.0, 0.5), y=Normal(5.0, 0.5), n=100000)

# Include Matchbox approximation for comparison
plot_product_factor(
    x=Normal(-6.5, 1.0), 
    y=Normal(-8.5, 1.0), 
    with_matchbox=true,
    base="~/Downloads/negative_case_"
)
```
"""
function plot_product_factor(; x = Normal(3.0, 0.5), y = Normal(5.0, 0.5), z = Normal(10.0, 1.0), n = 100000, α=0.01, base = "~/Downloads/base_", with_matchbox = false)
    z_samples = sample_product_factor_msg_to_z(x, y, n = n)
    approx_msg_to_z = approximate_product_factor_msg_to_z(x, y, matchbox = false)
    approx2_msg_to_z = approximate_product_factor_msg_to_z(x, y, matchbox = true)
    z_min, z_max = quantile(z_samples, [α, 1-α])
    p = histogram(
            z_samples[z_samples .>= z_min .&& z_samples .<= z_max], 
            label = L"\mathrm{msg}_{f \to Z}",
            normalize = :pdf, 
            # title = "Product Factor Message to Z", 
            bins = 50,
            legend = :topright,
            xlabel = L"z",
            ylabel = L"p(z)",
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
            legendfontsize=12,
    )
    plot!(
        range(z_min, stop = z_max, length = 1000), 
        x -> pdf(approx_msg_to_z, x), 
        label = L"\widehat{\mathrm{msg}}_{f \to Z}", 
        linewidth = 3,
        color = :red,
    )
    if with_matchbox
        plot!(
            range(z_min, stop = z_max, length = 1000), 
            x -> pdf(approx2_msg_to_z, x), 
            label = L"\widehat{\mathrm{msg}}_{f \to Z}", 
            linewidth = 3,
            color = :green,
        )
    end
    xlims!(z_min, z_max)
    display(p)
    savefig(p, base * "product_factor_msg_to_z.pdf") 
    savefig(p, base * "product_factor_msg_to_z.svg")


    x_samples = sample_product_factor_msg_to_x(z, y, n = n)
    x_min, x_max = quantile(x_samples, [α, 1-α])
    approx_msg_to_x = approximate_product_factor_msg_to_x(z, y, matchbox = false)
    approx2_msg_to_x = approximate_product_factor_msg_to_x(z, y, matchbox = true)
    p = histogram(
            x_samples[x_samples .>= x_min .&& x_samples .<= x_max], 
            label = L"\mathrm{msg}_{f \to X}",
            normalize = :pdf, 
            bins = 50,
            xlabel = L"x",
            ylabel = L"p(x)",
            # title = "Product Factor Message to X", 
            legend = :topright,
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
            legendfontsize=12,
    )
    plot!(
        range(x_min, stop = x_max, length = 1000), 
        x -> pdf(approx_msg_to_x, x), 
        label = L"\widehat{\mathrm{msg}}_{f \to X}", 
        linewidth = 3,
        color = :red,
    )
    if with_matchbox
        plot!(
            range(x_min, stop = x_max, length = 1000), 
            x -> pdf(approx2_msg_to_x, x), 
            label = L"\widehat{\mathrm{msg}}_{f \to X}", 
            linewidth = 3,
            color = :green,
        )
    end
    xlims!(x_min, x_max)
    display(p)
    savefig(p, base * "product_factor_msg_to_x.pdf")
    savefig(p, base * "product_factor_msg_to_x.svg")

    y_samples = sample_product_factor_msg_to_y(z, x, n = n)
    approx_msg_to_y = approximate_product_factor_msg_to_y(z, x, matchbox = false)
    approx2_msg_to_y = approximate_product_factor_msg_to_y(z, x, matchbox = true)
    y_min, y_max = quantile(y_samples, [α, 1-α])
    p = histogram(
            y_samples[y_samples .>= y_min .&& y_samples .<= y_max], 
            label = L"\mathrm{msg}_{f \to Y}",
            normalize = :pdf, 
            bins = 50,
            xlabel = L"y",
            ylabel = L"p(y)",
            # title = "Product Factor Message to Y", 
            legend = :topright,
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
            legendfontsize=12,
    )
    plot!(
        range(y_min, stop = y_max, length = 1000), 
        x -> pdf(approx_msg_to_y, x), 
        label = L"\widehat{\mathrm{msg}}_{f \to Y}", 
        linewidth = 3,
        color = :red,
    )
    if with_matchbox
        plot!(
            range(y_min, stop = y_max, length = 1000), 
            x -> pdf(approx2_msg_to_y, x), 
            label = L"\widehat{\mathrm{msg}}_{f \to Y}", 
            linewidth = 3,
            color = :green,
        )
    end
    xlims!(y_min, y_max)
    display(p)
    savefig(p, base * "product_factor_msg_to_y.pdf")
    savefig(p, base * "product_factor_msg_to_y.svg")
end

"""
    plot_lognormal(; μ=0.0, σ=1.0)

Plot the probability density function of a log-normal distribution.

This function visualizes a log-normal distribution with the specified parameters.
A log-normal distribution arises when the logarithm of a random variable is normally
distributed. It's relevant for understanding the theoretical basis of the log-normal
approximation used in product factors.

The plot shows the PDF over a range from 0 to mean + 3*std, which captures the
main mass of the distribution.

# Arguments
- `μ::Float64=0.0`: Mean of the underlying normal distribution (not the log-normal mean)
- `σ::Float64=1.0`: Standard deviation of the underlying normal distribution

# Returns
Nothing. Displays the plot.

# Note
The parameters μ and σ are for the underlying normal distribution. The actual mean
and variance of the log-normal distribution are:
- Mean: exp(μ + σ²/2)
- Variance: [exp(σ²) - 1] * exp(2μ + σ²)

# Example
```julia
plot_lognormal(μ=0.0, σ=1.0)
savefig("~/Downloads/lognormal.svg")
```
"""
function plot_lognormal(; μ = 0.0, σ = 1.0)
    d = Distributions.LogNormal(μ,σ)
    x_range = range(0, Distributions.mean(d) + 3*Distributions.std(d), length=1000)
    plt = plot(
        x_range,
        pdf.(d, x_range),
        linewidth = 3,
        label = false,
        xlabel = L"x",
        ylabel = L"p(x)",
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    display(plt)
end

"""
    main()

Main entry point for the ProductFactorModule demonstrations.

This function runs a comprehensive suite of demonstrations showing:
1. Visualization of message approximations with and without Matchbox method for two test cases:
   - Case 1: Positive means (x ~ N(1.5, 0.09), y ~ N(3, 0.04))
   - Case 2: Negative means (x ~ N(-6.5, 1), y ~ N(-8.5, 1))
2. Visualization of the log-normal distribution

The demonstrations highlight the accuracy and limitations of different approximation
strategies for product factors in factor graphs, particularly useful for understanding
message passing in systems like TrueSkill.

# Output Files
Creates multiple visualization files in ~/Downloads/:
- case1_with_matchbox_product_factor_msg_to_{x,y,z}.{pdf,svg}
- case2_with_matchbox_product_factor_msg_to_{x,y,z}.{pdf,svg}
- case1_without_matchbox_product_factor_msg_to_{x,y,z}.{pdf,svg}
- case2_without_matchbox_product_factor_msg_to_{x,y,z}.{pdf,svg}
- lognormal.svg

# Console Output
Prints comparison results for:
- Direct Gaussian approximation (messages to x, y, and z)
- Log-normal approximation (messages to x, y, and z)
"""
function main()
    Random.seed!(1508)

    println("\n" * "="^80)
    println("Product Factor Message Approximation - Comparative Analysis")
    println("="^80)
    
    # Plot Log-Normal distribution
    println("\n📊 Plotting Log-Normal Distribution")
    println("-"^80)
    println("Visualizing LogNormal(μ=0.0, σ=1.0) - theoretical basis for approximations")
    plot_lognormal()
    savefig("~/Downloads/lognormal.svg")
    println("✓ Saved: lognormal.svg")

    # Case 1: Positive means with Matchbox approximation
    println("\n📊 CASE 1: Positive Means + Matchbox Approximation")
    println("-"^80)
    println("Factor graph with product constraint Z = X × Y")
    println("  • Prior X ~ N(1.5, 0.09)  [μ=1.5, σ=0.3]")
    println("  • Prior Y ~ N(3.0, 0.04)  [μ=3.0, σ=0.2]")
    println("  • Prior Z ~ N(4.0, 1.0)")
    println("  • Using 100,000 samples for exact distribution")
    println("  • Comparing: Log-normal approximation (red) vs Matchbox approximation (green)")
    println("\nGenerating comparison plots for messages to X, Y, and Z...")
    plot_product_factor(x = Normal(1.5, 0.3), y = Normal(3, 0.2), z = Normal(4, 1), n = 100000, α=0.001, base="~/Downloads/case1_with_matchbox_", with_matchbox = true)
    println("✓ Saved: case1_with_matchbox_product_factor_msg_to_{x,y,z}.{pdf,svg}")

    # Case 2: Negative means with Matchbox approximation
    println("\n📊 CASE 2: Negative Means + Matchbox Approximation")
    println("-"^80)
    println("Factor graph with product constraint Z = X × Y")
    println("  • Prior X ~ N(-6.5, 1.0)")
    println("  • Prior Y ~ N(-8.5, 1.0)")
    println("  • Prior Z ~ N(4.0, 1.0)  [Note: Z ≈ X×Y ≈ 55, showing tension]")
    println("  • Using 100,000 samples for exact distribution")
    println("  • Comparing: Log-normal approximation (red) vs Matchbox approximation (green)")
    println("\nGenerating comparison plots for messages to X, Y, and Z...")
    plot_product_factor(x = Normal(-6.5, 1), y = Normal(-8.5, 1), z = Normal(4, 1), n = 100000, α=0.001, base="~/Downloads/case2_with_matchbox_", with_matchbox = true)
    println("✓ Saved: case2_with_matchbox_product_factor_msg_to_{x,y,z}.{pdf,svg}")

    # Case 1: Positive means without Matchbox (log-normal only)
    println("\n📊 CASE 1: Positive Means (Log-normal Only)")
    println("-"^80)
    println("Factor graph with product constraint Z = X × Y")
    println("  • Prior X ~ N(1.5, 0.09)  [μ=1.5, σ=0.3]")
    println("  • Prior Y ~ N(3.0, 0.04)  [μ=3.0, σ=0.2]")
    println("  • Prior Z ~ N(4.0, 1.0)")
    println("  • Using 100,000 samples for exact distribution")
    println("  • Showing only log-normal approximation (red)")
    println("\nGenerating comparison plots for messages to X, Y, and Z...")
    plot_product_factor(x = Normal(1.5, 0.3), y = Normal(3, 0.2), z = Normal(4, 1), n = 100000, α=0.001, base="~/Downloads/case1_without_matchbox_", with_matchbox = false)
    println("✓ Saved: case1_without_matchbox_product_factor_msg_to_{x,y,z}.{pdf,svg}")

    # Case 2: Negative means without Matchbox (log-normal only)
    println("\n📊 CASE 2: Negative Means (Log-normal Only)")
    println("-"^80)
    println("Factor graph with product constraint Z = X × Y")
    println("  • Prior X ~ N(-6.5, 1.0)")
    println("  • Prior Y ~ N(-8.5, 1.0)")
    println("  • Prior Z ~ N(4.0, 1.0)")
    println("  • Using 100,000 samples for exact distribution")
    println("  • Showing only log-normal approximation (red)")
    println("\nGenerating comparison plots for messages to X, Y, and Z...")
    plot_product_factor(x = Normal(-6.5, 1), y = Normal(-8.5, 1), z = Normal(4, 1), n = 100000, α=0.001, base="~/Downloads/case2_without_matchbox_", with_matchbox = false)
    println("✓ Saved: case2_without_matchbox_product_factor_msg_to_{x,y,z}.{pdf,svg}")

    println("\n" * "="^80)
    println("✅ All product factor demonstrations completed successfully!")
    println("="^80)
    println("\n📁 Output Location: ~/Downloads/")
    println("\nGenerated Files:")
    println("  Log-Normal Distribution:")
    println("    • lognormal.svg")
    println("  Case 1 (Positive Means) with Matchbox:")
    println("    • case1_with_matchbox_product_factor_msg_to_{x,y,z}.{pdf,svg}")
    println("  Case 2 (Negative Means) with Matchbox:")
    println("    • case2_with_matchbox_product_factor_msg_to_{x,y,z}.{pdf,svg}")
    println("  Case 1 (Positive Means) without Matchbox:")
    println("    • case1_without_matchbox_product_factor_msg_to_{x,y,z}.{pdf,svg}")
    println("  Case 2 (Negative Means) without Matchbox:")
    println("    • case2_without_matchbox_product_factor_msg_to_{x,y,z}.{pdf,svg}")
    println("\n💡 Key Insights:")
    println("  • Blue histograms: Exact sampled distributions")
    println("  • Red lines: Log-normal moment-matched approximations")
    println("  • Green lines: Matchbox approximations (2009 paper)")
    println("  • Compare approximation quality across different scenarios")
    println("\n" * "="^80 * "\n")
end

export approximate_product_factor_msg_to_x, approximate_product_factor_msg_to_y, approximate_product_factor_msg_to_z

end