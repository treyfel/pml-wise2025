"""
    ProductExample

Module for demonstrating the behavior of product factors in probabilistic graphical models.

This module provides functions to:
- Sample and visualize messages in product factors (z = x * y)
- Sample and visualize messages in division factors (y = z / x)
- Compute approximate marginal distributions using importance sampling
- Demonstrate failing cases where Gaussian approximations break down

The module is particularly useful for understanding message passing in factor graphs
and the limitations of Gaussian approximations for non-linear operations.

2025 by Ralf Herbrich
Hasso-Plattner Institute
"""
module ProductExample

using LaTeXStrings      # For LaTeX-formatted plot labels
using Distributions     # For probability distributions
using Statistics        # For statistical functions (mean, std)
using Random            # For random number generation
using Plots             # For visualization

"""
    sample_product_message(; μ_x=4.0, σ_x=1.0, μ_y=6.0, σ_y=1.0, n_samples=1000000, filename="~/Downloads/product_message_histogram.svg")

Sample and visualize the message from a product factor f(x,y,z) = δ(z - x*y) to variable z.

This function demonstrates the distribution of the product of two Gaussian random variables.
Given x ~ N(μ_x, σ_x²) and y ~ N(μ_y, σ_y²), it samples z = x * y and creates a histogram
showing the resulting distribution, which is typically not Gaussian.

# Arguments
- `μ_x::Float64=4.0`: Mean of the Gaussian distribution for x
- `σ_x::Float64=1.0`: Standard deviation of the Gaussian distribution for x
- `μ_y::Float64=6.0`: Mean of the Gaussian distribution for y
- `σ_y::Float64=1.0`: Standard deviation of the Gaussian distribution for y
- `n_samples::Int=1000000`: Number of samples to generate
- `filename::String`: Path where the histogram plot will be saved

# Returns
Nothing. Displays the histogram and saves it to the specified file.

# Example
```julia
sample_product_message(μ_x=5.0, μ_y=6.0, n_samples=10000)
```
"""
function sample_product_message(;
    μ_x = 4.0, σ_x=1.0, μ_y=6.0, σ_y=1.0, 
    n_samples=1000000,
    filename="~/Downloads/product_message_histogram.svg",
)
    # Stores the samples of the message to z
    samples_msg_to_z = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        # Sample x and y from their respective Gaussian distributions
        x = randn() * σ_x + μ_x
        y = randn() * σ_y + μ_y
        # Compute the product z = x * y
        samples_msg_to_z[i] = x * y
    end

    # Plot histogram of samples of the message to z
    plt = histogram(samples_msg_to_z,
        normalize = :pdf,
        alpha=0.5,
        color=:blue,
        xlabel=L"m_{f \to Z}(\cdot)",
        label=false,
        xtickfontsize=18,
        ytickfontsize=18,
        xguidefontsize=20,
        yguidefontsize=20,
        legendfontsize=20,
    )
    display(plt)
    savefig(plt, filename)
end

"""
    sample_division_message(; μ_x=4.0, σ_x=1.0, μ_z=6.0, σ_z=1.0, n_samples=1000000, filename="~/Downloads/division_message_histogram.svg")

Sample and visualize the message from a product factor f(x,y,z) = δ(z - x*y) to variable y.

This function demonstrates the distribution of the division of two Gaussian random variables.
Given x ~ N(μ_x, σ_x²) and z ~ N(μ_z, σ_z²), it samples y = z / x and creates a histogram
showing the resulting distribution. Note that division can produce heavy-tailed distributions,
especially when x is close to zero.

# Arguments
- `μ_x::Float64=4.0`: Mean of the Gaussian distribution for x
- `σ_x::Float64=1.0`: Standard deviation of the Gaussian distribution for x
- `μ_z::Float64=6.0`: Mean of the Gaussian distribution for z
- `σ_z::Float64=1.0`: Standard deviation of the Gaussian distribution for z
- `n_samples::Int=1000000`: Number of samples to generate
- `filename::String`: Path where the histogram plot will be saved

# Returns
Nothing. Displays the histogram and saves it to the specified file.

# Example
```julia
sample_division_message(μ_x=5.0, μ_z=10.0, n_samples=10000)
```
"""
function sample_division_message(;
    μ_x = 4.0, σ_x=1.0, μ_z=6.0, σ_z=1.0, 
    n_samples=1000000,
    filename="~/Downloads/division_message_histogram.svg",
)
    # Stores the samples of the message to y
    samples_msg_to_y = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        # Sample x and z from their respective Gaussian distributions
        x = randn() * σ_x + μ_x
        z = randn() * σ_z + μ_z
        # Compute the division y = z / x
        samples_msg_to_y[i] = z / x
    end

    # Plot histogram of samples of the message to y
    plt = histogram(samples_msg_to_y,
        normalize = :pdf,
        alpha=0.5,
        color=:blue,
        xlabel=L"m_{f \to y}(\cdot)",
        label=false,
        xtickfontsize=18,
        ytickfontsize=18,
        xguidefontsize=20,
        yguidefontsize=20,
        legendfontsize=20,
    )
    display(plt)
    savefig(plt, filename)
end

"""
    compute_approximate_marginal(; μ_x=-3.0, σ_x=1.0, μ_y=5.0, σ_y=1.0, μ_z=1.0, σ_z=1.0, n_samples=1000000, base="~/Downloads/failing_product_factor")

Compute and visualize approximate marginal distributions for a product factor using importance sampling.

This function demonstrates the computation of marginal distributions in a factor graph with
the constraint z = x * y, where x, y, and z have Gaussian prior beliefs. It uses importance
sampling to compute the posterior distribution of each variable given the product constraint.

The function generates three plots showing:
1. The marginal distribution of z given the constraint and priors on x and y
2. The marginal distribution of x given the constraint and priors on y and z
3. The marginal distribution of y given the constraint and priors on x and z

Each plot compares the prior distribution (blue) with the posterior distribution (red),
showing how the product constraint affects the beliefs about each variable.

# Arguments
- `μ_x::Float64=-3.0`: Mean of the prior Gaussian distribution for x
- `σ_x::Float64=1.0`: Standard deviation of the prior Gaussian distribution for x
- `μ_y::Float64=5.0`: Mean of the prior Gaussian distribution for y
- `σ_y::Float64=1.0`: Standard deviation of the prior Gaussian distribution for y
- `μ_z::Float64=1.0`: Mean of the prior Gaussian distribution for z
- `σ_z::Float64=1.0`: Standard deviation of the prior Gaussian distribution for z
- `n_samples::Int=1000000`: Number of samples to use for importance sampling
- `base::String`: Base path for saving the output plots (will append "_x.svg", "_y.svg", "_z.svg")

# Returns
Nothing. Displays the plots and saves them to files with the specified base path.
Prints empirical mean and variance of each variable to stdout.

# Example
```julia
compute_approximate_marginal(μ_x=5.0, μ_y=6.0, μ_z=10.0, n_samples=1000000)
```
"""
function compute_approximate_marginal(;μ_x = -3.0, σ_x=1.0, μ_y=5.0, σ_y=1.0, μ_z=1.0, σ_z=1.0, n_samples=1000000, base="~/Downloads/failing_product_factor")
    # Weights for importance sampling
    weights = Vector{Float64}(undef, n_samples)

    """
        plot_histogram(samples_prior, samples_posterior, weights, var_symbol, lims)

    Internal helper function to create histogram plots comparing prior and posterior distributions.

    # Arguments
    - `samples_prior`: Samples from the prior distribution
    - `samples_posterior`: Samples from the posterior distribution
    - `weights`: Importance weights for the posterior samples
    - `var_symbol`: LaTeX symbol for the variable being plotted
    - `lims`: Tuple of (min, max) limits for the x-axis
    """
    function plot_histogram(samples_prior, samples_posterior, weights, var_symbol, lims)
        # Output the empirical mean and variance based on the weighted samples
        μ = sum(weights .* samples_posterior) / sum(weights)
        σ2 = sum(weights .* (samples_posterior .- μ).^2) / sum(weights)
        println("Empirical Mean of ", var_symbol, ": ", μ)
        println("Empirical Variance of ", var_symbol, ": ", σ2)

        # Subsample the dataset to only draw within the specified limits
        idx = (samples_posterior .> lims[1]) .& (samples_posterior .< lims[2])
        weights = weights[idx]
        samples_posterior = samples_posterior[idx]
        samples_prior = samples_prior[(samples_prior .> lims[1]) .& (samples_prior .< lims[2])]
        
        # Create histogram of prior samples (blue)
        plt = histogram(samples_prior,
            normalize = :pdf,
            alpha=0.5,
            color=:blue,
            xlabel=var_symbol,
            label=false,
            nbins=75,
            xtickfontsize=18,
            ytickfontsize=18,
            xguidefontsize=20,
            yguidefontsize=20,
            legendfontsize=20,
        )
        # Overlay weighted histogram of posterior samples (red)
        histogram!(samples_posterior,
            normalize = :pdf,
            alpha=0.5,
            weights=weights,
            color=:red,
            label=false,
            nbins=75,
        )
        xlims!(plt, lims)

        # Plot fitted Gaussian for posterior (red line)
        x_range = range(minimum(vcat(samples_prior, samples_posterior)), stop=maximum(vcat(samples_prior, samples_posterior)), length=300)
        y_range = pdf(Normal(μ, sqrt(σ2)), x_range)
        plot!(x_range, y_range, label=false, color=:red, linewidth=3)

        # Plot fitted Gaussian for prior (blue line)
        y_prior_range = pdf(Normal(Statistics.mean(samples_prior), Statistics.std(samples_prior)), x_range)
        plot!(x_range, y_prior_range, label=false, color=:blue, linewidth=3)

        display(plt)
    end

    # Compute marginal distribution of z given product constraint z = x * y
    samples_z_prior = Vector{Float64}(undef, n_samples)
    samples_z_posterior = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        # Sample x and y from their prior Gaussian distributions
        sample_x = randn() * σ_x + μ_x
        sample_y = randn() * σ_y + μ_y
        # Compute z using the product factor constraint z = x * y
        samples_z_posterior[i] = sample_x * sample_y
        # Sample z from its prior distribution (for comparison)
        samples_z_prior[i] = randn() * σ_z + μ_z
        # Compute importance weight: likelihood of this z value under the prior
        weights[i] = pdf(Normal(μ_z, σ_z), samples_z_posterior[i])
    end
    # Plot histogram comparing prior and posterior of z
    plot_histogram(samples_z_prior, samples_z_posterior, weights, L"z", (μ_z - 3 * σ_z, μ_z + 3 * σ_z))
    savefig(base * "_z.svg")

    # Compute marginal distribution of x given product constraint z = x * y
    samples_x_prior = Vector{Float64}(undef, n_samples)
    samples_x_posterior = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        # Sample y and z from their prior Gaussian distributions
        sample_y = randn() * σ_y + μ_y
        sample_z = randn() * σ_z + μ_z
        # Compute x using the product factor constraint: x = z / y
        samples_x_posterior[i] = sample_z / sample_y
        # Sample x from its prior distribution (for comparison)
        samples_x_prior[i] = randn() * σ_x + μ_x
        # Compute importance weight: likelihood of this x value under the prior
        weights[i] = pdf(Normal(μ_x, σ_x), samples_x_posterior[i])
    end
    # Plot histogram comparing prior and posterior of x
    plot_histogram(samples_x_prior, samples_x_posterior, weights, L"x", (μ_x - 4 * σ_x, μ_x + 4 * σ_x))
    savefig(base * "_x.svg")

    # Compute marginal distribution of y given product constraint z = x * y
    samples_y_prior = Vector{Float64}(undef, n_samples)
    samples_y_posterior = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        # Sample x and z from their prior Gaussian distributions
        sample_z = randn() * σ_z + μ_z
        sample_x = randn() * σ_x + μ_x
        # Compute y using the product factor constraint: y = z / x
        samples_y_posterior[i] = sample_z / sample_x
        # Sample y from its prior distribution (for comparison)
        samples_y_prior[i] = randn() * σ_y + μ_y
        # Compute importance weight: likelihood of this y value under the prior
        weights[i] = pdf(Normal(μ_y, σ_y), samples_y_posterior[i])
    end
    # Plot histogram comparing prior and posterior of y
    plot_histogram(samples_y_prior, samples_y_posterior, weights, L"y", (μ_y - 5 * σ_y, μ_y + 5 * σ_y))
    savefig(base * "_y.svg")
end     

"""
    main()

Main entry point for the ProductExample module demonstrations.

This function runs a series of demonstrations showing:
1. The distribution of product messages (z = x * y) for different parameter values
2. The distribution of division messages (y = z / x) for different parameter values
3. Approximate marginal distributions using importance sampling for a product factor

The examples demonstrate cases where Gaussian approximations fail, particularly when
the true distributions have heavy tails or are significantly non-Gaussian.

# Example
```julia
using .ProductExample
ProductExample.main()
```
"""
function main()
    # Set random seed for reproducibility
    Random.seed!(1508)

    println("\n" * "="^80)
    println("Product Factor FAILING Cases - When Gaussian Approximations Break Down")
    println("="^80)
    
    # Demonstrate product message: z = 5 * 6 (with Gaussian noise)
    println("\n📊 DEMONSTRATION 1: Product Message Distribution")
    println("-"^80)
    println("Sampling from the message distribution of z = x × y")
    println("  • Input X ~ N(5.0, 1.0)")
    println("  • Input Y ~ N(6.0, 1.0)")
    println("  • Output: Distribution of Z = X × Y")
    println("  • Using 10,000 samples")
    println("  • Expected product: μ_z ≈ 30")
    println("\nGenerating histogram...")
    sample_product_message(μ_x = 5, μ_y = 6, n_samples=10000, filename="~/Downloads/product_message_histogram_5_times_6.svg")
    println("✓ Saved: product_message_histogram_5_times_6.svg")

    # Demonstrate division message: y = 10 / 5 (with Gaussian noise)
    println("\n📊 DEMONSTRATION 2: Division Message Distribution")
    println("-"^80)
    println("Sampling from the message distribution of y = z / x")
    println("  • Input X ~ N(5.0, 1.0)")
    println("  • Input Z ~ N(10.0, 1.0)")
    println("  • Output: Distribution of Y = Z / X")
    println("  • Using 10,000 samples")
    println("  • Expected quotient: μ_y ≈ 2")
    println("  • Note: Division can produce heavy-tailed distributions!")
    println("\nGenerating histogram...")
    sample_division_message(μ_x = 5, μ_z = 10, n_samples=10000, filename="~/Downloads/division_message_histogram_5_div_10.svg")
    println("✓ Saved: division_message_histogram_5_div_10.svg")

    # Demonstrate approximate marginals with importance sampling
    # This shows a "failing" case where the product constraint conflicts with priors
    println("\n📊 DEMONSTRATION 3: Failing Product Factor with Importance Sampling")
    println("-"^80)
    println("Computing approximate marginals with conflicting constraints")
    println("Factor graph with product constraint Z = X × Y:")
    println("  • Prior X ~ N(5.0, 1.0)")
    println("  • Prior Y ~ N(6.0, 1.0)")
    println("  • Prior Z ~ N(10.0, 1.0)")
    println("  • Conflict: X×Y ≈ 30 but Z ≈ 10 (strong tension!)")
    println("  • Using 1,000,000 samples for importance sampling")
    println("\nThis demonstrates a FAILING case where:")
    println("  ⚠️  The product constraint Z = X × Y conflicts with the prior beliefs")
    println("  ⚠️  Gaussian approximations will struggle with the resulting distributions")
    println("  ⚠️  Heavy tails and non-Gaussian shapes emerge from the division operations")
    println("\nComputing marginal distributions...")
    compute_approximate_marginal(μ_x = 5, μ_y = 6, μ_z = 10, σ_z = 1, n_samples=1000000, base="~/Downloads/failing_product_factor")

    println("\n" * "="^80)
    println("✅ All failing case demonstrations completed successfully!")
    println("="^80)
    println("\n📁 Output Location: ~/Downloads/")
    println("\nGenerated Files:")
    println("  Product Message Distribution:")
    println("    • product_message_histogram_5_times_6.svg")
    println("  Division Message Distribution:")
    println("    • division_message_histogram_5_div_10.svg")
    println("  Failing Product Factor Marginals:")
    println("    • failing_product_factor_x.svg")
    println("    • failing_product_factor_y.svg")
    println("    • failing_product_factor_z.svg")
    println("\n💡 Key Insights:")
    println("  • Product of Gaussians → Non-Gaussian (often log-normal-like)")
    println("  • Division of Gaussians → Heavy-tailed distributions")
    println("  • Blue histograms: Prior distributions")
    println("  • Red histograms: Posterior distributions (with importance weights)")
    println("  • Conflicting constraints lead to challenging posterior shapes")
    println("  • These cases show WHY Gaussian approximations can fail!")
    println("\n⚠️  Warning: The empirical statistics printed above show")
    println("    the weighted posterior means and variances, which may")
    println("    differ significantly from the prior beliefs.")
    println("\n" * "="^80 * "\n")
end

end
