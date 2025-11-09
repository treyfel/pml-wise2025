"""
    BayesianRegressionPlots

A module for visualizing Bayesian linear regression with various basis functions,
demonstrating posterior evolution, uncertainty quantification, and regularization effects.

# Overview

This module implements visualization tools for Bayesian linear regression, showing:
1. Evolution of posterior distributions as data arrives
2. Comparison of different basis function families
3. Effect of prior variance (regularization) on predictions
4. Uncertainty quantification via predictive distributions

# Bayesian Linear Regression

## Model

For regression with basis functions: y = wᵀφ(x) + ε where ε ~ N(0, σ²)

**Likelihood:**
```
p(y|X, w, σ²) = N(Φw, σ²I)
```
where Φ is the design matrix with rows φ(xᵢ)ᵀ

**Prior:**
```
p(w|τ²) = N(0, τ²I)
```

**Posterior:**
```
p(w|X, y, σ², τ²) = N(μₚₒₛₜ, Σₚₒₛₜ)
```

where:
```
Σₚₒₛₜ = (σ⁻²ΦᵀΦ + τ⁻²I)⁻¹
μₚₒₛₜ = σ⁻²ΣₚₒₛₜΦᵀy
```

**Predictive Distribution:**
```
p(y*|x*, X, y) = N(μₚₒₛₜᵀφ(x*), σ² + φ(x*)ᵀΣₚₒₛₜφ(x*))
```

# Basis Functions

The module supports multiple basis function families:

## Polynomial Basis
φⱼ(x) = xʲ
- Simple, interpretable
- Can suffer from numerical instability for high degrees
- Natural for polynomial trends

## Fourier Basis
φⱼ(x) = cos(πjx/2) for even j, sin(π(j-1)x/2) for odd j
- Periodic structure
- Good for oscillatory patterns
- Orthogonal basis

## Gaussian (RBF) Basis
φⱼ(x) = exp(-(x-μⱼ)²/(2σ²))
- Localized, smooth
- Commonly used in kernel methods
- Centers typically spread across input domain

## Sigmoid Basis
φⱼ(x) = 1/(1 + exp(-(x-μⱼ)))
- S-shaped curves
- Used in neural networks
- Saturates at extremes

# Key Concepts Demonstrated

## Sequential Learning
Posterior becomes more concentrated as more data arrives, representing
increased certainty about parameters.

## Prior as Regularization
- Large τ (weak prior): Allows complex fits, risk of overfitting
- Small τ (strong prior): Encourages simple solutions, regularization

## Predictive Uncertainty
- **Aleatoric**: Irreducible noise (σ²)
- **Epistemic**: Parameter uncertainty (φᵀΣₚₒₛₜφ)
- Total predictive variance combines both

## Weight Space vs. Function Space
- Weight space: Posterior over parameters
- Function space: Induced distribution over functions

# Visualizations Provided

1. **2D Weight Space**: Likelihood and posterior surfaces
2. **Function Samples**: Random functions from posterior
3. **Predictive Ribbons**: Mean ± std of predictions
4. **Multivariate Gaussians**: 3D density visualizations
5. **Dependence Examples**: Ring distribution (zero covariance, strong dependence)

# Applications

- Teaching Bayesian inference
- Understanding regularization
- Comparing basis functions
- Uncertainty quantification
- Model selection insights

# Author
Ralf Herbrich, Hasso-Plattner Institute, 2025

# References
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Chapter 3.
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. Chapter 7.
- Rasmussen & Williams (2006). Gaussian Processes for Machine Learning.
"""
module BayesianRegressionPlots

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

# ============================================================================
# Data Generation
# ============================================================================

"""
    generate_data_points(n; a0=0.5, a1=-0.3, σ=0.2) -> (X, y)

Generates n noisy observations from a linear function for 2D Bayesian regression.

Creates data from: y = a₀x + a₁ + ε where ε ~ N(0, σ²)

# Arguments
- `n::Int`: Number of data points

# Keywords
- `a0::Float64=0.5`: True slope
- `a1::Float64=-0.3`: True intercept
- `σ::Float64=0.2`: Noise standard deviation

# Returns
- `X::Matrix{Float64}`: n×2 design matrix [x, 1]
- `y::Matrix{Float64}`: n×1 observation vector

# Examples
```jldoctest
julia> X, y = generate_data_points(3)
([0.00013993... 1.0; ...], [...])
```

# Notes
- Uses fixed seed (40) for reproducibility
- x-values uniform in [-1, 1]
"""
function generate_data_points(n; a0 = 0.5, a1 = -0.3, σ = 0.2)
    # Set seed for reproducible generation
    Random.seed!(40)
    # Generate design matrix: [x, 1] for each sample
    X = hcat(2 * rand(n, 1) - ones(n, 1), ones(n, 1))
    # Generate noisy observations
    y = X * [a0; a1] + randn(n, 1) * σ

    return X, y
end

"""
    generate_data(n, f; σ=0.1, from=0, to=1) -> NamedTuple

Generates synthetic data from a function with Gaussian noise.

Creates observations yᵢ = f(xᵢ) + εᵢ where εᵢ ~ N(0, σ²) at n evenly-spaced
points in [from, to].

# Arguments
- `n::Int64`: Number of observations
- `f::Function`: True underlying function

# Keywords
- `σ::Float64=0.1`: Noise standard deviation
- `from::Real=0`: Start of input range
- `to::Real=1`: End of input range

# Returns
- `NamedTuple`: (x = [...], y = [...]) with observations

# Examples
```jldoctest
julia> data = generate_data(11, x -> sin(x*π), σ=0.15, from=0, to=1)
(x = [0.0, 0.1, ..., 1.0], y = [...])
```

# Use Cases
- Testing basis function fitting
- Demonstrating overfitting vs. regularization
- Creating synthetic regression problems

# Notes
- Points are evenly spaced (not random)
- Useful for visualizing function fits
- Noise is i.i.d. Gaussian
"""
function generate_data(n::Int64, f; σ = 0.1, from = 0, to = 1)
    # Generate evenly-spaced inputs
    xs = collect(range(from, to, n))
    # Evaluate function and add noise
    ys = map(f, xs) + randn((n, 1)) * σ
    return (x = xs, y = ys)
end

# ============================================================================
# Basis Functions
# ============================================================================

"""
    polynomial_basis(x::Float64, j) -> Float64

Computes the j-th polynomial basis function: φⱼ(x) = xʲ

# Arguments
- `x::Float64`: Input value
- `j`: Polynomial degree (integer ≥ 0)

# Returns
- `Float64`: The value xʲ

# Examples
```jldoctest
julia> polynomial_basis(2.0, 3)
8.0

julia> polynomial_basis(2.0, 0)
1.0
```

# Properties
- φ₀(x) = 1 (constant basis)
- Monomial basis
- Can be numerically unstable for high degrees

# Use Cases
- Polynomial regression
- Taylor series approximation
- Simple curve fitting
"""
function polynomial_basis(x::Float64, j)
    return x^j
end

"""
    fourier_basis(x::Float64, j) -> Float64

Computes the j-th Fourier basis function.

Uses alternating sine/cosine basis:
- Even j: φⱼ(x) = cos(πjx/2)
- Odd j: φⱼ(x) = sin(π(j-1)x/2)

# Arguments
- `x::Float64`: Input value
- `j`: Basis function index (integer ≥ 0)

# Returns
- `Float64`: Basis function value

# Examples
```jldoctest
julia> fourier_basis(2.0, 3)
0.10944260690631982

julia> fourier_basis(2.0, 0)
1.0
```

# Properties
- Orthogonal basis functions
- Periodic with period 4/j
- Good for oscillatory patterns
- φ₀(x) = 1 (constant)

# Use Cases
- Periodic signal modeling
- Frequency domain representation
- Spectral analysis
"""
function fourier_basis(x::Float64, j)
    return if (j % 2 == 0)
        cos(π * j / 2 * x)
    else
        sin(π * (j - 1) / 2 * x)
    end
end

"""
    gauss_basis(x::Float64, j; σ=1) -> Float64

Computes the j-th Gaussian (RBF) basis function centered at j.

Evaluates: φⱼ(x) = exp(-(x-j)²/(2σ²)) / (√(2πσ²))

# Arguments
- `x::Float64`: Input value
- `j`: Center location (typically integer)

# Keywords
- `σ::Float64=1`: Width parameter (standard deviation)

# Returns
- `Float64`: Gaussian basis function value

# Examples
```jldoctest
julia> gauss_basis(2.0, 3, σ=1)
0.24197072451914337

julia> gauss_basis(2.0, 0, σ=1)
0.05399096651318806
```

# Properties
- Localized around center j
- Smooth, differentiable everywhere
- Decays exponentially away from center
- Width controlled by σ

# Use Cases
- Radial Basis Function (RBF) networks
- Localized function approximation
- Kernel methods
- Smooth interpolation

# Notes
- For regression, centers typically spread across input domain
- Smaller σ → more localized, can lead to overfitting
- Larger σ → more overlap, smoother fits
"""
function gauss_basis(x::Float64, j; σ = 1)
    return pdf(Normal(j, σ), x)
end

"""
    sigmoid_basis(x::Float64, j) -> Float64

Computes the j-th sigmoid basis function: φⱼ(x) = σ(x - j)

where σ(z) = 1/(1 + e⁻ᶻ) is the logistic sigmoid.

# Arguments
- `x::Float64`: Input value
- `j`: Inflection point location

# Returns
- `Float64`: Sigmoid value in (0, 1)

# Examples
```jldoctest
julia> sigmoid_basis(2.0, 3)
0.2689414213699951

julia> sigmoid_basis(2.0, 0)
0.8807970779778824
```

# Properties
- S-shaped curve
- Inflection point at x = j
- Saturates at 0 and 1
- Smooth transition region

# Use Cases
- Neural network activations
- Smooth step functions
- Classification boundaries
- Gating mechanisms

# Notes
- Related to logistic regression
- Differentiable everywhere
- Can represent smooth thresholds
"""
function sigmoid_basis(x::Float64, j)
    return exp(x - j) / (1 + exp(x - j))
end

# ============================================================================
# 3D Visualization
# ============================================================================

"""
    plot_surface(f; colors=:blues, alpha=0.1, start=-1, stop=1) -> Plot

Creates 3D surface plot with mesh for 2D→1D functions.

Visualizes functions: f: R² → R

# Arguments
- `f::Function`: Two-variable function

# Keywords
- `colors`: Color scheme (default: :blues)
- `alpha`: Transparency (default: 0.1)
- `start`: Start of range for w₁, w₂ (default: -1)
- `stop`: End of range for w₁, w₂ (default: 1)

# Returns
- `Plot`: The 3D surface plot

# Examples
```julia
# Simple plane
plot_surface(w -> w[1] + w[2], colors=:reds)

# Saddle point
plot_surface(w -> w[1]^2 - w[2]^2, alpha=0.3)
```

# Details
- Creates a fine grid (100 points) and a coarse grid (30 points) for w₁, w₂
- Plots surface with color gradient and transparency
- Adds contour lines for clarity

# Notes
- Use `plot!(...)` to add to existing plots
- Fine-tune `start`/`stop` for different function domains
"""
function plot_surface(f; colors = :blues, alpha = 0.1, start=-1, stop=1)
    w1 = range(start, stop = stop, length = 100)
    w2 = range(start, stop = stop, length = 100)
    w1_coarse = range(start, stop = stop, length = 30)
    w2_coarse = range(start, stop = stop, length = 30)
    p = plot(
        w1,
        w2,
        (w1, w2) -> f([w1, w2]),
        seriestype = :surface,
        line_z = 0.5,
        color = colors,
        fillalpha = alpha,
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        ztickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        zguidefontsize = 16,
    )
    for w1 in w1_coarse
        plot!(
            [w1 for w2 in w2_coarse],
            w2_coarse,
            [f([w1; w2]) for w2 in w2_coarse],
            line_z=[f([w1; w2]) for w2 in w2_coarse],
            color=:black,
            linewidth=0.5
        )
    end

    for w2 in w2_coarse
        plot!(
            w1_coarse,
            [w2 for w1 in w1_coarse],
            [f([w1; w2]) for w1 in w1_coarse],
            line_z=[f([w1; w2]) for w1 in w1_coarse],
            color=:black,
            linewidth=0.5
        )
    end

    xlabel!(L"w_1")
    ylabel!(L"w_2")
    display(p)
    return p
end

# ============================================================================
# 2D Weight Space Visualization
# ============================================================================

"""
    plot_bayesian_inference(X, y; σ=0.2, τ=sqrt(1/2), no_samples=200, base_name="fig") -> String

Generates comprehensive Bayesian regression visualizations for 2D weight space.

Creates four plots:
1. Data scatter in input space
2. Likelihood surface over (w₁, w₂)
3. Posterior surface over (w₁, w₂)
4. Function samples from posterior

# Arguments
- `X::Matrix{Float64}`: Design matrix (n×2)
- `y::Matrix{Float64}`: Observation vector (n×1)

# Keywords
- `σ::Float64=0.2`: Observation noise standard deviation
- `τ::Float64=sqrt(1/2)`: Prior standard deviation
- `no_samples::Int=200`: Number of function samples to plot
- `base_name::String="fig"`: Base name for output files

# Returns
- `String`: Base name for saved figure files

# Examples
```jldoctest
julia> plot_bayesian_inference(X, y; σ=0.2, τ=sqrt(1/2), no_samples=200, base_name="~/Downloads/bayes3")
"/Users/rherbrich/Downloads/bayes3_functions.png"
```
"""
function plot_bayesian_inference(
    X,
    y;
    σ = 0.2,
    τ = sqrt(1 / 2),
    no_samples = 200,
    base_name = "fig",
)
    # Computes the likelihood of the 2D linear function for the dataset given by `X` and `y` using the noise standard deviation `σ`
    function likelihood(w; σ = sqrt(1 / 2))
        return pdf(MvNormal(X * w, I * σ^2), y)
    end

    # compute the Bayesian posterior
    Σ = Hermitian(inv(1 / σ^2 * X' * X + 1 / τ^2 * Diagonal(ones(2))))
    μ = vec(1 / σ^2 * Σ * X' * y)

    # plot the likelihood over the weight space
    p = plot_surface(
        W -> likelihood(W)[1],
        colors = :reds,
        alpha = 0.1,
        start=-1,
        stop=1
    )
    savefig(p, base_name * "_likel.png")

    # plot the posterior over the weight space
    p =plot_surface(
        W -> pdf(MvNormal(μ, Σ), W)[1],
        colors = :greens,
        alpha = 0.1,
        start=-1,
        stop=1
    )
    savefig(p, base_name * "_post.png")

    # plot the input space
    p = plot(
        X[:, 1],
        y,
        color = :blue,
        seriestype = :scatter,
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x")
    ylabel!(L"y")
    xlims!(-1, 1)
    ylims!(-1, 1)
    display(p)
    savefig(p, base_name * "_data.png")

    xs = range(-1, stop = 1, length = 100)
    w = rand(MvNormal(μ, Σ), no_samples)
    for j = 1:no_samples
        width = if (j % no_samples ÷ 3 == 0)
            0.1
        else
            0.1
        end
        plot!(xs, map(x -> w[1, j] * x + w[2, j], xs), linewidth = width, color = :red)
    end
    plot!(X[:, 1], y, color = :blue, seriestype = :scatter)
    display(p)
    savefig(p, base_name * "_functions.png")
end

# ============================================================================
# General Basis Function Fitting
# ============================================================================

"""
    plot_Bayesian_fit(train_data; σ=0.1, τ=0.5, ϕ=..., color=:blue) -> Nothing

Adds Bayesian regression fit with uncertainty ribbons to current plot.

Computes posterior over weights for arbitrary basis functions and plots:
- Mean prediction: E[y*|x*] = μₚₒₛₜᵀφ(x*)
- Uncertainty ribbon: ±√(σ² + φ(x*)ᵀΣₚₒₛₜφ(x*))

# Arguments
- `train_data::NamedTuple`: Training data with fields (x, y)

# Keywords
- `σ::Float64=0.1`: Observation noise std dev
- `τ::Float64=0.5`: Prior std dev (regularization strength)
- `ϕ::Function`: Feature map x → φ(x) (vector of basis evaluations)
- `color::Symbol=:blue`: Color for line and ribbon

# Feature Map
The function ϕ should map a scalar x to a vector of basis function values:
```julia
ϕ = x -> [φ₀(x), φ₁(x), ..., φₘ(x)]
```

# Examples
```julia
# Polynomial of degree 3
plot_Bayesian_fit(data, ϕ = x -> map(j -> polynomial_basis(x,j), 0:3))

# Gaussian RBF with 10 centers
plot_Bayesian_fit(data, ϕ = x -> map(j -> gauss_basis(x, j/10), 0:10))
```

# Algorithm
1. Construct design matrix Φ from training inputs
2. Compute posterior: Σₚₒₛₜ = (τ⁻²I + σ⁻²ΦᵀΦ)⁻¹, μₚₒₛₜ = σ⁻²ΣₚₒₛₜΦᵀy
3. For each test point, compute predictive mean and variance
4. Plot mean with uncertainty ribbon

# Predictive Distribution
For a test point x*:
```
p(y*|x*, D) = N(μₚₒₛₜᵀφ(x*), σ² + φ(x*)ᵀΣₚₒₛₜφ(x*))
```

The variance has two components:
- **σ²**: Aleatoric (observation noise)
- **φᵀΣₚₒₛₜφ**: Epistemic (parameter uncertainty)

# Notes
- Modifies current plot (use plot!())
- Ribbon shows ±1 std dev
- Cholesky decomposition for numerical stability
"""
function plot_Bayesian_fit(
    train_data;
    σ = 0.1,
    τ = 0.5,
    ϕ = x -> map(j -> polynomial_basis(x, j), 1:2),
    color = :blue,
)
    # Generate design matrix Φ: each row is φ(xᵢ)ᵀ
    Φ = transpose(hcat([ϕ(x) for x in train_data.x]...))

    # Compute posterior parameters
    # Prior precision matrix
    Σinv = 1.0 / τ^2 * Diagonal(ones(size(Φ, 2)))
    y = train_data.y
    # Posterior precision via Cholesky for stability
    C = cholesky(Σinv + 1.0 / σ^2 * Φ' * Φ)
    # Posterior mean: μ = Σ(σ⁻²Φᵀy)
    μ = C.U \ (C.L \ (1.0 / σ^2 * Φ' * y))

    # Compute predictive distribution for test points
    xs = collect(range(-0.3, 1.3, 100))
    Φ_test = transpose(hcat([ϕ(x) for x in xs]...))
    # For each test point: (mean, std_dev)
    pred = map(ϕ -> (vec(μ)' * ϕ, sqrt(σ^2 + ϕ' * (C.U \ (C.L \ ϕ)))), eachrow(Φ_test))
    
    # Add to current plot
    plot!(
        xs,
        map(x -> x[1], pred),
        ribbon = map(x -> x[2], pred),
        fillalpha = 0.2,
        linewidth = 3,
        color = color,
    )
end

"""
    plot_fit(train_data; ϕ=..., file_name="~/Downloads/poly_fit.svg") -> Nothing

Creates comparison plot of Bayesian fits with different regularization strengths.

Plots training data with two posterior predictive distributions:
- Blue: Strong regularization (τ = 1)
- Red: Weak regularization (τ = 10)

# Arguments
- `train_data::NamedTuple`: Training data (x, y)

# Keywords
- `ϕ::Function`: Basis function map (default: polynomial degree 6)
- `file_name::String`: Output file path

# Examples
```julia
data = generate_data(11, x -> sin(x*π), σ=0.15)

# Polynomial basis
plot_fit(data, ϕ = x -> map(j -> polynomial_basis(x,j), 0:6))

# Gaussian RBF basis
plot_fit(data, ϕ = x -> map(j -> gauss_basis(x, j/20, σ=0.15), 0:20))
```

# Interpretation

**Strong regularization (τ = 1, blue):**
- Smoother fit
- Less prone to overfitting
- Higher bias, lower variance
- Wider uncertainty bands

**Weak regularization (τ = 10, red):**
- More flexible fit
- Can overfit with complex bases
- Lower bias, higher variance
- Narrower uncertainty bands near data

# Bias-Variance Tradeoff
- Small τ: High bias, low variance (underfitting risk)
- Large τ: Low bias, high variance (overfitting risk)
- Optimal τ balances the two

# Use Cases
- Demonstrating regularization effects
- Model selection visualization
- Teaching bias-variance tradeoff
- Comparing basis function families
"""
function plot_fit(train_data; ϕ = x -> map(j -> polynomial_basis(x, j), 0:6), file_name="~/Downloads/poly_fit.svg")
    # Create scatter plot of training data
    p = plot(
        train_data.x,
        train_data.y,
        seriestype = :scatter,
        legend = false,
        color = :orange,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x")
    ylabel!(L"y")
    
    # Add strong regularization fit (blue)
    plot_Bayesian_fit(
        train_data,
        τ = 1,
        color = :blue,
        ϕ = ϕ,
    )
    
    # Add weak regularization fit (red)
    plot_Bayesian_fit(
        train_data,
        τ = 10,
        color = :red,
        ϕ = ϕ,
    )
    
    display(p)
    savefig(p, file_name)
end

# ============================================================================
# Gaussian Distribution Visualization
# ============================================================================

"""
    plot_normal(μ, Σ; lims=(-3,3), file_name="marginal.png") -> String

Visualizes 2D Gaussian distribution as 3D surface.

Creates a 3D plot of the probability density function:
p(x) = (2π)⁻¹|Σ|⁻¹/² exp(-½(x-μ)ᵀΣ⁻¹(x-μ))

# Arguments
- `μ::Vector{Float64}`: Mean vector (2D)
- `Σ::Matrix{Float64}`: Covariance matrix (2D)

# Keywords
- `lims::Tuple=(-3,3)`: Limits for x₁, x₂ axes
- `file_name::String="marginal.png"`: Output file name

# Returns
- `String`: File name of the saved plot

# Examples
```jldoctest
julia> plot_normal(vec([0;0]),[[1 0];[0 1]]; file_name="~/Downloads/std_normal.png")
"/Users/rherbrich/Downloads/std_normal.png"
```
"""
function plot_normal(μ, Σ; lims = (-3, 3), file_name = "marginal.png")
    # plot the Normal distribution
    p = plot_surface(
        x -> pdf(MvNormal(μ, Σ), x)[1],
        colors = :greens,
        alpha = 0.1,
        start = lims[1],
        stop = lims[2],
    )
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    zlabel!(L"p(x_1,x_2)")
    display(p)
    savefig(p, file_name)
end

"""
    plot_dist(n; α=0.1, file_name="ring.png") -> String

Demonstrates dependence without correlation using a ring distribution.

Generates samples from a distribution with:
- **Zero covariance**: E[(X₁-μ₁)(X₂-μ₂)] = 0
- **Strong dependence**: X₁² + X₂² ≈ constant

# Arguments
- `n::Int`: Number of samples

# Keywords
- `α::Float64=0.1`: Scatter plot transparency
- `file_name::String`: Output file path

# Distribution

Samples (X₁, X₂) are generated as:
```
θ ~ Uniform(0, 2π)
r ~ Uniform(1, 2)
X₁ = r cos(θ)
X₂ = r sin(θ)
```

This creates a ring (annulus) distribution.

# Examples
```jldoctest
julia> plot_dist(10000, file_name="~/Downloads/ring.png")
Empirical covariance: -0.00544...
"~/Downloads/ring.png"
```

# Key Insight

**Covariance ≠ Independence**

- Covariance measures *linear* dependence
- This distribution has zero covariance but strong *nonlinear* dependence
- Variables are clearly not independent: knowing X₁ constrains X₂

# Educational Value

Demonstrates that:
1. Zero covariance doesn't imply independence
2. Correlation only captures linear relationships
3. Higher-order moments or copulas needed for general dependence

# Notes
- Empirical covariance printed to console
- Should be close to 0 for large n
- Useful counterexample for teaching probability theory
"""
function plot_dist(n; α = 0.1, file_name = "ring.png")
    # Generate uniform angles
    θ = 2 * π * rand(n)
    # Generate random radii in [1, 2]
    r = ones(n) + rand(n)
    # Convert to Cartesian coordinates
    X = hcat(r .* cos.(θ), r .* sin.(θ))
    
    # Compute empirical covariance
    μ1, μ2 = map(x -> mean(x), eachcol(X))
    println("Empirical covariance: ", mean(map(x -> (x[1] - μ1) * (x[2] - μ2), eachrow(X))))

    # Create scatter plot
    p = plot(
        X[:, 1],
        X[:, 2],
        seriestype = :scatter,
        color = :blue,
        alpha = α,
        legend = false,
        aspect_ratio = :equal,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
    savefig(p, file_name)
end

# ============================================================================
# Main Demonstration
# ============================================================================

"""
    main() -> Nothing

Runs comprehensive Bayesian regression demonstrations.

Generates visualizations showing:
1. Sequential learning with 2, 3, 5, 20 data points
2. 2D Gaussian distribution
3. Ring distribution (covariance vs. dependence)
4. Polynomial basis fitting with regularization
5. Gaussian RBF basis fitting with regularization

# Output Files

Generated in ~/Downloads/:
- `bayes2_*.png`: 2-point regression (4 files)
- `bayes3_*.png`: 3-point regression (4 files)
- `bayes5_*.png`: 5-point regression (4 files)
- `bayes20_*.png`: 20-point regression (4 files)
- `std_normal.png`: Standard 2D Gaussian
- `ring.png`: Ring distribution
- `poly_fit.svg`: Polynomial basis comparison
- `gauss_fit.svg`: Gaussian RBF basis comparison

# Educational Sequence

1. **Sequential learning**: Watch posterior concentrate
2. **Likelihood vs. posterior**: See prior regularization
3. **Function samples**: Visualize uncertainty
4. **Regularization**: Compare strong vs. weak priors
5. **Basis functions**: Compare polynomial vs. RBF

# Notes
- Uses fixed random seeds for reproducibility
- Total execution time: ~30-60 seconds
- Requires write access to ~/Downloads/
"""
function main()
    # 2D weight space demonstrations
    X, y = generate_data_points(20)

    println("Generating 2D weight space visualizations...")
    plot_bayesian_inference(X[1:2, :], y[1:2], base_name = "~/Downloads/bayes2")
    plot_bayesian_inference(X[1:3, :], y[1:3], base_name = "~/Downloads/bayes3")
    plot_bayesian_inference(X[1:5, :], y[1:5], base_name = "~/Downloads/bayes5")
    plot_bayesian_inference(X[1:20, :], y[1:20], base_name = "~/Downloads/bayes20")

    # Gaussian distribution examples
    println("\nGenerating Gaussian distribution visualizations...")
    plot_normal(vec([0; 0]), [[1 0]; [0 1]]; file_name = "~/Downloads/std_normal.png")
    plot_dist(10000; file_name = "~/Downloads/ring.png")

    # Basis function fitting demonstrations
    println("\nGenerating basis function fitting examples...")
    Random.seed!(41)
    train_data = generate_data(11, x -> sin(x * π), σ = 0.15, from = 0, to = 1)
    
    # Polynomial basis
    plot_fit(train_data; ϕ = x -> map(j -> polynomial_basis(x, j), 0:6), file_name="~/Downloads/poly_fit.svg")
    
    # Gaussian RBF basis
    plot_fit(train_data; ϕ = x -> map(j -> gauss_basis(x, j / 20, σ = 0.15), 0:20), file_name="~/Downloads/gauss_fit.svg")
    
    println("\nAll visualizations complete!")
end

end # module