# Computes the closest Gaussian approximation for a mixture of Gaussians
#
# This module implements various divergence measures (KL, reverse KL, α-divergence)
# and uses them to find optimal Gaussian approximations to mixtures of Gaussians.
# It demonstrates how different divergence measures lead to different approximations
# with varying moment-matching properties.
#
# 2025 by Ralf Herbrich
# Hasso-Plattner Institute

module DistancePlots

using LaTeXStrings
using Plots
using Distributions

"""
    MixtureOfGaussian

Represents a finite mixture of Gaussian distributions.

# Fields
- `weights::Vector{Float64}`: Mixing weights (must sum to 1)
- `gaussians::Vector{Normal}`: Component Gaussian distributions

# Example
```julia
mog = MixtureOfGaussian([0.3, 0.7], [Normal(0, 0.5), Normal(6, 1.5)])
# Creates a mixture: 0.3 * N(0, 0.5²) + 0.7 * N(6, 1.5²)
```
"""
struct MixtureOfGaussian
    weights::Vector{Float64}
    gaussians::Vector{Normal}
    
    # Constructor with validation
    function MixtureOfGaussian(weights::Vector{Float64}, gaussians::Vector{Normal{Float64}})
        @assert length(weights) == length(gaussians) "Number of weights must match number of Gaussians"
        @assert all(weights .>= 0) "All weights must be non-negative"
        @assert abs(sum(weights) - 1.0) < 1e-10 "Weights must sum to 1"
        new(weights, gaussians)
    end
end

"""
    get_range(mog::MixtureOfGaussian)

Computes the effective range of values for a mixture of Gaussians, extending ±6 standard
deviations from each component mean.

# Arguments
- `mog::MixtureOfGaussian`: The mixture of Gaussians

# Returns
- `(min_x, max_x)`: Tuple of minimum and maximum x values covering the mixture support
"""
function get_range(mog::MixtureOfGaussian)
    min_x = minimum(map(x -> x.μ, mog.gaussians) - 6 * map(x -> x.σ, mog.gaussians))
    max_x = maximum(map(x -> x.μ, mog.gaussians) + 6 * map(x -> x.σ, mog.gaussians))
    return min_x, max_x
end

"""
    mog_pdf(mog::MixtureOfGaussian, x::Float64)

Evaluates the probability density function of the mixture of Gaussians at point x.

# Arguments
- `mog::MixtureOfGaussian`: The mixture of Gaussians
- `x::Float64`: Point at which to evaluate the PDF

# Returns
- `Float64`: PDF value p(x) = Σᵢ wᵢ * N(x; μᵢ, σᵢ²)
"""
function mog_pdf(mog::MixtureOfGaussian, x::Float64)
    y = 0.0
    for i in 1:length(mog.weights)
        y += mog.weights[i] * pdf(mog.gaussians[i], x)
    end
    return y
end

"""
    plot_mog(mog::MixtureOfGaussian; approx=nothing, title=nothing, ylim=nothing)

Creates a visualization of a mixture of Gaussians, optionally with a Gaussian approximation overlay.

# Arguments
- `mog::MixtureOfGaussian`: The mixture to plot
- `approx::Union{Normal,Nothing}`: Optional Gaussian approximation to overlay in red (default: nothing)
- `title::Union{String,Nothing}`: Optional plot title (default: nothing)
- `ylim::Union{Tuple,Nothing}`: Optional y-axis limits (default: nothing)

# Example
```julia
mog = MixtureOfGaussian([0.5, 0.5], [Normal(0, 1), Normal(3, 1)])
plot_mog(mog, approx=Normal(1.5, 1.5), title="Mixture vs Approximation")
```
"""
function plot_mog(mog::MixtureOfGaussian; approx = nothing, title=nothing, ylim=nothing)
    min_x, max_x = get_range(mog)
    xs = range(min_x, max_x, length=1000)

    p = plot(xs,
        x -> mog_pdf(mog, x),
        label=false,
        xlabel=L"x",
        ylabel=L"p(x)",
        linewidth=3,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
        legendfontsize=16,
    )
    if !isnothing(approx)
        plot!(xs, 
            x -> pdf(approx, x), 
            linewidth=3,
            label = false,
            color=:red,
        )
    end
    if !isnothing(title)
        title!(title)
    end
    if !isnothing(ylim)
        ylims!(ylim)
    end

    display(p)
end

"""
    KL(p::Vector{Float64}, q::Vector{Float64})

Computes the Kullback-Leibler divergence KL(p || q) between two discretized probability distributions.

KL(p || q) = Σᵢ p(xᵢ) log(p(xᵢ) / q(xᵢ))

This is the forward KL divergence, which penalizes q for having mass where p has little mass.
It tends to produce mode-seeking behavior (the approximation focuses on a single mode).

# Arguments
- `p::Vector{Float64}`: Discretized target distribution (should sum to 1)
- `q::Vector{Float64}`: Discretized approximating distribution (should sum to 1)

# Returns
- `Float64`: The KL divergence value (always non-negative)
"""
function KL(p::Vector{Float64}, q::Vector{Float64})
    return sum(p .* log.(p ./ q))
end

"""
    KL_reverse(p::Vector{Float64}, q::Vector{Float64})

Computes the reverse Kullback-Leibler divergence KL(q || p) between two discretized distributions.

KL(q || p) = Σᵢ q(xᵢ) log(q(xᵢ) / p(xᵢ))

This is the reverse KL divergence, which penalizes q for missing mass where p has mass.
It tends to produce moment-matching behavior (the approximation covers all modes).

# Arguments
- `p::Vector{Float64}`: Discretized target distribution (should sum to 1)
- `q::Vector{Float64}`: Discretized approximating distribution (should sum to 1)

# Returns
- `Float64`: The reverse KL divergence value (always non-negative)
"""
function KL_reverse(p::Vector{Float64}, q::Vector{Float64})
    return sum(q .* log.(q ./ p))
end

"""
    α_divergence(p::Vector{Float64}, q::Vector{Float64}; α=0.5)

Computes the α-divergence between two discretized probability distributions.

The α-divergence is a family of divergences parameterized by α:
- α = 1: Forward KL divergence KL(p || q)
- α = 0: Reverse KL divergence KL(q || p)
- α = 0.5: Hellinger distance
- α → ±∞: Different limiting behaviors

D_α(p || q) = (1 - Σᵢ q(xᵢ) (p(xᵢ)/q(xᵢ))^α) / (α(1-α))

# Arguments
- `p::Vector{Float64}`: Discretized target distribution (should sum to 1)
- `q::Vector{Float64}`: Discretized approximating distribution (should sum to 1)
- `α::Float64`: Divergence parameter (default: 0.5)

# Returns
- `Float64`: The α-divergence value
"""
function α_divergence(p::Vector{Float64}, q::Vector{Float64}; α=0.5)
    if α == 1
        return KL(p, q)
    elseif α == 0
        return KL_reverse(p, q)
    else
        return (1 - sum(q .* (p ./ q).^α)) / (α * (1 - α))
    end
end

"""
    mean_and_variance(mog::MixtureOfGaussian; n=10000)

Computes the true mean and variance of a mixture of Gaussians using numerical integration.

# Arguments
- `mog::MixtureOfGaussian`: The mixture of Gaussians
- `n::Int`: Number of discretization points (default: 10000)

# Returns
- `(μ, σ²)`: Tuple of mean and variance
"""
function mean_and_variance(mog::MixtureOfGaussian; n=10000)
    min_x, max_x = get_range(mog)
    xs = range(2*min_x, 2*max_x, length=n)
    p_mog = [mog_pdf(mog, x) for x in xs]
    p_mog = p_mog ./ sum(p_mog)

    μ = sum(xs .* p_mog)
    σ2 = sum((xs .- μ).^2 .* p_mog)
    return μ, σ2
end

"""
    closest_gaussian(mog::MixtureOfGaussian; n=10000, distance=KL)

Finds the Gaussian distribution that best approximates a mixture of Gaussians according
to a specified divergence measure.

The function performs a grid search over possible mean and variance parameters to find
the Gaussian that minimizes the specified divergence from the mixture.

# Arguments
- `mog::MixtureOfGaussian`: The mixture to approximate
- `n::Int`: Number of discretization points for integration (default: 10000)
- `distance::Function`: Divergence function to minimize, signature (p, q) -> Float64 (default: KL)

# Returns
- `Normal`: The optimal Gaussian approximation

# Example
```julia
mog = MixtureOfGaussian([0.5, 0.5], [Normal(0, 1), Normal(4, 1)])
# Mode-seeking (forward KL)
approx_fwd = closest_gaussian(mog, distance=KL)
# Moment-matching (reverse KL)
approx_rev = closest_gaussian(mog, distance=KL_reverse)
```
"""
function closest_gaussian(mog::MixtureOfGaussian; n = 10000, distance = KL)
    # Discretize the mixture distribution
    min_x, max_x = get_range(mog)
    xs = range(2*min_x, 2*max_x, length=n)
    p_mog = [mog_pdf(mog, x) for x in xs]
    p_mog = p_mog ./ sum(p_mog)

    # Compute initial estimates from moments
    μ = sum(xs .* p_mog)
    σ = sqrt(sum((xs .- μ).^2 .* p_mog))

    # Define search grid around initial estimates
    μs = range(μ / 1.5, μ*1.5, length=100)
    σs = range(σ / 2, σ * 2, length=100)

    # Grid search for optimal parameters
    smallest_distance = Inf
    best_μ = 0.0
    best_σ = 1.0

    for μ in μs
        for σ in σs
            p_normal = [pdf(Normal(μ, σ), x) for x in xs]
            p_normal = p_normal ./ sum(p_normal)
            d = distance(p_mog, p_normal)
            if d < smallest_distance
                smallest_distance = d
                best_μ = μ
                best_σ = σ
            end
        end
    end

    println("Optimal Gaussian: μ = $best_μ, σ² = $(best_σ^2)")

    return Normal(best_μ, best_σ)
end

"""
    plot_α_anim(mog; αs=range(-1.5, 1.5, length=30), anim_filename="~/Downloads/mog_anim.mp4", 
                μ_match_filename="~/Downloads/mog_μ_match.png", σ2_match_filename="~/Downloads/mog_σ2_match.png")

Generates an animation showing optimal Gaussian approximations under α-divergence for different
values of α, and creates plots comparing the approximation moments to the true moments.

This visualization demonstrates how different α values lead to different approximation behaviors:
- Negative α: Mode-covering behavior
- α = 0: Reverse KL (moment-matching)
- α = 1: Forward KL (mode-seeking)
- Positive α: Mode-seeking behavior

# Arguments
- `mog::MixtureOfGaussian`: The mixture to approximate
- `αs::AbstractRange`: Range of α values to animate (default: -1.5 to 1.5 in 30 steps)
- `anim_filename::String`: Path for saving animation (default: "~/Downloads/mog_anim.mp4")
- `μ_match_filename::String`: Path for mean comparison plot (default: "~/Downloads/mog_μ_match.png")
- `σ2_match_filename::String`: Path for variance comparison plot (default: "~/Downloads/mog_σ2_match.png")

# Output
- Saves animation showing approximations for each α value
- Saves plot comparing approximation means vs true mean
- Saves plot comparing approximation variances vs true variance
"""
function plot_α_anim(mog; αs=range(-1.5, 1.5, length=30), 
                     anim_filename="~/Downloads/mog_anim.mp4",
                     μ_match_filename="~/Downloads/mog_μ_match.png",
                     σ2_match_filename="~/Downloads/mog_σ2_match.png")
    # Compute true moments
    mog_μ, mog_σ2 = mean_and_variance(mog)

    # Storage for approximation moments
    μs = Vector{Float64}(undef, length(αs))
    σ2s = Vector{Float64}(undef, length(αs))
    
    # Generate animation over α values
    anim = @animate for (i, α) in enumerate(αs)
        best_approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=α))
        μs[i], σ2s[i] = best_approx.μ, best_approx.σ^2
        plot_mog(
            mog, 
            approx = best_approx, 
            title="α = $(round(α, digits=1))",
            ylim=(0, 0.32)
        )
    end
    mp4(anim, anim_filename, fps=10)
    
    # Plot mean comparison
    p = plot(
            αs, 
            μs, 
            linewidth=3, 
            color = :blue,
            label = L"\mu_{\mathrm{approximation}}",
            xlabel=L"\alpha", 
            ylabel=L"\mu_{\mathrm{approximation}}",
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
            legendfontsize=16,
        )
    plot!(
        αs, 
        mog_μ * ones(length(αs)), 
        linewidth=3,
        label = L"\mu_p",
        color = :red,
    )
    display(p)
    savefig(μ_match_filename)
    
    # Plot variance comparison
    p = plot(
            αs, 
            σ2s, 
            linewidth=3, 
            color = :blue,
            label=L"\sigma^2_{\mathrm{approximation}}",
            xlabel=L"\alpha", 
            ylabel=L"\sigma^2_{\mathrm{approximation}}",
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
            legendfontsize=16,
        )
    plot!(
        αs, 
        mog_σ2 * ones(length(αs)), 
        linewidth=3,
        label = L"\sigma^2_p",
        color = :red,
    )
    display(p)    
    savefig(σ2_match_filename)
end

"""
    main()

Main demonstration function that creates visualizations of Gaussian approximations to a
mixture of Gaussians under various α-divergences.

Generates:
- Animation showing approximations for α ∈ [-1, 2]
- Mean and variance comparison plots
- Individual approximation plots for α ∈ {-100, -1, 0, 1, 100}
"""
function main()
    # Create a three-component mixture
    mog = MixtureOfGaussian([0.3, 0.4, 0.3], [Normal(0, 0.5), Normal(3, 1), Normal(6, 1.5)])

    # Generate animation and moment-matching plots
    plot_α_anim(mog,
                αs = range(-1, 2, length=50),
                anim_filename="~/Downloads/mog_anim.mp4",
                μ_match_filename="~/Downloads/mog_μ_match.svg",
                σ2_match_filename="~/Downloads/mog_σ2_match.svg"
    )

    # Generate individual approximation plots for extreme and key α values
    plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=-100)))
    savefig("~/Downloads/mog_α_-100.svg")
    plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=-1)))
    savefig("~/Downloads/mog_α_-1.svg")
    plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=0)))
    savefig("~/Downloads/mog_α_0.svg")
    plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=1)))
    savefig("~/Downloads/mog_α_1.svg")
    plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=100)))
    savefig("~/Downloads/mog_α_100.svg")
end

end