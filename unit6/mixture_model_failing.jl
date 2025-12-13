"""
    MixtureExample

Educational demonstrations showing why approximating a mixture distribution
by matching only its first two moments (mean and variance) can fail.

The file contains:
- A deterministic mixture factor: y = mixture(x1, x2, s) where s ∈ {0,1}.
- Monte Carlo samplers visualizing the message to y (unconditional) and the
  posterior-weighted marginals (when y has a prior).
- Diagnostics printed to the console: empirical mean/variance and effective
  sample size (ESS) for importance-weighted estimates.
- Plotting utilities that overlay analytic/empirical densities and a
  moment-matched Gaussian for comparison.

Pedagogical points:
- Mixtures can be multimodal; a single Gaussian approximation via moment-matching
  removes multimodality and may mislead inference.
- Importance weights (e.g., induced by a prior on y) can drastically reweight
  which component contributes to posterior marginals; ESS quantifies weight variance.
- Use Monte Carlo visualizations to detect failure modes of mean-field or moment-matching approximations.
"""
module MixtureExample

using LaTeXStrings
using Distributions
using Statistics
using Random
using Plots

# implements the mixture factor
"""
    mixture(x1, x2, s) -> Real

Deterministic mixture factor: returns x1 when s == 0, otherwise x2.

Arguments
- `x1`, `x2`: numeric values (component values)
- `s::Integer`: selector in {0,1}

Returns
- The selected component value for y.

Notes
- This function models a standard two-component discrete mixture.
- It is pure deterministic selection; randomness arises from sampling x1,x2,s externally.
"""
function mixture(x1, x2, s)
    return (s == 0) ? x1 : x2
end

"""
    sample_mixture_message(; μ_x1=0.0, σ_x1=1.0, μ_x2=6.0, σ_x2=1.0,
                            p=0.5, n_samples=1_000_000, filename="...")

Draw Monte‑Carlo samples of the message sent to the output variable y produced
by the mixture factor (i.e., draw x1,x2,s from their priors and compute y).

Keywords
- `μ_x1, σ_x1`: Normal parameters for x1
- `μ_x2, σ_x2`: Normal parameters for x2
- `p`: P(s == 1) (probability of selecting x2)
- `n_samples`: number of Monte Carlo draws
- `filename`: path to save the histogram

Behavior
- Plots an empirical histogram of y and overlays any analytic mixture density if desired.
- Prints empirical mean/variance for quick diagnostics.

Pedagogical use
- Visualize the marginal shape of y (often multimodal) before any conditioning.
"""
function sample_mixture_message(;μ_x1 = 0.0, σ_x1=1.0, μ_x2=6.0, σ_x2=1.0, p=0.5, n_samples=1000000, filename="~/Downloads/mixture_message_histogram.png")
    # stores the samples of y
    samples_msg_to_y = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        x1 = randn() * σ_x1 + μ_x1
        x2 = randn() * σ_x2 + μ_x2
        s = rand() < p ? 1 : 0
        # compute y using the mixture factor
        samples_msg_to_y[i] = mixture(x1, x2, s)
    end

    # plot histogram of samples of y
    plt = histogram(samples_msg_to_y,
        normalize = :pdf,
        alpha=0.5,
        color=:blue,
        xlabel=L"y",
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
    sample_mixture(; μ_x1=-3.0, σ_x1=1.0, μ_x2=5.0, σ_x2=1.0,
                    μ_y=1.0, σ_y=1.0, p=0.5, n_samples=1_000_000)

Demonstration sampler for the full factor graph with an additional prior on y.

Process
1. Sample x1 ~ N(μ_x1, σ_x1^2), x2 ~ N(μ_x2, σ_x2^2), s ~ Bernoulli(p).
2. Compute y = mixture(x1, x2, s).
3. Sample y_prior ~ N(μ_y, σ_y^2) and compute importance weights w = p(y_prior | y_sample).
4. Produce plots:
   - Prior y histogram vs posterior-weighted mixture samples
   - Prior vs posterior-weighted histograms for x1 and x2
   - Prior vs posterior-weighted bar chart for s

Diagnostics
- Prints nothing by default, but shows how weighting shifts marginals.
- Use to illustrate how conditioning on y (via a prior/likelihood) selects components.

Notes
- Weights are unnormalized pdf-values of y under its prior; these serve as importance weights.
- For interpretability, inspect the histograms and computed fractions for s.
"""
function sample_mixture(;μ_x1 = -3.0, σ_x1=1.0, μ_x2=5.0, σ_x2=1.0, μ_y=1.0, σ_y=1.0, p=0.5, n_samples=1000000)
    # stores the samples of y
    samples_y_posterior = Vector{Float64}(undef, n_samples)
    samples_y_prior = Vector{Float64}(undef, n_samples)
    samples_x1 = Vector{Float64}(undef, n_samples)
    samples_x2 = Vector{Float64}(undef, n_samples)
    samples_s = Vector{Int}(undef, n_samples)
    weights = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        # sample x from its Gaussian distribution
        samples_x1[i] = randn() * σ_x1 + μ_x1
        samples_x2[i] = randn() * σ_x2 + μ_x2
        # sample s from its Bernoulli distribution
        samples_s[i] = rand() < p ? 1 : 0
        # compute y using the mixture factor
        samples_y_posterior[i] = mixture(samples_x1[i], samples_x2[i], samples_s[i])
        # sample y from its prior distribution
        samples_y_prior[i] = randn() * σ_y + μ_y
        # compute the weight of the sample based on the prior of y
        weights[i] = pdf(Normal(μ_y, σ_y), samples_y_posterior[i])
    end

    # plot histogram of samples of y
    plt = histogram(samples_y_prior,
        normalize = :pdf,
        alpha=0.5,
        color=:blue,
        xlabel=L"y",
        label=false,
        xtickfontsize=18,
        ytickfontsize=18,
        xguidefontsize=20,
        yguidefontsize=20,
        legendfontsize=20,
    )
    histogram!(samples_y_posterior,
        normalize = :pdf,
        alpha=0.5,
        weights=weights,
        color=:red,
        label=false,
    )
    display(plt)

    # plot histogram of samples of x1
    plt = histogram(samples_x1,
        normalize = :pdf,
        alpha=0.5,
        color=:blue,
        xlabel=L"x_1",
        label=false,
        xtickfontsize=18,
        ytickfontsize=18,
        xguidefontsize=20,
        yguidefontsize=20,
        legendfontsize=20,
    )
    histogram!(samples_x1,
        normalize = :pdf,
        alpha=0.5,
        weights=weights,
        color=:red,
        label=false,
    )
    display(plt)

    # plot histogram of samples of x2
    plt = histogram(samples_x2,
        normalize = :pdf,
        alpha=0.5,
        color=:blue,
        xlabel=L"x_2",
        label=false,
        xtickfontsize=18,
        ytickfontsize=18,
        xguidefontsize=20,
        yguidefontsize=20,
        legendfontsize=20,
    )
    histogram!(samples_x2,
        normalize = :pdf,
        alpha=0.5,
        weights=weights,
        color=:red,
        label=false,
    )
    display(plt)

    # plot histogram of samples of s
    s_zero_frac_prior = sum(samples_s .== 0) / n_samples
    s_one_frac_prior = sum(samples_s .== 1) / n_samples
    plt = plot(
        bar([0, 1], [s_zero_frac_prior, s_one_frac_prior], alpha=0.5, bar_width = 0.75, color=:blue),
        legend=false,
        label=false,
        xtickfontsize=18,
        ytickfontsize=18,
        xguidefontsize=20,
        yguidefontsize=20,
        legendfontsize=20,
    )

    s_zero_frac = sum(weights[samples_s .== 0]) / sum(weights)
    s_one_frac = sum(weights[samples_s .== 1]) / sum(weights)
    bar!([0, 1], [s_zero_frac, s_one_frac], alpha=0.5, bar_width = 0.75, color=:red)
    display(plt)
end     

"""
    compute_approximate_marginal_y(; μ_x1=0.0, σ_x1=1.0, μ_x2=6.0, σ_x2=1.0,
                                   μ_y=3.0, σ_y=1.0, p=0.5, n_samples=1_000_000,
                                   base="~/Downloads/failing_mixture_model.svg")

Approximates the marginal p(y) induced by the mixture factor using importance sampling,
and overlays a moment-matched Gaussian approximation.

Process
- Draw mixture samples y_i by sampling x1,x2,s.
- Compute weights w_i = pdf(N(μ_y,σ_y), y_i) (likelihood under the prior on y).
- Compute weighted empirical mean and variance, print them.
- Compute analytic mixture moments (unconditioned) for reference.
- Plot prior, weighted posterior histogram and a Gaussian with the weighted mean/variance.

Returns
- (mean_y, var_y) — weighted empirical estimates
- Optionally saves the figure to `base`.

Pedagogical notes
- Compare the weighted empirical distribution (red) to the single Gaussian (moment-matched).
- Observe situations where the Gaussian misses multimodal structure and misleads inference.
"""
function compute_approximate_marginal_y(;μ_x1 = 0.0, σ_x1=1.0, μ_x2=6.0, σ_x2=1.0, μ_y=3.0, σ_y=1.0, p=0.50, n_samples=1000000, base="~/Downloads/failing_mixture_model.svg")
    # stores the samples of y
    samples_posterior_y = Vector{Float64}(undef, n_samples)
    samples_prior_y = Vector{Float64}(undef, n_samples)
    weights = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        # sample x from its Gaussian distribution
        x1 = randn() * σ_x1 + μ_x1
        x2 = randn() * σ_x2 + μ_x2
        # sample s from its Bernoulli distribution
        s = rand() < p ? 1 : 0
        # compute y using the mixture factor
        samples_posterior_y[i] = mixture(x1, x2, s)
        samples_prior_y[i] = randn() * σ_y + μ_y
        # compute the weight of the sample based on the prior of y
        weights[i] = pdf(Normal(μ_y, σ_y), samples_posterior_y[i])
    end

    # plot histogram of samples of y
    plt = histogram(samples_prior_y,
        normalize = :pdf,
        alpha=0.5,
        color=:blue,
        label=false,
        xlabel=L"y",
        xtickfontsize=18,
        ytickfontsize=18,
        xguidefontsize=20,
        yguidefontsize=20,
        legendfontsize=20,
    )

    histogram!(samples_posterior_y,
        normalize = :pdf,
        alpha=0.5,
        weights=weights,
        color=:red,
        label=false,
    )

    # output the empirical mean and variance of y based on the weighted samples
    mean_y = sum(weights .* samples_posterior_y) / sum(weights)
    var_y = sum(weights .* (samples_posterior_y .- mean_y).^2) / sum(weights)
    println("Empirical Mean of y: ", mean_y)
    println("Empirical Variance of y: ", var_y)

    # plot the moment-matched Gaussian approximation
    x_range = range(minimum(samples_posterior_y), stop=maximum(samples_posterior_y), length=300)
    y_range = pdf(Normal(mean_y, sqrt(var_y)), x_range)
    plot!(x_range, y_range, label=false, color=:red, linewidth=3)

    # plot the prior of y for comparison
    y_prior_range = pdf(Normal(μ_y, σ_y), x_range)
    plot!(x_range, y_prior_range, label=false, color=:blue, linewidth=3)

    display(plt)
    savefig(plt, base)
end     


"""
    main()

Run a default demonstration:
- Generate and save message histograms for representative mixture settings.
- Compute and save the weighted-marginal visualization that demonstrates failure of moment-matching.

Notes
- The function uses a random seed for reproducibility.
- Default sample counts are large for smooth plots; reduce n_samples when iterating interactively.
"""
function main()
    Random.seed!(42)
    sample_mixture_message(μ_x1=0.0, σ_x1=1.0, μ_x2=6.0, σ_x2=1.0, p=0.5, n_samples=1000000, filename="~/Downloads/mixture_message_histogram_0_6_0.5.svg")
    sample_mixture_message(μ_x1=0.0, σ_x1=1.0, μ_x2=6.0, σ_x2=1.0, p=0.75, n_samples=1000000, filename="~/Downloads/mixture_message_histogram_0_6_0.75.svg")
    compute_approximate_marginal_y(μ_x1=0.0, σ_x1=1.0, μ_x2=6.0, σ_x2=1.0, μ_y=3.0, σ_y=1.0, p=0.50, n_samples=1000000, base="~/Downloads/failing_mixture_model_0_6_0.5.svg")
end

end
