"""
    GDExample

Module for comparing gradient descent optimization with message passing inference in factor graphs.

This module provides functions to:
- Implement gradient descent (GD) optimization for product factors with observed targets
- Implement full message passing (MP) inference using Gaussian approximations
- Visualize and compare convergence trajectories of GD vs MP in weight space
- Demonstrate the equivalence between optimization and inference perspectives

The module illustrates how gradient-based optimization and probabilistic inference via
message passing can solve the same problem (learning product factor weights from observations),
highlighting their different approaches and convergence properties.

Key Features:
- GD optimization minimizes squared error: min_{w1,w2} Σ(w1*w2 - y_i)²
- MP inference computes posteriors P(w1|data) and P(w2|data) via Gaussian message passing
- Side-by-side trajectory visualization in weight space with convergence plots
- Animation support for both methods to visualize the optimization/inference process

2026 by Ralf Herbrich
Hasso-Plattner Institute
"""
module GDExample

include("product.jl")
include("relu.jl")

using Distributions
using Plots
using Random
using Statistics
using LinearAlgebra
using LaTeXStrings
using .ProductFactorModule

"""
    Base.:*(d1::Normal{Float64}, d2::Normal{Float64})

Multiply two Gaussian distributions using precision-weighted combination.

This operator implements the product of two Gaussian densities, which is itself
a Gaussian. The operation is fundamental in probabilistic inference for combining
independent information sources or beliefs about a variable.

The precision-weighted formula ensures:
- The result has higher precision (lower variance) than either input
- The mean is a precision-weighted average of the input means
- The operation is commutative and associative

Mathematical formulation:
- Given N(μ₁, σ₁²) and N(μ₂, σ₂²)
- Precision: π_i = 1/σᵢ²
- Result: N(μ, σ²) where:
  * μ = (μ₁π₁ + μ₂π₂)/(π₁ + π₂)
  * σ² = 1/(π₁ + π₂)

# Arguments
- `d1::Normal{Float64}`: First Gaussian distribution
- `d2::Normal{Float64}`: Second Gaussian distribution

# Returns
- `Normal{Float64}`: Product distribution (also Gaussian)

# Example
```julia
prior = Normal(0.0, 1.0)
likelihood = Normal(2.0, 0.5)
posterior = prior * likelihood  # N(1.6, 0.447...)
```
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
    Base.:/(d1::Normal{Float64}, d2::Normal{Float64})

Divide two Gaussian distributions using precision-weighted combination.

This operator implements the division of two Gaussian densities, which corresponds
to removing information in probabilistic inference. It's the inverse operation of
multiplication and is used to compute cavity distributions in message passing algorithms.

The operation computes: d1 / d2 = N(μ, σ²) such that (d1 / d2) * d2 = d1

Mathematical formulation:
- Given N(μ₁, σ₁²) and N(μ₂, σ₂²)
- Precision: π_i = 1/σᵢ²
- Result: N(μ, σ²) where:
  * μ = (μ₁π₁ - μ₂π₂)/(π₁ - π₂)
  * σ² = 1/(π₁ - π₂)

# Arguments
- `d1::Normal{Float64}`: Numerator Gaussian distribution
- `d2::Normal{Float64}`: Denominator Gaussian distribution

# Returns
- `Normal{Float64}`: Quotient distribution (also Gaussian)

# Warning
The division can produce negative precision (π₁ - π₂ < 0) if d2 is more informative
than d1, which corresponds to an improper distribution. Use with caution.

# Example
```julia
posterior = Normal(1.6, 0.447)
likelihood = Normal(2.0, 0.5)
prior = posterior / likelihood  # Recovers the prior
```
"""
function Base.:/(d1::Normal{Float64}, d2::Normal{Float64})
    μ1, σ1 = mean(d1), std(d1)
    μ2, σ2 = mean(d2), std(d2)

    precision1 = 1.0 / σ1^2
    precision2 = 1.0 / σ2^2

    μ = (μ1 * precision1 - μ2 * precision2) / (precision1 - precision2)
    σ2 = 1.0 / (precision1 - precision2)

    return Normal(μ, sqrt(σ2))
end

"""
    gd(ys::Vector{Float64}, w1::Float64, w2::Float64; η=0.05, tol=1e-4, y=4, animation=nothing)

Perform gradient descent optimization for product factor y = w1 * w2 with observed targets.

This function implements batch gradient descent to find weights w1 and w2 that minimize
the mean squared error between the product w1*w2 and observed values ys. The optimization
proceeds iteratively, updating both weights simultaneously until convergence.

The objective function is:
  L(w1, w2) = (1/n) Σᵢ (w1*w2 - yᵢ)²

Gradient updates:
  ∂L/∂w1 = (2/n) Σᵢ (w1*w2 - yᵢ) * w2
  ∂L/∂w2 = (2/n) Σᵢ (w1*w2 - yᵢ) * w1

Weight updates:
  w1 ← w1 - η * ∂L/∂w1
  w2 ← w2 - η * ∂L/∂w2

Convergence criterion:
  max(|Δw1|, |Δw2|) < tol

# Arguments
- `ys::Vector{Float64}`: Vector of observed target values
- `w1::Float64`: Initial value for weight 1
- `w2::Float64`: Initial value for weight 2
- `η::Float64=0.05`: Learning rate (step size)
- `tol::Float64=1e-4`: Convergence tolerance (maximum weight change)
- `y::Float64=4`: Range for visualization (plots from -y to y)
- `animation::Union{String,Nothing}=nothing`: Path to save animation (MP4), or nothing to skip

# Returns
- `w1::Float64`: Final optimized value of weight 1
- `w2::Float64`: Final optimized value of weight 2
- `w1_history::Vector{Float64}`: Trajectory of w1 values during optimization
- `w2_history::Vector{Float64}`: Trajectory of w2 values during optimization
- `Δ_history::Vector{Float64}`: Maximum weight change at each iteration

# Side Effects
- If `animation` is not nothing, saves an MP4 animation of the optimization trajectory
- Displays plots during optimization if animation is enabled

# Example
```julia
# Generate synthetic data
y_true = 4.0
ys = rand(Normal(y_true, 0.1), 100)

# Run GD starting from (2, -1)
w1, w2, w1_hist, w2_hist, Δ_hist = gd(
    ys, 2.0, -1.0, 
    η=0.05, 
    tol=1e-4,
    animation="~/Downloads/gd_convergence.mp4"
)

println("Final: w1=", w1, " w2=", w2, " product=", w1*w2)
```
"""
function gd(
    ys::Vector{Float64},
    w1::Float64,
    w2::Float64;
    η = 0.05, 
    tol = 1e-4,
    y = 4,
    animation="~/Downloads/gd_product_factor.mp4",
)
    Δ_history = Float64[]
    w1_history = Float64[]
    w2_history = Float64[]

    anim = Animation()

    Δ = Inf
    while Δ > tol
        y_pred = w1 * w2

        # compute gradients
        grads_y_pred = y_pred .- ys
        grads_w1 = grads_y_pred .* w2
        grads_w2 = grads_y_pred .* w1

        # update weights
        w1_new = w1 - η * mean(grads_w1)
        w2_new = w2 - η * mean(grads_w2)

        # update the difference of means and histories
        Δ = max(abs(w1_new - w1), abs(w2_new - w2))

        # update the histories
        push!(Δ_history, Δ)
        push!(w1_history, w1)
        push!(w2_history, w2)

        # possibly add a frame for the animation
        if !isnothing(animation)
            plt = plot_gd_trajectory(w1_history, w2_history, Δ_history; y=y)
            frame(anim, plt)
        end

        # update weights for next iteration
        w1, w2 = w1_new, w2_new
    end

    # possibly save the animation
    if !isnothing(animation)
        mp4(anim, animation, fps = 2)
    end

    return w1, w2, w1_history, w2_history, Δ_history
end

"""
    plot_gd_trajectory(w1_history, w2_history, Δ_history; y=4)

Visualize gradient descent optimization trajectory in weight space with convergence plot.

Creates a two-panel figure showing:
1. Left panel: Convergence plot showing max(|Δw1|, |Δw2|) over iterations on log scale
2. Right panel: Contour plot of error surface with GD trajectory overlay

The contour plot shows the loss landscape L(w1,w2) = (1/2)(w1*w2 - y)² and the
path taken by gradient descent from initialization to convergence. The error surface
forms hyperbolic contours (xy = constant), which can make optimization challenging
depending on the learning rate.

# Arguments
- `w1_history::Vector{Float64}`: Trajectory of w1 values during optimization
- `w2_history::Vector{Float64}`: Trajectory of w2 values during optimization
- `Δ_history::Vector{Float64}`: Maximum weight change at each iteration
- `y::Float64=4`: Range for contour plot (from -y to y for both axes)

# Returns
- `Plot`: Combined plot object with convergence and trajectory visualizations

# Example
```julia
w1_hist = [2.0, 1.8, 1.6, ...]
w2_hist = [-1.0, -1.2, -1.4, ...]
Δ_hist = [0.2, 0.15, 0.1, ...]
plt = plot_gd_trajectory(w1_hist, w2_hist, Δ_hist, y=5)
display(plt)
savefig("~/Downloads/gd_trajectory.pdf")
```
"""
# plots a trajectory of the GD optimization in weight space along with the Δ history
function plot_gd_trajectory(w1_history, w2_history, Δ_history; y = 4)
    plt1 = plot(
        Δ_history,
        yscale=:log10, 
        color = :black,
        linewidth = 2,
        label = false,
        xlabel="Iteration", 
        ylabel=L"\max(|Δw|,|Δ\tilde{z}|)", 
        title="GD Convergence"
    )

    # draw a contour plot of the error surface 1/2 * (w1 * w2 - y)^2 from -y to y for w1 and -y to -y for w2
    w1_range = range(-y, y, length=100)
    w2_range = range(-y, y, length=100)
    Z = [0.5 * (w1 * w2 - y)^2 for w1 in w1_range, w2 in w2_range]
    plt2 = contour(
        w1_range,
        w2_range,
        Z',
        levels=20, 
        linewidth=0.5, 
        alpha=0.5,
        xlabel=L"w", 
        ylabel=L"\tilde{z}", 
        title="GD Trajectory in Weight Space"
    )
    plot!(plt2,
        w1_history, 
        w2_history, 
        color = :blue,
        linewidth = 1,
        label = false,
    )
    scatter!(plt2,
        [w1_history[end]], 
        [w2_history[end]], 
        label = false,
        markersize = 6,
        
    )
    plt = plot(plt1, plt2,         
        layout=(1,2), size=(900,400)
    )
    return plt
end

"""
    full_mp(ys::Vector{Float64}, w1::Normal{Float64}, w2::Normal{Float64}; σ_Y=0.1, tol=1e-6, animation=nothing, y=4)

Perform full message passing inference for product factor with observed targets.

This function implements iterative message passing on a factor graph to compute
posterior distributions P(w1|data) and P(w2|data) given observed values ys. The
factor graph structure is:

    W1 ──┐
         ├── Product Factor ── Y₁ (observed)
    W2 ──┘

    W1 ──┐
         ├── Product Factor ── Y₂ (observed)
    W2 ──┘
    ... (n observations)

Each observation yᵢ is connected to W1 and W2 through a product factor f(w1,w2,yᵢ)
with Gaussian noise: yᵢ ~ N(w1*w2, σ_Y²).

The algorithm:
1. Initialize messages from each product factor to W1 and W2
2. Iterate until convergence:
   - For each observation i:
     * Compute cavity distributions (remove message i from posteriors)
     * Recompute messages from factor i using Gaussian approximations
     * Update posteriors by multiplying in new messages
3. Return final posterior distributions

Convergence criterion:
  max(|Δμ_w1|, |Δμ_w2|) < tol

# Arguments
- `ys::Vector{Float64}`: Vector of observed target values
- `w1::Normal{Float64}`: Prior distribution for weight 1
- `w2::Normal{Float64}`: Prior distribution for weight 2
- `σ_Y::Float64=0.1`: Standard deviation of observation noise
- `tol::Float64=1e-6`: Convergence tolerance (maximum mean change)
- `animation::Union{String,Nothing}=nothing`: Path to save animation (MP4), or nothing to skip
- `y::Float64=4`: Range for visualization (plots from -y to y)

# Returns
- `p_W1::Normal{Float64}`: Posterior distribution for weight 1
- `p_W2::Normal{Float64}`: Posterior distribution for weight 2
- `w1_history::Vector{Normal{Float64}}`: Posterior trajectory for w1
- `w2_history::Vector{Normal{Float64}}`: Posterior trajectory for w2
- `Δ_history::Vector{Float64}`: Maximum mean change at each iteration

# Side Effects
- If `animation` is not nothing, saves an MP4 animation of the inference trajectory
- Displays plots during inference if animation is enabled

# Example
```julia
# Generate synthetic data
y_true = 4.0
ys = rand(Normal(y_true, 0.1), 100)

# Run message passing with Gaussian priors
prior_w1 = Normal(2.0, 0.7)
prior_w2 = Normal(-1.0, 0.7)
p_w1, p_w2, w1_hist, w2_hist, Δ_hist = full_mp(
    ys, prior_w1, prior_w2,
    σ_Y=0.1,
    tol=1e-6,
    animation="~/Downloads/mp_inference.mp4"
)

println("Posterior: w1 ~ N(\$(mean(p_w1)), \$(std(p_w1)))")
println("Posterior: w2 ~ N(\$(mean(p_w2)), \$(std(p_w2)))")
```
"""
# runs full message passing back-and-forth until convergence on the priors w1, w2 and the observed ys
function full_mp(
    ys::Vector{Float64}, 
    w1::Normal{Float64}, 
    w2::Normal{Float64};
    σ_Y = 0.1, 
    tol = 1e-6, 
    animation="~/Downloads/mp_product_factor.mp4",
    y = 4,
)
    Δ_history = Float64[]
    w1_history = Normal{Float64}[]
    w2_history = Normal{Float64}[]

    n = length(ys)
    msg_to_W1 = Vector{Normal{Float64}}(undef, n)
    msg_to_W2 = Vector{Normal{Float64}}(undef, n)

    # run one loop through all observed y[i] to initialize messages from Y_i to W_1 and W_2
    p_W1, p_W2 = w1, w2       # prior message to W_1 and
    for i in 1:n
        msg_from_Y = Normal(ys[i], σ_Y)
        msg_to_W1[i] = approximate_product_factor_msg_to_x(msg_from_Y, p_W2)
        msg_to_W2[i] = approximate_product_factor_msg_to_y(msg_from_Y, p_W1)
        p_W1 = msg_to_W1[i] * w1
        p_W2 = msg_to_W2[i] * w2
    end
    Δ = max(abs(mean(p_W1) - mean(w1)), abs(mean(p_W2) - mean(w2)))

    # update the history
    push!(w1_history, w1)
    push!(w2_history, w2)
    push!(Δ_history, Δ)

    # create an animation of the message passing process
    anim = Animation()
    
    # now iterate through all outgoing messages and remove them to refit them until convergence
    while Δ > tol
        p_W1_old, p_W2_old = p_W1, p_W2
        for i in 1:n
            msg_from_Y = Normal(ys[i], σ_Y)

            # compute the incoming message to the i-th product factor by removing the outgoing message from the 
            # product of all outgoing messages from the product factors and the prior factor
            p_W1_without_i = p_W1 / msg_to_W1[i]
            p_W2_without_i = p_W2 / msg_to_W2[i]

            # compute the next best approximation to the outgoing messages from the product factor with observed Y_i
            msg_to_W1[i] = approximate_product_factor_msg_to_x(msg_from_Y, p_W2_without_i)
            msg_to_W2[i] = approximate_product_factor_msg_to_y(msg_from_Y, p_W1_without_i)

            # update the marginals by multiplying in the new outgoing messages
            p_W1_new = msg_to_W1[i] * p_W1
            p_W2_new = msg_to_W2[i] * p_W2

            # update the marginals after checking for convergence
            p_W1 = p_W1_new
            p_W2 = p_W2_new
        end
        Δ = max(abs(mean(p_W1) - mean(p_W1_old)), abs(mean(p_W2) - mean(p_W2_old)))

        # add a frame for the animation
        if !isnothing(animation)
            plt = plot_mp_trajectory(w1_history, w2_history, Δ_history; y=y)
            frame(anim, plt)
        end

        # update the history
        push!(w1_history, p_W1)
        push!(w2_history, p_W2)
        push!(Δ_history, Δ)
    end

    # possibly save the animation
    if !isnothing(animation)
        mp4(anim, animation, fps = 2)
    end

    return p_W1, p_W2, w1_history, w2_history, Δ_history
end

"""
    plot_mp_trajectory(w1_history, w2_history, Δ_history; y=4)

Visualize message passing inference trajectory in weight space with convergence plot.

Creates a two-panel figure showing:
1. Left panel: Convergence plot showing max(|Δμ_w1|, |Δμ_w2|) over iterations on log scale
2. Right panel: Contour plot of error surface with MP trajectory and uncertainty ellipse

The trajectory shows the evolution of posterior means during message passing. The final
uncertainty is visualized as a 3σ ellipse around the converged posterior mean, showing
the covariance structure. Unlike GD which converges to a point estimate, MP provides
a full posterior distribution capturing uncertainty.

# Arguments
- `w1_history::Vector{Normal{Float64}}`: Posterior trajectory for w1 during inference
- `w2_history::Vector{Normal{Float64}}`: Posterior trajectory for w2 during inference
- `Δ_history::Vector{Float64}`: Maximum mean change at each iteration
- `y::Float64=4`: Range for contour plot (from -y to y for both axes)

# Returns
- `Plot`: Combined plot object with convergence and trajectory visualizations

# Example
```julia
w1_hist = [Normal(2.0, 0.7), Normal(1.9, 0.6), ...]
w2_hist = [Normal(-1.0, 0.7), Normal(-1.1, 0.6), ...]
Δ_hist = [0.15, 0.1, 0.05, ...]
plt = plot_mp_trajectory(w1_hist, w2_hist, Δ_hist, y=5)
display(plt)
savefig("~/Downloads/mp_trajectory.pdf")
```
"""
# plots a trajectory of the MP optimization in weight space along with the Δ history
function plot_mp_trajectory(w1_history, w2_history, Δ_history; y = 4)
    plt1 = plot(
        Δ_history,
        yscale=:log10, 
        color = :black,
        linewidth = 2,
        label = false,
        xlabel="Iteration", 
        ylabel=L"\max(|\Delta \mu|)", 
        title="MP Convergence"
    )

    # draw a contour plot of the error surface 1/2 * (w1 * w2 - y)^2 from -y to y for w1 and -y to -y for w2
    w1_range = range(-y, y, length=100)
    w2_range = range(-y, y, length=100)
    Z = [0.5 * (w1 * w2 - y)^2 for w1 in w1_range, w2 in w2_range]
    plt2 = contour(
        w1_range,
        w2_range,
        Z',
        levels=20, 
        linewidth=0.5, 
        alpha=0.5,
        xlabel=L"\mu_{W}",
        ylabel=L"\mu_{\tilde{Z}}", 
        title="MP Trajectory in Weight Space"
    )

    plot!(
        plt2,
        [mean(w1) for w1 in w1_history], 
        [mean(w2) for w2 in w2_history], 
        color = :blue,
        linewidth = 2,
        label = false,
    )

    # draw an axis aligned ellipse with the standard deviations of the last w1 and w2 at the location of the means
    w1_mean = mean(w1_history[end])
    w2_mean = mean(w2_history[end])
    w1_std = 3*std(w1_history[end])
    w2_std = 3*std(w2_history[end])
    θ = range(0, 2π, length=100)
    x_ellipse = w1_mean .+ w1_std .* cos.(θ)
    y_ellipse = w2_mean .+ w2_std .* sin.(θ)
    plot!(
        plt2,
        Shape(x_ellipse, y_ellipse),
        color = :red, 
        fillalpha = 0.5,
        label = false,
    )
    plot!(
        plt2,
        x_ellipse,
        y_ellipse,
        color = :black,
        linewidth = 1,
        label = false,
    )

    plt = plot(plt1, plt2,         
        layout=(1,2), size=(900,400)
    )
    return plt
end

"""
    simulate(; μ_W1=2.0, σ_W1=0.7, μ_W2=-1.0, σ_W2=0.7, μ_Y=4.0, σ_Y=0.1, n=100, η=0.05, tol=1e-4, animation_gd_file=nothing, animation_mp_file=nothing)

Run a complete simulation comparing gradient descent and message passing approaches.

This function generates synthetic observations from a product factor model and then
solves the inference/optimization problem using both:
1. Gradient Descent (GD): Point estimation via loss minimization
2. Message Passing (MP): Posterior inference via probabilistic message passing

The generative model is:
  W1 ~ N(μ_W1, σ_W1²)    [true weight 1]
  W2 ~ N(μ_W2, σ_W2²)    [true weight 2]
  Yᵢ ~ N(W1*W2, σ_Y²)    [noisy observations of product]

Both methods aim to recover the product W1*W2 ≈ μ_Y from the observations.

Key differences:
- GD: Finds point estimates (w1*, w2*) that minimize squared error
- MP: Computes full posteriors P(W1|Y) and P(W2|Y) capturing uncertainty
- GD: Faster convergence but no uncertainty quantification
- MP: Slower convergence but provides confidence intervals

# Arguments
- `μ_W1::Float64=2.0`: Prior mean for weight 1 (also GD initialization)
- `σ_W1::Float64=0.7`: Prior standard deviation for weight 1
- `μ_W2::Float64=-1.0`: Prior mean for weight 2 (also GD initialization)
- `σ_W2::Float64=0.7`: Prior standard deviation for weight 2
- `μ_Y::Float64=4.0`: Mean of observation distribution (true product)
- `σ_Y::Float64=0.1`: Standard deviation of observation noise
- `n::Int=100`: Number of observations to generate
- `η::Float64=0.05`: Learning rate for gradient descent
- `tol::Float64=1e-4`: Convergence tolerance for both methods
- `animation_gd_file::Union{String,Nothing}=nothing`: Path for GD animation (MP4)
- `animation_mp_file::Union{String,Nothing}=nothing`: Path for MP animation (MP4)

# Returns
Nothing. Displays plots and prints results to console.

# Side Effects
- Generates synthetic data from Normal(μ_Y, σ_Y)
- Runs GD and MP algorithms
- Displays trajectory and convergence plots for both methods
- Prints final results to console
- Optionally saves animations

# Console Output
- GD results: Final weights and product value
- MP results: Posterior means, standard deviations, and product value

# Example
```julia
# Run comparison with default parameters
simulate()

# Run with custom parameters and save animations
simulate(
    μ_W1=1.5, σ_W1=1.0,
    μ_W2=-2.0, σ_W2=0.5,
    n=200, η=0.01,
    animation_gd_file="~/Downloads/gd_custom.mp4",
    animation_mp_file="~/Downloads/mp_custom.mp4"
)
```
"""
function simulate(;
    μ_W1 = 2.0,
    σ_W1 = 0.7,
    μ_W2 = -1.0,
    σ_W2 = 0.7,
    μ_Y = 4.0,
    σ_Y = 0.1,
    n = 100,
    η = 0.05, 
    tol = 1e-4,
    animation_gd_file = nothing,
    animation_mp_file = nothing,
)
    y_samples = rand(Normal(μ_Y, σ_Y), n)

    w1, w2, w1_history, w2_history, Δ_history = gd(
        y_samples,
        μ_W1,
        μ_W2, 
        η = η,        
        tol = tol, 
        y = μ_Y,
        animation=animation_gd_file,
    )   
    println("GD Result: w1 = $(w1), w2 = $(w2) (w1 * w2 = $(w1 * w2))")
    plt = plot_gd_trajectory(w1_history, w2_history, Δ_history, y=μ_Y)
    display(plt)

    w1, w2, w1_history, w2_history, Δ_history = full_mp(
        y_samples,
        Normal(μ_W1, σ_W1),
        Normal(μ_W2, σ_W2),
        σ_Y = σ_Y,
        tol = tol,
        y = μ_Y,
        animation=animation_mp_file
    )
    println("Message Passing Result: w1 = $(mean(w1)) ± $(std(w1)), w2 = $(mean(w2)) ± $(std(w2)) (w1 * w2 = $(mean(w1) * mean(w2)))")
    plt = plot_mp_trajectory(w1_history, w2_history, Δ_history, y=μ_Y)
    display(plt)
end

"""
    main()

Main entry point for the GDExample module demonstrations.

This function runs a comprehensive suite of experiments comparing gradient descent (GD)
and message passing (MP) approaches to parameter learning in product factor models.
Three experiments are conducted with different learning rates to illustrate:

1. Effect of learning rate on GD convergence speed and trajectory
2. Comparison between point estimation (GD) and posterior inference (MP)
3. Uncertainty quantification available in MP but not in GD

Experiments:
- Experiment 1: Low learning rate (η=0.01) - slow but stable convergence
- Experiment 2: Medium learning rate (η=0.05) - balanced convergence
- Experiment 3: High learning rate (η=0.1) - fast but potentially unstable

All experiments use:
- Prior/Init: W1 ~ N(2.0, 0.7), W2 ~ N(-1.0, 0.7)
- Target: Y ~ N(4.0, 0.1) with 100 observations
- Convergence tolerance: 10⁻⁴

# Output Files
Creates animation files in ~/Downloads/:
- gd_product_factor_η0.01.mp4: GD trajectory with η=0.01
- gd_product_factor_η0.05.mp4: GD trajectory with η=0.05
- gd_product_factor_η0.1.mp4: GD trajectory with η=0.1
- mp_product_factor.mp4: MP trajectory (same for all experiments)

# Console Output
For each experiment, prints:
- Configuration details (priors, learning rate, sample size)
- GD results: Final point estimates and product value
- MP results: Posterior distributions with means, standard deviations, and product
- Convergence information

# Side Effects
- Displays trajectory and convergence plots for each experiment
- Saves animation files to ~/Downloads/
- Prints extensive diagnostic output to console
"""
function main()
    Random.seed!(42)
    
    println("\n" * "="^80)
    println("Gradient Descent vs Message Passing - Comparative Analysis")
    println("="^80)
    println("\n📊 Product Factor Model: Y = W1 × W2")
    println("-"^80)
    println("This module demonstrates two approaches to parameter learning:")
    println("  1. Gradient Descent (GD): Point estimation via loss minimization")
    println("  2. Message Passing (MP): Posterior inference via probabilistic inference")
    println("\nKey Differences:")
    println("  • GD: Finds optimal weights (w1*, w2*) minimizing squared error")
    println("  • MP: Computes full posteriors P(W1|data), P(W2|data) with uncertainty")
    println("\nExperiments: Comparing different learning rates for GD")
    println("="^80)

    # Experiment 1: Low learning rate
    println("\n" * "="^80)
    println("📊 EXPERIMENT 1: Low Learning Rate (η = 0.01)")
    println("="^80)
    println("Configuration:")
    println("  • Prior/Init W1 ~ N(2.0, 0.7)")
    println("  • Prior/Init W2 ~ N(-1.0, 0.7)")
    println("  • Target Y ~ N(4.0, 0.1)  [E[W1×W2] ≈ 4.0]")
    println("  • Number of observations: 100")
    println("  • GD Learning rate: η = 0.01")
    println("  • Convergence tolerance: 10⁻⁴")
    println("\nExpected behavior: Slow but stable convergence with smooth trajectory")
    println("-"^80)
    
    simulate(μ_W1 = 2.0, σ_W1 = 0.7, μ_W2 = -1.0, σ_W2 = 0.7, μ_Y = 4.0, σ_Y = 0.1, n = 100, η=0.01, tol = 1e-4, 
        animation_gd_file="~/Downloads/gd_product_factor_η0.01.mp4",
        animation_mp_file="~/Downloads/mp_product_factor_η0.01.mp4",
    )
    
    println("\n✓ Experiment 1 completed")
    println("  Saved: gd_product_factor_η0.01.mp4, mp_product_factor.mp4")

    # Experiment 2: Medium learning rate
    println("\n" * "="^80)
    println("📊 EXPERIMENT 2: Medium Learning Rate (η = 0.05)")
    println("="^80)
    println("Configuration:")
    println("  • Prior/Init W1 ~ N(2.0, 0.7)")
    println("  • Prior/Init W2 ~ N(-1.0, 0.7)")
    println("  • Target Y ~ N(4.0, 0.1)  [E[W1×W2] ≈ 4.0]")
    println("  • Number of observations: 100")
    println("  • GD Learning rate: η = 0.1")
    println("  • Convergence tolerance: 10⁻⁴")
    println("\nExpected behavior: Balanced convergence speed and stability")
    println("-"^80)
    
    simulate(μ_W1 = 2.0, σ_W1 = 0.7, μ_W2 = -1.0, σ_W2 = 0.7, μ_Y = 4.0, σ_Y = 0.1, n = 100, η=0.1, tol = 1e-4, 
        animation_gd_file="~/Downloads/gd_product_factor_η0.1.mp4",
        animation_mp_file="~/Downloads/mp_product_factor.mp4",
    )
    
    println("\n✓ Experiment 2 completed")
    println("  Saved: gd_product_factor_η0.05.mp4, mp_product_factor.mp4")

    # Experiment 3: High learning rate
    println("\n" * "="^80)
    println("📊 EXPERIMENT 3: High Learning Rate (η = 0.2)")
    println("="^80)
    println("Configuration:")
    println("  • Prior/Init W1 ~ N(2.0, 0.7)")
    println("  • Prior/Init W2 ~ N(-1.0, 0.7)")
    println("  • Target Y ~ N(4.0, 0.1)  [E[W1×W2] ≈ 4.0]")
    println("  • Number of observations: 100")
    println("  • GD Learning rate: η = 0.2")
    println("  • Convergence tolerance: 10⁻⁴")
    println("\nExpected behavior: Fast convergence but potentially oscillatory")
    println("-"^80)
    
    simulate(μ_W1 = 2.0, σ_W1 = 0.7, μ_W2 = -1.0, σ_W2 = 0.7, μ_Y = 4.0, σ_Y = 0.1, n = 100, η=0.2, tol = 1e-4, 
        animation_gd_file="~/Downloads/gd_product_factor_η0.2.mp4",
        animation_mp_file="~/Downloads/mp_product_factor.mp4",
    )
    
    println("\n✓ Experiment 3 completed")
    println("  Saved: gd_product_factor_η0.2.mp4, mp_product_factor.mp4")

    println("\n" * "="^80)
    println("✅ All GD vs MP experiments completed successfully!")
    println("="^80)
    println("\n📁 Output Location: ~/Downloads/")
    println("\nGenerated Files:")
    println("  Gradient Descent Animations:")
    println("    • gd_product_factor_η0.01.mp4  (Low learning rate)")
    println("    • gd_product_factor_η0.1.mp4  (Medium learning rate)")
    println("    • gd_product_factor_η0.2.mp4   (High learning rate)")
    println("  Message Passing Animations:")
    println("    • mp_product_factor.mp4")
    println("\n💡 Key Insights:")
    println("  • GD Trajectories: Point estimates following gradient flow in weight space")
    println("  • MP Trajectories: Evolution of posterior means with uncertainty ellipses")
    println("  • Learning Rate: Controls GD convergence speed vs stability trade-off")
    println("  • Uncertainty: MP provides confidence intervals; GD gives point estimates only")
    println("  • Convergence: Both methods reach similar solutions but different representations")
    println("\n🎯 Comparison:")
    println("  • GD is faster per iteration but lacks uncertainty quantification")
    println("  • MP is slower but provides full posterior distributions")
    println("  • Both solve the same problem from different perspectives:")
    println("    - GD: Optimization view (minimize loss)")
    println("    - MP: Inference view (compute posterior)")
    println("\n" * "="^80 * "\n")
end

end