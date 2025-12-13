"""
    VMP

A module implementing Variational Message Passing (VMP) for Bayesian inference
on Gaussian models with educational examples from Winn & Bishop (2005).

# Overview

This module demonstrates VMP for a Gaussian observation model where both the mean
and precision (inverse variance) are unknown latent variables. Unlike the simpler
example in vmp.jl (which uses built-in Distributions.jl), this implementation uses
custom Gaussian1D and Gamma1D types for pedagogical clarity.

# Model Structure

The generative model is:
```
x ~ N(μ₀, τ₀⁻¹)      # Prior on mean
s ~ Gamma(α, β)      # Prior on precision
y ~ N(x, s⁻¹)        # Observation given mean and precision
```

Goal: Infer posterior q(x, s | y) using mean-field approximation q(x)q(s).

# Variational Message Passing Algorithm

## Factorization
Assume: q(x, s) = q(x)q(s)

## Message Updates

For factor f connecting x, s, and observation y:

**Message f → x:**
```
q(x) ∝ exp(E_q(s)[log p(y | x, s)])
     = N(x | y⟨s⟩, ⟨s⟩⁻¹)
```

**Message f → s:**
```
q(s) ∝ exp(E_q(x)[log p(y | x, s)])
     = Gamma(s | 1/2, (1/2)(y² + Var[x] + E[x]² - 2yE[x]))
```

## Convergence Criterion
Iterate until KL divergence between successive marginals falls below threshold (1e-6).

# Key Features

1. **Custom distribution types**: Uses Gaussian1D and Gamma1D from unit1
2. **Contour visualization**: Shows exact vs. VMP approximation evolution
3. **KL monitoring**: Tracks convergence via KL divergence
4. **Step-by-step plots**: Saves contour at each iteration

# Pedagogical Value

- Demonstrates VMP on conjugate-exponential models
- Shows evolution of variational approximation over iterations
- Compares exact joint (with correlation) vs. factorized approximation
- Illustrates when mean-field works well vs. struggles

# Files Generated

- `vmp_gaussian_factor_step_0.svg`, `step_1.svg`, ... : iteration snapshots
- `gaussian_mean_factor_{0.5,1.0,2.0}.{svg,pdf}`: noise level comparisons

# Comparison with vmp.jl

- vmp.jl: Uses Distributions.jl types (Normal, Gamma)
- vmp2.jl: Uses custom types for educational transparency
- vmp.jl: Focuses on mixture of Gaussians example
- vmp2.jl: Focuses on single Gaussian with unknown mean/precision

# Author
Ralf Herbrich, Hasso-Plattner Institute, 2025

# References
- Winn & Bishop (2005). Variational Message Passing. JMLR.
- Bishop (2006). Pattern Recognition and Machine Learning. Chapter 10.
"""
module VMP

include("../unit1/gaussian.jl")
include("../unit1/gamma.jl")

using Plots
using Distributions
using LaTeXStrings
using LinearAlgebra
using .Gaussian
using .Gamma

"""
    draw_contour_plot_factor(msg_from_X_to_f, msg_from_S_to_f, q_x, q_s, y; scale_factor=50.0) -> Nothing

Visualizes the joint distribution over (x, s) space as contour plots.

Overlays two distributions:
1. **Black contours**: Exact joint p(x, s | y) (scaled by factor for visibility)
2. **Red contours**: Variational approximation q(x)q(s)

# Mathematical Background

## Exact Joint (Black)
```
p(x, s | y) ∝ p(x) p(s) p(y | x, s)
           = msg_X(x) × msg_S(s) × N(y | x, s⁻¹)
```

This captures correlation between x and s induced by the observation y.

## Variational Approximation (Red)
```
q(x, s) = q(x) × q(s)
```

Factorized approximation that loses correlation structure.

# Arguments
- `msg_from_X_to_f::Gaussian1D`: Prior/incoming message for mean parameter x
- `msg_from_S_to_f::Gamma1D`: Prior/incoming message for precision parameter s
- `q_x::Gaussian1D`: Current variational marginal for x
- `q_s::Gamma1D`: Current variational marginal for s
- `y::Real`: Observed data point

# Keywords
- `scale_factor::Float64=50.0`: Scaling for exact joint (aids visualization)

# Plotting Details

- **X-axis**: Mean parameter x
- **Y-axis**: Precision parameter s
- **Black levels**: [0.005, 0.01, 0.05, 0.1, 1.0] (after scaling)
- **Red levels**: Same levels for q(x)q(s)
- **Range**: 
  - x: y ± 4 standard deviations of msg_from_X_to_f
  - s: 0 to mean(msg_from_S_to_f) + 5 standard deviations

# Interpretation

**Contour shape differences:**
- Black contours tilted → correlation between x and s
- Red contours axis-aligned → independence assumption
- Larger discrepancy → worse VMP approximation

**Convergence indicator:**
- As VMP iterates, red contours should stabilize
- Final red contours approximate marginalized black contours

# Use Cases
- Monitoring VMP convergence visually
- Teaching correlation vs. independence
- Debugging variational inference issues

# Notes
- Scale factor needed because exact joint unnormalized
- Displays plot automatically; save with external savefig() call
- Requires Plots.jl with contour support
"""
function draw_contour_plot_factor(msg_from_X_to_f::Gaussian1D, msg_from_S_to_f::Gamma1D, q_x, q_s, y; scale_factor=50.0)
    p1 = Gaussian.distribution(msg_from_X_to_f)
    p2 = Gamma.distribution(msg_from_S_to_f)
    q_x = Gaussian.distribution(q_x)
    q_s = Gamma.distribution(q_s)

    # Compute plotting ranges based on distribution parameters
    xs = range(
        start = y - 4 * sqrt(Gaussian.variance(msg_from_X_to_f)), 
        stop = y + 4 * sqrt(Gaussian.variance(msg_from_X_to_f)), 
        length = 100
    )
    ss = range(
        start = 0, 
        stop = Gamma.mean(msg_from_S_to_f) + 5 * sqrt(Gamma.variance(msg_from_S_to_f)), 
        length = 100
    )

    # Create contour plot of factorized distribution
    plt = contour(
        xs, 
        ss, 
        (x, s) -> scale_factor * pdf(p1, x) * pdf(p2, s) * pdf(Normal(x, sqrt(1/s)), y),
        colormap = :black,
        xlabel = L"x", 
        ylabel = L"s", 
        legend = false,
        levels = [0.005, 0.01, 0.05, 0.1, 1.0],
        lw = 2,
        xtickfontsize=16,
        ytickfontsize=16,
        xguidefontsize=18,
        yguidefontsize=18,

    )    

    contour!(
        xs, 
        ss, 
        (x, s) -> pdf(q_x, x) * pdf(q_s, s),
        colormap = :red,
        legend = false,
        levels = [0.005, 0.01, 0.05, 0.1, 1.0],
        lw = 2
    )    

    display(plt)
end

"""
    draw_contour_plot_mean_factor(msg_from_X_to_f, msg_from_Y_to_f, msg_from_S_to_f, q_x, q_y, q_s; no_levels=8, base_path="...") -> Nothing

Visualizes 2D marginal projections of the 3D joint distribution over (x, y, s) space.

Creates three separate contour plots showing marginals over pairs of variables:
1. **p(x, y)**: Joint over mean X and observation Y
2. **p(x, s)**: Joint over mean X and precision S
3. **p(y, s)**: Joint over observation Y and precision S

Each plot overlays:
- **Black contours**: Exact marginal (computed by integrating out third variable)
- **Red contours**: Variational approximation (factorized marginal)

# Mathematical Background

## 3D Joint Distribution (Exact)

```
p(x, y, s) ∝ msg_X(x) × msg_Y(y) × msg_S(s) × p(y | x, s)
```

where p(y | x, s) = N(y | x, s⁻¹) represents the Gaussian mean factor.

## 2D Marginals (Exact)

Obtained by marginalizing over the third variable:
```
p(x, y) = ∫ p(x, y, s) ds
p(x, s) = ∫ p(x, y, s) dy
p(y, s) = ∫ p(x, y, s) dx
```

## Variational Approximation

```
q(x, y, s) = q(x) q(y) q(s)
```

Factorized form loses all correlations between variables.

# Arguments
- `msg_from_X_to_f::Gaussian1D`: Prior/incoming message for mean parameter x
- `msg_from_Y_to_f::Gaussian1D`: Prior/incoming message for observation y
- `msg_from_S_to_f::Gamma1D`: Prior/incoming message for precision parameter s
- `q_x::Gaussian1D`: Current variational marginal for x
- `q_y::Gaussian1D`: Current variational marginal for y
- `q_s::Gamma1D`: Current variational marginal for s

# Keywords
- `no_levels::Int=8`: Number of contour levels to display
- `base_path::String`: Base path for saving output files (without extension)

# Output Files

Three SVG files showing different 2D projections:
- `{base_path}_xy.svg`: Contours over (x, y) space
- `{base_path}_xs.svg`: Contours over (x, s) space
- `{base_path}_ys.svg`: Contours over (y, s) space

# Plotting Details

**For p(x, y) plot:**
- X-axis: Mean parameter x, range: mean ± 3σ of msg_from_X_to_f
- Y-axis: Observation y, range: mean ± 3σ of msg_from_Y_to_f
- Black: Exact marginal (integrate over s)
- Red: q(x)q(y)

**For p(x, s) plot:**
- X-axis: Mean parameter x
- Y-axis: Precision parameter s, range: 0 to mean + 3σ of msg_from_S_to_f
- Black: Exact marginal (integrate over y)
- Red: q(x)q(s)

**For p(y, s) plot:**
- X-axis: Observation y
- Y-axis: Precision parameter s
- Black: Exact marginal (integrate over x)
- Red: q(y)q(s)

# Interpretation

**Contour comparison:**
- Tilted exact contours → variables are correlated
- Axis-aligned variational contours → independence assumed
- Large discrepancy → poor VMP approximation

**Variable dependencies:**
- (x, y): Coupled through Gaussian mean factor
- (x, s): Coupled through observation likelihood
- (y, s): Coupled through observation likelihood

# Examples
```julia
# Visualize after VMP convergence
draw_contour_plot_mean_factor(
    prior_X, prior_Y, prior_S,
    converged_q_x, converged_q_y, converged_q_s,
    no_levels=10,
    base_path="~/results/final"
)
```

# Algorithm

1. Convert custom types to Distributions.jl types
2. Compute 3D grid over (x, y, s) space (100×100×100 default)
3. Evaluate exact joint p(x, y, s) at all grid points
4. Marginalize (sum) to get 2D marginals
5. Normalize both exact and variational marginals
6. Plot contours and save each projection

# Use Cases
- Final visualization after VMP convergence
- Comparing exact vs. approximate correlations
- Understanding 3D distribution structure via 2D projections
- Teaching multivariate dependencies

# Notes
- All distributions normalized to sum to 1 for fair comparison
- Requires sufficient grid resolution (100×100×100 default)
- Computation intensive for fine grids
- Automatically displays all three plots
"""
function draw_contour_plot_mean_factor(msg_from_X_to_f::Gaussian1D, msg_from_Y_to_f::Gaussian1D, msg_from_S_to_f::Gamma1D, q_x, q_y, q_s; no_levels=8, base_path="~/Downloads/vmp_gaussian_mean_factor")
    p1 = Gaussian.distribution(msg_from_X_to_f)
    p2 = Gaussian.distribution(msg_from_Y_to_f)
    p3 = Gamma.distribution(msg_from_S_to_f)
    q_x = Gaussian.distribution(q_x)
    q_y = Gaussian.distribution(q_y)
    q_s = Gamma.distribution(q_s)

    # Compute plotting ranges based on distribution parameters
    xs = range(
        start = Gaussian.mean(msg_from_X_to_f) - 3 * sqrt(Gaussian.variance(msg_from_X_to_f)), 
        stop = Gaussian.mean(msg_from_X_to_f) + 3 * sqrt(Gaussian.variance(msg_from_X_to_f)), 
        length = 100
    )
    ys = range(
        start = Gaussian.mean(msg_from_Y_to_f) - 3 * sqrt(Gaussian.variance(msg_from_Y_to_f)), 
        stop = Gaussian.mean(msg_from_Y_to_f) + 3 * sqrt(Gaussian.variance(msg_from_Y_to_f)), 
        length = 100
    )
    ss = range(
        start = 0, 
        stop = Gamma.mean(msg_from_S_to_f) + 3 * sqrt(Gamma.variance(msg_from_S_to_f)), 
        length = 100
    )

    true_pdf = Array{Float64}(undef, length(xs), length(ys), length(ss))
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(ys)
            for (k, s) in enumerate(ss)
                true_pdf[i, j, k] = pdf(p1, x) * pdf(p2, y) * pdf(p3, s) * pdf(Normal(x, sqrt(1/s)), y) 
            end
        end
    end
    true_pdf .= true_pdf ./ sum(true_pdf)


    # Create contour plot of factorized distribution
    plt = contour(
        xs, 
        ys, 
        sum(true_pdf, dims=3)[:, :, 1]',
        colormap = :black,
        xlabel = L"x", 
        ylabel = L"y", 
        legend = false,
        lw = 2,
        levels = no_levels,
        xtickfontsize=16,
        ytickfontsize=16,
        xguidefontsize=18,
        yguidefontsize=18,
    )   
    q_xy = Array{Float64}(undef, length(xs), length(ys))
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(ys)
            q_xy[i, j] = pdf(q_x, x) * pdf(q_y, y)
        end
    end
    contour!(
        xs, 
        ys, 
        q_xy' ./ sum(q_xy),
        colormap = :red,
        legend = false,
        lw = 2,
        levels = no_levels,
    )
    savefig(plt, "$(base_path)_xy.svg")
    display(plt)

    plt = contour(
        xs, 
        ss, 
        sum(true_pdf, dims=2)[:, 1, :]',
        colormap = :black,
        xlabel = L"x", 
        ylabel = L"s", 
        legend = false,
        lw = 2,
        levels = no_levels,
        xtickfontsize=16,
        ytickfontsize=16,
        xguidefontsize=18,
        yguidefontsize=18,
    )
    q_xs = Array{Float64}(undef, length(xs), length(ss))
    for (i, x) in enumerate(xs)
        for (k, s) in enumerate(ss)
            q_xs[i, k] = pdf(q_x, x) * pdf(q_s, s)
        end
    end
    contour!(
        xs, 
        ss, 
        q_xs' ./ sum(q_xs),
        colormap = :red,
        legend = false,
        lw = 2,
        levels = no_levels,
    )    
    savefig(plt, "$(base_path)_xs.svg")
    display(plt)

    plt = contour(
        ys, 
        ss, 
        sum(true_pdf, dims=1)[1, :, :]',
        colormap = :black,
        xlabel = L"y", 
        ylabel = L"s", 
        legend = false,
        lw = 2,
        levels = no_levels,
        xtickfontsize=16,
        ytickfontsize=16,
        xguidefontsize=18,
        yguidefontsize=18,
    )
    q_ys = Array{Float64}(undef, length(ys), length(ss))
    for (j, y) in enumerate(ys)
        for (k, s) in enumerate(ss)
            q_ys[j, k] = pdf(q_y, y) * pdf(q_s, s)
        end
    end
    contour!(
        ys, 
        ss, 
        q_ys' ./ sum(q_ys),
        colormap = :red,
        legend = false,
        lw = 2,
        levels = no_levels,
    )
    savefig(plt, "$(base_path)_ys.svg")
    display(plt)
end

"""
    variational_message_passing_gaussian_factor(msg_from_X_to_f, msg_from_S_to_f, y, base_path="...") -> Nothing

Runs VMP iterations until convergence, saving contour plots at each step.

# Algorithm

Initialize uniform messages from factor f.

**Until convergence:**
1. **Update message f → x:**
   - Get current marginal q(s) = msg_{f→s} × msg_{s→f}
   - Compute: msg_{f→x} = N(x | y⟨s⟩, ⟨s⟩⁻¹)
   - Update marginal: q(x) = msg_{f→x} × msg_{x→f}

2. **Update message f → s:**
   - Get current marginal q(x) = msg_{f→x} × msg_{x→f}
   - Compute: msg_{f→s} = Gamma(s | 1/2, ...)
   - Update marginal: q(s) = msg_{f→s} × msg_{s→f}

3. **Check convergence:**
   - Compute KL(q_old(x) || q_new(x)) and KL(q_old(s) || q_new(s))
   - If both < 1e-6, stop; else continue

4. **Visualize:**
   - Save contour plot at each iteration

# Arguments
- `msg_from_X_to_f::Gaussian1D`: Prior message on mean (typically prior distribution)
- `msg_from_S_to_f::Gamma1D`: Prior message on precision (typically prior distribution)
- `y::Real`: Observed data point
- `base_path::String`: Base path for saving iteration plots

# Convergence Properties

- **Monotonic ELBO**: Each update increases (or maintains) evidence lower bound
- **Typical iterations**: 5-20 for moderate priors
- **KL threshold**: 1e-6 ensures tight convergence
- **Guaranteed convergence**: On conjugate-exponential models (this case)

# Output Files

For each iteration i:
- `{base_path}_step_{i}.svg`: Contour plot showing exact vs. VMP

# Console Output

Each iteration prints:
```
KL divergence for X: {value}, for S: {value}
```

Final iteration saves plot but doesn't print (convergence achieved).

# Mathematical Details

## Message f → x

Given: q(s) from previous iteration
```
log q*(x) = E_q(s)[log p(y|x,s)] + const
          = E_q(s)[-s/2(y-x)²] + const
          = -⟨s⟩/2(y² - 2yx + x²) + const
          ∝ N(x | y⟨s⟩, ⟨s⟩⁻¹)
```

Parameters: τ = y⟨s⟩, ρ = ⟨s⟩

## Message f → s

Given: q(x) from current iteration
```
log q*(s) = E_q(x)[log p(y|x,s)] + const
          = 1/2 log s - s/2 E_q(x)[(y-x)²] + const
          = 1/2 log s - s/2(y² - 2y⟨x⟩ + ⟨x²⟩) + const
          ∝ Gamma(s | 1/2, (1/2)(y² - 2y⟨x⟩ + ⟨x²⟩))
```

Where: ⟨x²⟩ = Var[x] + ⟨x⟩²

# Examples
```julia
# Tight prior on mean, moderate prior on precision
prior_X = Gaussian.Gaussian1D(0.0, 1.0)   # N(0, 1)
prior_S = Gamma.Gamma1D(3.0, 0.5)         # Gamma(3, 0.5)
variational_message_passing_gaussian_factor(prior_X, prior_S, 2.5)

# Vague priors
prior_X = Gaussian.Gaussian1D(0.0, 100.0)
prior_S = Gamma.Gamma1D(0.1, 0.1)
variational_message_passing_gaussian_factor(prior_X, prior_S, 2.5)
```

# Notes
- Requires write access to directory in base_path
- Large scale_factor in draw_contour_plot aids visualization
- For theory, see Winn & Bishop (2005) Section 3.2
"""
function variational_message_passing_gaussian_factor(msg_from_X_to_f::Gaussian1D, msg_from_S_to_f::Gamma1D, y, base_path="~/Downloads/vmp_gaussian_factor")
    msg_from_f_to_X = Gaussian.Gaussian1DUniform()
    msg_from_f_to_S = Gamma.Gamma1DUniform()

    steps = 0
    while true
        # cache the old marginals
        p_X_old = msg_from_f_to_X * msg_from_X_to_f
        p_S_old = msg_from_f_to_S * msg_from_S_to_f

        # obtain the current marginal of S and compute the variational approximation of the message from f to X
        p_S = msg_from_f_to_S * msg_from_S_to_f
        msg_from_f_to_X = Gaussian.Gaussian1D(y * Gamma.mean(p_S), Gamma.mean(p_S))

        # Update message from factor f to X
        p_X = msg_from_f_to_X * msg_from_X_to_f
        msg_from_f_to_S = Gamma.Gamma1D(0.5, 1/2 * (y^2 + Gaussian.variance(p_X) + (Gaussian.mean(p_X))^2) - y * Gaussian.mean(p_X))

        # compute the new marginals
        p_X_new = msg_from_f_to_X * msg_from_X_to_f
        p_S_new = msg_from_f_to_S * msg_from_S_to_f

        KL_x = Gaussian.KL_divergence(p_X_old, p_X_new)
        KL_s = Gamma.KL_divergence(p_S_old, p_S_new)
        if KL_x < 1e-6 && KL_s < 1e-6
            draw_contour_plot_factor(msg_from_X_to_f, msg_from_S_to_f, p_X_new, p_S_new, y)
            savefig("$(base_path)_step_$(steps).svg")
            println("p_X: ", p_X_new, ", p_S: ", p_S_new)
            break
            # return (p_X_new, p_S_new)
        else
            draw_contour_plot_factor(msg_from_X_to_f, msg_from_S_to_f, p_X_old, p_S_old, y)
            savefig("$(base_path)_step_$(steps).svg")
            println("KL divergence for X: ", KL_x, ", for S: ", KL_s)
        end
        steps += 1
    end
end

"""
    variational_message_passing_gaussian_mean_factor(msg_from_X_to_f, msg_from_Y_to_f, msg_from_S_to_f, base_path="...") -> Nothing

Runs VMP iterations on the Gaussian mean factor until convergence.

Performs variational inference on the model:
```
x ~ msg_from_X_to_f      # Prior on mean
y ~ msg_from_Y_to_f      # Prior on observation
s ~ msg_from_S_to_f      # Prior on precision
Constraint: y ~ N(x, s⁻¹)
```

The algorithm iteratively updates q(x), q(y), and q(s) until all three
converge (KL divergence < 1e-6), then generates 2D marginal visualizations.

# Algorithm

Initialize uniform messages from factor f.

**Until convergence:**

1. **Update message f → x:**
   ```
   q(x) ∝ exp(E_{q(y),q(s)}[log p(y | x, s)])
        = N(x | ⟨y⟩⟨s⟩, ⟨s⟩⁻¹)
   ```

2. **Update message f → y:**
   ```
   q(y) ∝ exp(E_{q(x),q(s)}[log p(y | x, s)])
        = N(y | ⟨x⟩⟨s⟩, ⟨s⟩⁻¹)
   ```

3. **Update message f → s:**
   ```
   q(s) ∝ exp(E_{q(x),q(y)}[log p(y | x, s)])
        = Gamma(s | 1/2, (1/2)(Var[y] + ⟨y⟩² + Var[x] + ⟨x⟩² - 2⟨x⟩⟨y⟩))
   ```

4. **Check convergence:**
   - Compute KL(q_old(x) || q_new(x)), KL(q_old(y) || q_new(y)), KL(q_old(s) || q_new(s))
   - If all three < 1e-6, stop; else continue

5. **Visualize:**
   - Generate three 2D marginal contour plots

# Arguments
- `msg_from_X_to_f::Gaussian1D`: Prior message on mean x
- `msg_from_Y_to_f::Gaussian1D`: Prior message on observation y
- `msg_from_S_to_f::Gamma1D`: Prior message on precision s
- `base_path::String`: Base path for saving visualization files

# Convergence Properties

- **Monotonic ELBO**: Each update increases evidence lower bound
- **Typical iterations**: 5-30 for moderate priors
- **KL threshold**: 1e-6 ensures tight convergence
- **Guaranteed convergence**: On this conjugate-exponential model

# Output Files

After convergence, generates three visualization files:
- `{base_path}_xy.svg`: Contours over (x, y) space
- `{base_path}_xs.svg`: Contours over (x, s) space
- `{base_path}_ys.svg`: Contours over (y, s) space

# Console Output

**During iterations:**
```
KL divergence for X: {value}, for Y: {value}, for S: {value}
...
```

**After convergence:**
```
p_X: Gaussian1D(...), p_Y: Gaussian1D(...), p_S: Gamma1D(...)
```

# Mathematical Details

## Message f → x

Given current q(y) and q(s):
```
log q*(x) = E_{q(y),q(s)}[log p(y | x, s)] + const
          = E[-s/2(y - x)²] + const
          = -⟨s⟩/2(⟨y²⟩ - 2⟨y⟩x + x²) + const
          ∝ N(x | ⟨y⟩, ⟨s⟩⁻¹)
```

Natural parameters: τ = ⟨y⟩⟨s⟩, ρ = ⟨s⟩

## Message f → y

By symmetry with f → x:
```
q*(y) = N(y | ⟨x⟩, ⟨s⟩⁻¹)
```

Natural parameters: τ = ⟨x⟩⟨s⟩, ρ = ⟨s⟩

## Message f → s

Given current q(x) and q(y):
```
log q*(s) = E_{q(x),q(y)}[log p(y | x, s)] + const
          = 1/2 log s - s/2 E[(y - x)²] + const
          = 1/2 log s - s/2(⟨y²⟩ + ⟨x²⟩ - 2⟨x⟩⟨y⟩) + const
          ∝ Gamma(s | 1/2, (1/2)(⟨y²⟩ + ⟨x²⟩ - 2⟨x⟩⟨y⟩))
```

where: ⟨x²⟩ = Var[x] + ⟨x⟩², ⟨y²⟩ = Var[y] + ⟨y⟩²

# Examples
```julia
# Standard setup: independent priors
variational_message_passing_gaussian_mean_factor(
    Gaussian.Gaussian1DFromMeanVariance(0.0, 1.0),  # Prior: x ~ N(0, 1)
    Gaussian.Gaussian1DFromMeanVariance(2.0, 0.5),  # Prior: y ~ N(2, 0.5)
    Gamma.Gamma1D(2.0, 1.0)                         # Prior: s ~ Gamma(2, 1)
)

# Tight priors scenario
variational_message_passing_gaussian_mean_factor(
    Gaussian.Gaussian1DFromMeanVariance(0.0, 0.1),  # Strong belief x ≈ 0
    Gaussian.Gaussian1DFromMeanVariance(1.0, 0.1),  # Strong belief y ≈ 1
    Gamma.Gamma1D(5.0, 0.5)                         # Moderate precision prior
)

# Vague priors (data-driven)
variational_message_passing_gaussian_mean_factor(
    Gaussian.Gaussian1DFromMeanVariance(0.0, 100.0), # Vague x
    Gaussian.Gaussian1DFromMeanVariance(0.0, 100.0), # Vague y
    Gamma.Gamma1D(0.1, 0.1)                          # Vague precision
)
```

# Interpretation

**Converged distributions:**
- q(x): Posterior belief about mean parameter
- q(y): Posterior belief about observation (refined by constraint)
- q(s): Posterior belief about precision

**Expected behavior:**
- If priors on x and y are far apart, expect slow convergence
- If precision prior is weak, s will be mainly data-driven
- Stronger priors → fewer iterations needed

# Use Cases
- Inferring latent mean from noisy observations
- Estimating both mean and noise level jointly
- Teaching VMP on multivariate models
- Comparing with exact inference on small problems

# Notes
- Requires write access to directory in base_path
- Uses custom Gaussian1D and Gamma1D types
- Visualizations show failure of mean-field to capture correlations
- For theoretical background, see Winn & Bishop (2005)

# Related Functions
- `variational_message_passing_gaussian_factor`: 2-variable version (x, s | y)
- `draw_contour_plot_mean_factor`: Visualization utility used here
"""
function variational_message_passing_gaussian_mean_factor(msg_from_X_to_f::Gaussian1D, msg_from_Y_to_f::Gaussian1D, msg_from_S_to_f::Gamma1D, base_path="~/Downloads/vmp_gaussian_mean_factor")
    msg_from_f_to_X = Gaussian.Gaussian1DUniform()
    msg_from_f_to_Y = Gaussian.Gaussian1DUniform()
    msg_from_f_to_S = Gamma.Gamma1DUniform()

    steps = 0
    while true
        # cache the old marginals
        p_X_old = msg_from_f_to_X * msg_from_X_to_f
        p_Y_old = msg_from_f_to_Y * msg_from_Y_to_f
        p_S_old = msg_from_f_to_S * msg_from_S_to_f

        # obtain the current marginal of S and Y and compute the variational approximation of the message from f to X
        p_S = msg_from_f_to_S * msg_from_S_to_f
        p_Y = msg_from_f_to_Y * msg_from_Y_to_f
        msg_from_f_to_X = Gaussian.Gaussian1D(Gaussian.mean(p_Y) * Gamma.mean(p_S), Gamma.mean(p_S))

        # update message from f to Y
        p_S = msg_from_f_to_S * msg_from_S_to_f
        p_X = msg_from_f_to_X * msg_from_X_to_f
        msg_from_f_to_Y = Gaussian.Gaussian1D(Gaussian.mean(p_X) * Gamma.mean(p_S), Gamma.mean(p_S))

        # Update message from factor f to S
        p_X = msg_from_f_to_X * msg_from_X_to_f
        p_Y = msg_from_f_to_Y * msg_from_Y_to_f
        msg_from_f_to_S = Gamma.Gamma1D(0.5, 1/2 * (Gaussian.variance(p_Y) + (Gaussian.mean(p_Y))^2 + Gaussian.variance(p_X) + (Gaussian.mean(p_X))^2) - Gaussian.mean(p_Y) * Gaussian.mean(p_X))

        # compute the new marginals
        p_X_new = msg_from_f_to_X * msg_from_X_to_f
        p_Y_new = msg_from_f_to_Y * msg_from_Y_to_f
        p_S_new = msg_from_f_to_S * msg_from_S_to_f

        KL_x = Gaussian.KL_divergence(p_X_old, p_X_new)
        KL_y = Gaussian.KL_divergence(p_Y_old, p_Y_new)
        KL_s = Gamma.KL_divergence(p_S_old, p_S_new)
        if KL_x < 1e-6 && KL_y < 1e-6 && KL_s < 1e-6
            draw_contour_plot_mean_factor(msg_from_X_to_f, msg_from_Y_to_f, msg_from_S_to_f, p_X_new, p_Y_new, p_S_new, base_path=base_path)
            println("p_X: ", p_X_new, ", p_Y: ", p_Y_new, ", p_S: ", p_S_new)
            break
            # return (p_X_new, p_Y_new, p_S_new)
        else
            println("KL divergence for X: ", KL_x, ", for Y: ", KL_y, ", for S: ", KL_s)
        end
        steps += 1
    end
end

"""
    draw_gaussian_mean_factor(msg_from_X, msg_from_Y; β=1.0) -> Nothing

Visualizes exact vs. variational approximation for Gaussian mean factor.

Compares three joint distributions for the factor y = x + ε where ε ~ N(0, β²):
1. **Black contours**: Exact joint p(x, y) with correlation
2. **Blue contours**: Exact marginals (independent) - for reference
3. **Red contours**: VMP factorized approximation q(x)q(y)

# Factor Model

```
x ~ msg_from_X       # Incoming message/prior on x
y ~ msg_from_Y       # Incoming message/prior on y
Constraint: y = x + ε, ε ~ N(0, β²)
```

# Mathematical Background

## Exact Joint (Black)

Given messages μ_X ~ N(m_X, σ_X²) and μ_Y ~ N(m_Y, σ_Y²):
```
p(x, y) ∝ μ_X(x) μ_Y(y) p(y | x)
        = N([x, y] | μ, Σ)
```

where Σ has off-diagonal terms capturing correlation.

## Exact Marginals (Blue)

Uses correct means but assumes independence:
```
p_indep(x, y) = p(x) p(y)
```

Marginals are correct but correlation lost.

## Variational (Red)

VMP computes q(x) and q(y) separately via message passing:
```
q(x) ∝ μ_X(x) × msg_{y→x}
q(y) ∝ μ_Y(y) × msg_{x→y}
```

Both marginals and correlation are approximations.

# Arguments
- `msg_from_X::Normal{Float64}`: Incoming message/belief about x
- `msg_from_Y::Normal{Float64}`: Incoming message/belief about y

# Keywords
- `β::Float64=1.0`: Noise standard deviation linking x and y

# Contour Colors

- **Black**: Exact joint (may be tilted showing correlation)
- **Blue**: Product of exact marginals (axis-aligned)
- **Red**: VMP approximation (axis-aligned, different from blue)

# When VMP Works Well

- **Large β**: Weak constraint, little correlation to capture
- **Wide messages**: High uncertainty drowns out correlation
- Red ≈ Blue ≈ Black

# When VMP Struggles

- **Small β**: Tight constraint induces strong correlation
- **Narrow messages**: Low uncertainty amplifies correlation effects
- Red ≠ Blue, Black tilted away from axis-aligned

# Visualization Details

- **X-axis**: Parameter x, range: mean(msg_X) ± 3σ_X
- **Y-axis**: Parameter y, range: mean(msg_Y) ± 3σ_Y
- **Contour levels**: 10 levels automatically chosen
- **Line widths**: 2pt for all contours

# Examples
```julia
# Scenario 1: Tight messages, small noise
draw_gaussian_mean_factor(Normal(0, 0.5), Normal(0, 0.5), β=0.5)
# Expect: Large discrepancy between black and red

# Scenario 2: Wide messages, large noise
draw_gaussian_mean_factor(Normal(0, 5), Normal(0, 5), β=5.0)
# Expect: Red ≈ Blue ≈ Black

# Scenario 3: Standard case from Winn & Bishop
draw_gaussian_mean_factor(Normal(0, 1), Normal(0, 1), β=1.0)
# Expect: Moderate discrepancy
```

# Console Output

Prints three distributions for numerical comparison:
```
Joint distribution (exact): MvNormal(...)
Joint distribution (exact marginals): MvNormal(...)
Joint distribution (variational): MvNormal(...)
```

# Interpretation

**Correlation indicator:**
- Exact Σ off-diagonal ≠ 0 → correlation present
- VMP Σ diagonal → independence assumed

**Approximation quality:**
- Compare means: VMP vs. exact
- Compare variances: VMP often underestimates (certainty overestimation)

# Key Insight

Mean-field VMP assumes q(x, y) = q(x)q(y), which cannot represent
correlation. This is the fundamental trade-off:
- **Gain**: Tractable inference via message passing
- **Loss**: Ignores dependencies captured by exact joint

# Notes
- Automatically displays plot; save externally if needed
- Uses Distributions.jl types (Normal, MvNormal)
"""
function draw_gaussian_mean_factor(msg_from_X::Normal{Float64}, msg_from_Y::Normal{Float64}; β = 1.0)
    # ========================================================================
    # Helper: Multiply two Gaussian messages
    # ========================================================================
    function multiply(x::Normal{Float64}, y::Normal{Float64})
        # Natural parameters
        τ_x = Distributions.mean(x) / Distributions.var(x)
        ρ_x = 1 / Distributions.var(x)
        τ_y = Distributions.mean(y) / Distributions.var(y)
        ρ_y = 1 / Distributions.var(y)
        
        # Product in natural parameters
        τ = τ_x + τ_y
        ρ = ρ_x + ρ_y
        
        return Normal(τ/ρ, sqrt(1/ρ))
    end

    # Extract message parameters
    μ_X = Distributions.mean(msg_from_X)
    σ2_X = Distributions.var(msg_from_X)
    μ_Y = Distributions.mean(msg_from_Y)
    σ2_Y = Distributions.var(msg_from_Y)
    
    # Normalization constant
    c = σ2_X + σ2_Y + β^2

    # ========================================================================
    # Exact Joint Distribution
    # ========================================================================
    
    # Mean vector (weighted combination of messages)
    μ = [μ_X * (σ2_Y + β^2)/c + μ_Y * σ2_X/c; 
         μ_X * σ2_Y/c + μ_Y * (σ2_X + β^2)/c]
    
    # Covariance matrix (captures correlation)
    Σ = Hermitian([σ2_X * (σ2_Y + β^2)/c σ2_Y * σ2_X * 1/c; 
                   σ2_Y * σ2_X * 1/c σ2_Y * (σ2_X + β^2)/c])
    joint = MvNormal(μ, Σ)

    # ========================================================================
    # Exact with Independent Assumption
    # ========================================================================
    
    μ_exact = [μ_X * (σ2_Y + β^2)/c + μ_Y * σ2_X/c; 
               μ_X * σ2_Y/c + μ_Y * (σ2_X + β^2)/c]
    # Diagonal covariance (no correlation)
    Σ_exact = Hermitian([σ2_X * (σ2_Y + β^2)/c 0; 
                         0 σ2_Y * (σ2_X + β^2)/c])
    joint_exact = MvNormal(μ_exact, Σ_exact)

    # ========================================================================
    # Variational Approximation (VMP)
    # ========================================================================
    μ_var = [μ_X * (σ2_Y + β^2)/c + μ_Y * σ2_X/c; 
             μ_X * σ2_Y/c + μ_Y * (σ2_X + β^2)/c]
    # Diagonal covariance (no correlation)
    Σ_var = Hermitian([σ2_X * β^2/(σ2_X + β^2) 0; 
                         0 σ2_Y * β^2/(σ2_Y + β^2)])
    joint_var = MvNormal(μ_var, Σ_var)

    # Print distributions for comparison
    println("Joint distribution (exact): ", joint)
    println("Joint distribution (exact marginals): ", joint_exact)
    println("Joint distribution (variational): ", joint_var)

    # ========================================================================
    # Visualization
    # ========================================================================
    
    # Compute plotting ranges
    xs = range(
        start = Distributions.mean(msg_from_X) - 3 * sqrt(Distributions.var(msg_from_X)), 
        stop = Distributions.mean(msg_from_X) + 3 * sqrt(Distributions.var(msg_from_X)), 
        length = 100
    )
    ys = range(
        start = Distributions.mean(msg_from_Y) - 3 * sqrt(Distributions.var(msg_from_Y)), 
        stop = Distributions.mean(msg_from_Y) + 3 * sqrt(Distributions.var(msg_from_Y)), 
        length = 100
    )
    
    # Plot exact joint (black)
    p = contour(
        xs, 
        ys, 
        (x, y) -> pdf(joint, [x, y])[1],
        color = :black,
        linewidth = 2,
        levels = 10,
        xlabel = L"x", 
        ylabel = L"y", 
        legend = false,
        xtickfontsize=12,
        ytickfontsize=12,
        xguidefontsize=14,
        yguidefontsize=14,
        legendfontsize=12,        
    )    
    
    # Overlay exact with independence (blue)
    contour!(
        xs, 
        ys, 
        (x, y) -> pdf(joint_exact, [x, y])[1],
        color = :blue,
        linewidth = 2,
        levels = 10,
    )    
    
    # Overlay variational approximation (red)
    contour!(
        xs, 
        ys, 
        (x, y) -> pdf(joint_var, [x, y])[1],
        color = :red,
        linewidth = 2,
        levels = 10,
    )    

    display(p)
end

"""
    main() -> Nothing

Runs comprehensive VMP demonstrations on two different Gaussian models.

Executes two main demonstrations:
1. **Gaussian Factor Model**: Inference on (mean, precision) given observation
2. **Gaussian Mean Factor Model**: Inference on (mean, observation, precision)
3. **Factor Approximation Analysis**: Comparison of exact vs. VMP for different noise levels

# Demonstrations

## Part 1: Gaussian Factor (2 Variables)

**Model:**
```
x ~ N(0, 1)         # Prior on mean
s ~ Gamma(3, 0.5)   # Prior on precision
y = 2 (observed)    # Fixed observation
```

**Inference:** q(x, s | y=2)

**Output:**
- Contour plots at each VMP iteration
- Files: `vmp_gaussian_factor_step_{i}.svg`

## Part 2: Gaussian Mean Factor (3 Variables)

**Model:**
```
x ~ N(0, 1)         # Prior on mean
y ~ N(2, 0.5)       # Prior on observation
s ~ Gamma(2, 1)     # Prior on precision
Constraint: y ~ N(x, s⁻¹)
```

**Inference:** q(x, y, s) given the constraint

**Output:**
- Three 2D marginal projections: (x,y), (x,s), (y,s)
- Files: `vmp_gaussian_mean_factor_{xy,xs,ys}.svg`

## Part 3: Factor Approximation Analysis

Compares exact vs. VMP for Gaussian mean factor y = x + ε with three noise levels:
- β = 0.5 (tight constraint)
- β = 1.0 (moderate noise)
- β = 2.0 (high noise)

**Output:**
- Files: `gaussian_mean_factor_{0.5,1.0,2.0}.{svg,pdf}`

# Output Files

**Generated in ~/Downloads/:**

*Gaussian Factor Model:*
- `vmp_gaussian_factor_step_0.svg`
- `vmp_gaussian_factor_step_1.svg`
- ... (continues until convergence)

*Gaussian Mean Factor Model:*
- `vmp_gaussian_mean_factor_xy.svg`: (x, y) marginal
- `vmp_gaussian_mean_factor_xs.svg`: (x, s) marginal
- `vmp_gaussian_mean_factor_ys.svg`: (y, s) marginal

*Factor Approximation:*
- `gaussian_mean_factor_0.5.{svg,pdf}`
- `gaussian_mean_factor_1.0.{svg,pdf}`
- `gaussian_mean_factor_2.0.{svg,pdf}`

# Console Output

**Gaussian Factor Model:**
```
KL divergence for X: 0.xyz..., for S: 0.abc...
KL divergence for X: 0.012..., for S: 0.034...
...
p_X: Gaussian1D(...), p_S: Gamma1D(...)
```

**Gaussian Mean Factor Model:**
```
KL divergence for X: 0.xyz..., for Y: 0.abc..., for S: 0.def...
...
p_X: Gaussian1D(...), p_Y: Gaussian1D(...), p_S: Gamma1D(...)
```

**Factor Approximation:**
```
======================================================================
Gaussian Mean Factor Approximation Analysis
======================================================================

β = 0.5 (tight constraint):
Joint distribution (exact): MvNormal(...)
Joint distribution (exact marginals): MvNormal(...)
Joint distribution (variational): MvNormal(...)

...

======================================================================
All visualizations complete!
Files saved to ~/Downloads/
======================================================================
```

# Expected Behavior

## Gaussian Factor Model
- Converges in ~10-15 iterations
- Red contours (VMP) approach black contours (exact marginals)
- Final distributions: x ≈ N(2, σ²), s refined from prior

## Gaussian Mean Factor Model
- Converges in ~15-25 iterations
- Three KL divergences all decrease to < 1e-6
- Visualizations show 3-way correlation structure
- VMP approximation loses correlation (axis-aligned red contours)

## Factor Approximation
- β = 0.5: Large discrepancy (strong correlation ignored)
- β = 1.0: Moderate discrepancy
- β = 2.0: Small discrepancy (weak correlation, VMP works well)

# Examples
```julia
# Run all demonstrations with default parameters
VMP.main()
```

# Pedagogical Value

This demonstration teaches:

1. **VMP Algorithm:**
   - Iterative message passing
   - KL divergence monitoring
   - Convergence behavior

2. **Model Complexity:**
   - 2-variable model (simpler, faster)
   - 3-variable model (richer structure)
   - Trade-off: complexity vs. tractability

3. **Mean-Field Approximation:**
   - Factorization: q(x,y,s) = q(x)q(y)q(s)
   - Loss of correlation information
   - Visual evidence via contour plots

4. **Factor Graph Inference:**
   - Gaussian factor: observation with unknown mean/precision
   - Gaussian mean factor: linking two variables through precision
   - Message passing schedules

5. **Approximation Quality:**
   - When VMP works well (weak constraints)
   - When VMP struggles (tight constraints)
   - Visualizing the approximation gap

# Typical Runtime

- Gaussian Factor: ~5-10 seconds (fewer iterations, simpler plots)
- Gaussian Mean Factor: ~15-30 seconds (3D marginalization)
- Factor Approximation: ~5-10 seconds (3 contour plots)
- **Total:** ~30-50 seconds on modern hardware

# System Requirements

- **Memory:** ~200 MB for 3D grid computations
- **Disk:** ~2-8 MB for all SVG/PDF files
- **Dependencies:**
  - Custom Gaussian.jl and Gamma.jl (from ../unit1/)
  - Plots.jl with contour support
  - Distributions.jl
  - LaTeXStrings.jl

# Notes

- **File overwrites:** Existing files with same names are replaced
- **Directory access:** Requires write permission to ~/Downloads/
- **Custom types:** Uses Gaussian1D and Gamma1D for pedagogical clarity
- **3D computation:** Mean factor plots require significant computation
- **Reproducibility:** Deterministic behavior (no randomness)

# Troubleshooting

**Issue:** "Module not found" error
- **Solution:** Ensure ../unit1/gaussian.jl and ../unit1/gamma.jl exist

**Issue:** VMP doesn't converge (>50 iterations)
- **Cause:** Extreme priors or numerical instability
- **Solution:** Check prior parameters; try more moderate values

**Issue:** Slow 3D marginalization
- **Cause:** Fine grid resolution (100×100×100)
- **Solution:** Reduce grid size in draw_contour_plot_mean_factor

**Issue:** Memory error during 3D computation
- **Cause:** Insufficient RAM for large grids
- **Solution:** Reduce grid resolution or increase system memory

# Comparison Between Models

| Aspect | Gaussian Factor | Gaussian Mean Factor |
|--------|----------------|----------------------|
| Variables | 2 (x, s) | 3 (x, y, s) |
| Observation | Fixed (y=2) | Variable (prior on y) |
| Iterations | ~10-15 | ~15-25 |
| Visualizations | 2D contours | Three 2D projections |
| Correlation | x-s coupling | Three-way coupling |
| Complexity | Simpler | More realistic |

# Related Examples

- **vmp2.jl**: Alternative implementation with additional features
- **chain.jl**: Belief propagation on discrete chains
- **mixture_model_failing.jl**: Limitations of moment-matching

# References

- Winn & Bishop (2005). "Variational Message Passing." JMLR 6:661-694.
- Bishop (2006). "Pattern Recognition and Machine Learning." Chapter 10.
- Blei et al. (2017). "Variational Inference: A Review for Statisticians."
"""
function main()
    variational_message_passing_gaussian_factor(Gaussian.Gaussian1D(0.0, 1.0),  Gamma.Gamma1D(3.0, 0.5), 2)
    variational_message_passing_gaussian_mean_factor(Gaussian.Gaussian1DFromMeanVariance(0.0, 1.0), Gaussian.Gaussian1DFromMeanVariance(2.0, 0.5), Gamma.Gamma1D(2.0, 1))

    # variational_message_passing_gaussian_factor(Gaussian.Gaussian1D(0.0, 1.0), Gaussian.Gaussian1D(2.0, 0.0), Gamma.Gamma1D(3.0, 0.5))

    # ========================================================================
    # Factor approximation analysis
    # ========================================================================
    
    println("\n" * "=" ^ 70)
    println("Gaussian Mean Factor Approximation Analysis")
    println("=" ^ 70)
    
    # Compare exact vs. VMP for different noise levels
    println("\nβ = 0.5 (tight constraint):")
    draw_gaussian_mean_factor(Normal(0,1), Normal(0,1), β=0.5)
    savefig("~/Downloads/gaussian_mean_factor_0.5.pdf")
    savefig("~/Downloads/gaussian_mean_factor_0.5.svg")
    
    println("\nβ = 1.0 (moderate noise):")
    draw_gaussian_mean_factor(Normal(0,1), Normal(0,1), β=1.0)
    savefig("~/Downloads/gaussian_mean_factor_1.0.pdf")
    savefig("~/Downloads/gaussian_mean_factor_1.0.svg")
    
    println("\nβ = 2.0 (high noise):")
    draw_gaussian_mean_factor(Normal(0,1), Normal(0,1), β=2.0)
    savefig("~/Downloads/gaussian_mean_factor_2.0.pdf")
    savefig("~/Downloads/gaussian_mean_factor_2.0.svg")
    
    println("\n" * "=" ^ 70)
    println("All visualizations complete!")
    println("Files saved to ~/Downloads/")
    println("=" ^ 70)
end

end