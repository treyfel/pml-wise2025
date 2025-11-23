# Plots for normalization constant algorithms
#
# This module implements discretized belief propagation for the TrueSkill factor graph.
# It demonstrates message passing algorithms and normalization constant computation
# through discretization of continuous distributions.
#
# 2025 by Ralf Herbrich
# Hasso-Plattner Institute

module NormalizationPlots

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

"""
    Discretization

Represents a discretization of the real line into n+1 intervals, where n is the number
of thresholds. The first interval extends from -∞ to the first threshold, and the last
interval extends from the last threshold to +∞.

# Fields
- `threshold::Vector{Float64}`: Sorted vector of threshold values defining interval boundaries

# Example
```julia
d = Discretization([-2.0, -1.0, 0.0, 1.0, 2.0])
# Creates 6 intervals: (-∞,-2], (-2,-1], (-1,0], (0,1], (1,2], (2,∞)
```
"""
struct Discretization
    threshold::Vector{Float64}
end

"""
    Base.iterate(d::Discretization, state=0)

Iterator over all intervals of the discretization. Returns tuples of (lower_bound, upper_bound)
for each interval.

# Arguments
- `d::Discretization`: The discretization to iterate over
- `state::Int`: Current iteration state (default: 0)

# Returns
- `Nothing` when iteration is complete
- `((a, b), new_state)` where (a,b) is the current interval and new_state is the next state
"""
function Base.iterate(d::Discretization, state=0)
    i = state
    if i == 0
        return ((-Inf, d.threshold[i + 1]), i + 1)
    elseif i < length(d.threshold)
        return ((d.threshold[i], d.threshold[i + 1]), i + 1)
    elseif i == length(d.threshold)
        return ((d.threshold[i], Inf), i + 1)
    else
        return nothing
    end
end

"""
    Base.length(d::Discretization)

Returns the number of intervals in the discretization (number of thresholds + 1).
"""
function Base.length(d::Discretization)
    return length(d.threshold) + 1
end

"""
    Base.size(d::Discretization)

Returns the size of the discretization as a tuple, compatible with array-like interfaces.

# Returns
- `Tuple{Int}`: A tuple containing the number of intervals
"""
function Base.size(d::Discretization)
    return (length(d.threshold) + 1, )
end

"""
    Base.eachindex(d::Discretization)

Returns an iterator over all valid indices of the discretization (1 to length(d)).

# Returns
- `UnitRange{Int}`: Range from 1 to the number of intervals
"""
function Base.eachindex(d::Discretization)
    return 1:length(d)
end

"""
    Base.getindex(d::Discretization, i::Int)

Returns the i-th interval as a tuple (lower_bound, upper_bound).

# Arguments
- `d::Discretization`: The discretization
- `i::Int`: Index of the interval (1-based)

# Returns
- `(a, b)`: Tuple representing the interval bounds
"""
function Base.getindex(d::Discretization, i::Int)
    if i == 1
        return (-Inf, d.threshold[i])
    elseif i == length(d.threshold) + 1
        return (d.threshold[i - 1], Inf)
    else
        return (d.threshold[i - 1], d.threshold[i])
    end
end

"""
    mid_point(d::Discretization, i::Int)

Computes the midpoint of the i-th interval. For the infinite intervals at the boundaries,
uses an extrapolation based on the spacing of adjacent finite intervals.

# Arguments
- `d::Discretization`: The discretization
- `i::Int`: Index of the interval

# Returns
- `Float64`: The midpoint (or approximated midpoint for infinite intervals)
"""
function mid_point(d::Discretization, i::Int)
    if i == 1
        # Extrapolate to the left
        return d.threshold[1] - (d.threshold[2] - d.threshold[1]) / 2
    elseif i == length(d.threshold) + 1
        # Extrapolate to the right
        return d.threshold[i - 1] + (d.threshold[i - 1] - d.threshold[i - 2]) / 2
    else
        # Regular midpoint
        return (d.threshold[i - 1] + d.threshold[i]) / 2
    end
end

"""
    find_interval(d::Discretization, x::Float64)

Finds the interval index containing the value x using binary search.

# Arguments
- `d::Discretization`: The discretization
- `x::Float64`: The value to locate

# Returns
- `Int`: The index of the interval containing x
"""
function find_interval(d::Discretization, x::Float64)
    l = 1
    r = length(d.threshold)
    while l < r
        m = div(l + r, 2)
        if x < d.threshold[m]
            r = m
        else
            l = m + 1
        end
    end
    return l
end

"""
    plot_factor_function(discrete::Discretization, f; x_title="x", y_title="f(x)")

Creates a bar plot visualization of a discretized function.

# Arguments
- `discrete::Discretization`: The discretization defining the x-axis intervals
- `f::Vector{Float64}`: Function values for each interval
- `x_title::String`: Label for x-axis (default: "x")
- `y_title::String`: Label for y-axis (default: "f(x)")
"""
function plot_factor_function(discrete::Discretization, f; x_title="x", y_title="f(x)")
    p = bar(
        [mid_point(discrete, i) for i in 1:length(discrete)],
        f,
        legend=false,
        color=:blue,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
    )
    xlabel!(x_title)
    ylabel!(y_title)
    display(p)
end


"""
    compute_messages(discrete::Discretization, μ1::Float64, σ1::Float64, μ2::Float64, σ2::Float64, β::Float64)

Computes all messages for the two-player TrueSkill factor graph using discretized belief propagation.

The factor graph structure:
- f₁(s₁) = N(s₁; μ₁, σ₁) - Prior on skill of player 1
- f₂(s₂) = N(s₂; μ₂, σ₂) - Prior on skill of player 2
- g₁(s₁, p₁) = N(p₁; s₁, β) - Performance given skill for player 1
- g₂(s₂, p₂) = N(p₂; s₂, β) - Performance given skill for player 2
- h(p₁, p₂) = I(p₁ > p₂) - Indicator that player 1 wins

# Arguments
- `discrete::Discretization`: Discretization of the continuous variables
- `μ1::Float64`: Mean of player 1's skill prior
- `σ1::Float64`: Standard deviation of player 1's skill prior
- `μ2::Float64`: Mean of player 2's skill prior
- `σ2::Float64`: Standard deviation of player 2's skill prior
- `β::Float64`: Performance noise standard deviation

# Returns
- Tuple of (factors, messages) where:
  - factors: (f1, f2, g1, g2, h) - discretized factor functions
  - messages: (msg_f1_s1, msg_f2_s2, msg_g1_p1, msg_g2_p2, msg_h_p1, msg_h_p2, msg_g1_s1, msg_g2_s2)
"""
function compute_messages(discrete::Discretization, μ1::Float64, σ1::Float64, μ2::Float64, σ2::Float64, β::Float64)
    # Initialize factor functions
    # f₁ and f₂: probability mass in each interval for skill priors
    f1 = [cdf(Normal(μ1, σ1), b) - cdf(Normal(μ1, σ1), a) for (a, b) in discrete]
    f2 = [cdf(Normal(μ2, σ2), b) - cdf(Normal(μ2, σ2), a) for (a, b) in discrete]
    
    # g₁ and g₂: performance likelihood given skill (discretized)
    g1 = zeros(length(discrete), length(discrete))
    g2 = zeros(length(discrete), length(discrete))
    
    # h: indicator function for player 1 winning (p₁ > p₂)
    h = zeros(length(discrete), length(discrete))
    
    for i in eachindex(discrete)
        for (j, (c, d)) in enumerate(discrete)
            h[i, j] = (i > j) ? 1.0 : 0.0
            m = mid_point(discrete, i)
            # Probability of performance in interval j given skill at midpoint m
            g1[i, j] = g2[i, j] = cdf(Normal(m, β), d) - cdf(Normal(m, β), c)
        end
    end 

    # Message passing: forward direction (from priors to comparison)
    msg_f1_s1 = deepcopy(f1)
    msg_f2_s2 = deepcopy(f2)

    # Messages from g factors to performance variables
    msg_g1_p1 = zeros(length(discrete))
    for p1 in eachindex(discrete)
        msg_g1_p1[p1] = sum(g1[:, p1] .* msg_f1_s1)
    end
    msg_g2_p2 = zeros(length(discrete))
    for p2 in eachindex(discrete)
        msg_g2_p2[p2] = sum(g2[:, p2] .* msg_f2_s2)
    end

    # Messages from comparison factor h to performance variables
    msg_h_p1 = zeros(length(discrete))
    for p1 in eachindex(discrete)
        msg_h_p1[p1] = sum(h[p1, :] .* msg_g2_p2)
    end
    msg_h_p2 = zeros(length(discrete))
    for p2 in eachindex(discrete)
        msg_h_p2[p2] = sum(h[:, p2] .* msg_g1_p1)
    end

    # Backward messages from g factors to skill variables
    msg_g1_s1 = zeros(length(discrete))
    for s1 in eachindex(discrete)
        msg_g1_s1[s1] = sum(g1[s1, :] .* msg_h_p1)
    end
    msg_g2_s2 = zeros(length(discrete))
    for s2 in eachindex(discrete)
        msg_g2_s2[s2] = sum(g2[s2, :] .* msg_h_p2)
    end

    return (f1, f2, g1, g2, h), (msg_f1_s1, msg_f2_s2, msg_g1_p1, msg_g2_p2, msg_h_p1, msg_h_p2, msg_g1_s1, msg_g2_s2)
end

"""
    main(n=100, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=0.5; base="~/Downloads/norm_")

Main function that runs the discretized belief propagation, generates plots, and computes
normalization constants.

# Arguments
- `n::Int`: Number of discretization intervals (default: 100)
- `μ1::Float64`: Mean of player 1's skill (default: 0.0)
- `σ1::Float64`: Std dev of player 1's skill (default: 1.0)
- `μ2::Float64`: Mean of player 2's skill (default: 0.0)
- `σ2::Float64`: Std dev of player 2's skill (default: 1.0)
- `β::Float64`: Performance noise (default: 0.5)
- `base::String`: Base path for saving plots (default: "~/Downloads/norm_")

# Output
- Saves visualization plots to files
- Prints normalization constants for variables and factors
"""
function main(n = 100, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=0.5; base="~/Downloads/norm_")
    # Create discretization
    thresholds = collect(range(start=-6.0, stop=6.0, length=n))
    dt = Discretization(thresholds)
    
    # Run message passing
    (_, _, g1, g2, h), (msg_f1_s1, msg_f2_s2, msg_g1_p1, msg_g2_p2, msg_h_p1, msg_h_p2, msg_g1_s1, msg_g2_s2) = compute_messages(dt, μ1, σ1, μ2, σ2, β)
    
    # Plot and save forward messages
    plot_factor_function(dt, msg_f1_s1, x_title = L"s_1", y_title = L"m_{f_1 \to s_1}")
    savefig(base * "f1_s1.svg")
    plot_factor_function(dt, msg_f2_s2, x_title = L"s_2", y_title = L"m_{f_2 \to s_2}")
    savefig(base * "f2_s2.svg")
    plot_factor_function(dt, msg_g1_p1, x_title = L"p_1", y_title = L"m_{g_1 \to p_1}")
    savefig(base * "g1_p1.svg")
    plot_factor_function(dt, msg_g2_p2, x_title = L"p_2", y_title = L"m_{g_2 \to p_2}")
    savefig(base * "g2_p2.svg")
    
    # Plot and save messages from comparison factor
    plot_factor_function(dt, msg_h_p1, x_title = L"p_1", y_title = L"m_{h \to p_1}")
    savefig(base * "h_p1.svg")
    plot_factor_function(dt, msg_h_p2, x_title = L"p_2", y_title = L"m_{h \to p_2}")
    savefig(base * "h_p2.svg")
    
    # Plot and save backward messages
    plot_factor_function(dt, msg_g1_s1, x_title = L"s_1", y_title = L"m_{g_1 \to s_1}")
    savefig(base * "g1_s1.svg")
    plot_factor_function(dt, msg_g2_s2, x_title = L"s_2", y_title = L"m_{g_2 \to s_2}")
    savefig(base * "g2_s2.svg")
    
    # Compute and plot marginal distributions (unnormalized)
    s1 = msg_g1_s1 .* msg_f1_s1
    s2 = msg_g2_s2 .* msg_f2_s2
    p1 = msg_h_p1 .* msg_g1_p1
    p2 = msg_h_p2 .* msg_g2_p2
    
    plot_factor_function(dt, s1, x_title = L"s_1", y_title = L"p(s_1)")
    savefig(base * "s1.svg")
    plot_factor_function(dt, s2, x_title = L"s_2", y_title = L"p(s_2)")
    savefig(base * "s2.svg")
    plot_factor_function(dt, p1, x_title = L"p_1", y_title = L"p(p_1)")
    savefig(base * "p1.svg")
    plot_factor_function(dt, p2, x_title = L"p_2", y_title = L"p(p_2)")
    savefig(base * "p2.svg")
    
    # Print unnormalized marginal normalization constants
    println("\n" * "="^60)
    println("Unnormalized Marginal Normalization Constants")
    println("="^60)
    println("Z(s₁) = ", sum(s1))
    println("Z(s₂) = ", sum(s2))
    println("Z(p₁) = ", sum(p1))
    println("Z(p₂) = ", sum(p2))

    # Normalize all messages
    msg_f1_s1 /= sum(msg_f1_s1)
    msg_f2_s2 /= sum(msg_f2_s2)
    msg_g1_p1 /= sum(msg_g1_p1)
    msg_g2_p2 /= sum(msg_g2_p2)
    msg_h_p1 /= sum(msg_h_p1)
    msg_h_p2 /= sum(msg_h_p2)
    msg_g1_s1 /= sum(msg_g1_s1)
    msg_g2_s2 /= sum(msg_g2_s2)

    # Compute normalization constants per variable
    Z_s1 = sum(msg_g1_s1 .* msg_f1_s1)
    Z_s2 = sum(msg_g2_s2 .* msg_f2_s2)
    Z_p1 = sum(msg_h_p1 .* msg_g1_p1)
    Z_p2 = sum(msg_h_p2 .* msg_g2_p2)

    # Compute normalization constants per factor
    Z_f1 = sum(msg_f1_s1)
    Z_f2 = sum(msg_f2_s2)
    
    Z_g1 = 0.0
    for s1 in eachindex(dt)
        for p1 in eachindex(dt)
            Z_g1 += g1[s1, p1] * msg_f1_s1[s1] * msg_h_p1[p1]
        end
    end
    Z_g1 /= (Z_s1 * Z_p1)

    Z_g2 = 0.0
    for s2 in eachindex(dt)
        for p2 in eachindex(dt)
            Z_g2 += g2[s2, p2] * msg_f2_s2[s2] * msg_h_p2[p2]
        end
    end
    Z_g2 /= (Z_s2 * Z_p2)

    Z_h = 0.0
    for p1 in eachindex(dt)
        for p2 in eachindex(dt)
            Z_h += h[p1, p2] * msg_g1_p1[p1] * msg_g2_p2[p2]
        end
    end
    Z_h /= (Z_p1 * Z_p2)

    # Print detailed normalization analysis
    println("\n" * "="^60)
    println("Variable Normalization Constants (with normalized messages)")
    println("="^60)
    println("Z(s₁) = ", Z_s1)
    println("Z(s₂) = ", Z_s2)
    println("Z(p₁) = ", Z_p1)
    println("Z(p₂) = ", Z_p2)
    
    println("\n" * "="^60)
    println("Factor Normalization Constants")
    println("="^60)
    println("Z(f₁) = ", Z_f1)
    println("Z(f₂) = ", Z_f2)
    println("Z(g₁) = ", Z_g1)
    println("Z(g₂) = ", Z_g2)
    println("Z(h)  = ", Z_h)

    Z_total = Z_s1 * Z_s2 * Z_p1 * Z_p2 * Z_f1 * Z_f2 * Z_g1 * Z_g2 * Z_h
    println("\n" * "="^60)
    println("Overall Normalization Constant")
    println("="^60)
    println("Z_total = ∏ Z(variables) × ∏ Z(factors)")
    println("        = ", Z_total)
    println("="^60 * "\n")
end


end