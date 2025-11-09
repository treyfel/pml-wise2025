"""
    DiscreteDistribution

A module for representing and manipulating discrete probability distributions
over finite state spaces using log-space arithmetic for numerical stability.

# Overview

This module implements discrete probability distributions over a finite set of
states {1, 2, ..., n} using log-probabilities to avoid numerical underflow
issues common in probabilistic computations.

# Log-Space Representation

All probabilities are stored as log-probabilities: logP[i] = log(P[i])

Benefits:
- **Numerical stability**: Prevents underflow for very small probabilities
- **Efficient multiplication**: P₁ × P₂ becomes logP₁ + logP₂
- **Efficient division**: P₁ / P₂ becomes logP₁ - logP₂
- **Normalization**: Uses log-sum-exp trick for stable computation

# Mathematical Operations

- **Multiplication**: (p * q)[i] ∝ p[i] × q[i] (element-wise product)
- **Division**: (p / q)[i] ∝ p[i] / q[i] (element-wise quotient)
- **Normalization**: Automatic via the ℙ function

# Type System

The distribution type `Discrete{T}` is parameterized by the number of states T,
enabling compile-time type checking and optimization.

# Applications

- Factor graph message passing
- Bayesian inference over discrete variables
- Hidden Markov Models
- Graphical model inference

# Author
Ralf Herbrich, Hasso-Plattner Institute, 2025
"""
module DiscreteDistribution

export Discrete, ℙ, *, /

using LogExpFunctions

"""
    Discrete{T}

A discrete probability distribution over T states, represented in log-space.

# Fields
- `logP::Vector{Float64}`: Log-probabilities for each state

# Type Parameter
- `T`: Number of discrete states (fixed at compile time)

# Invariants
- length(logP) == T
- Normalization is handled implicitly by the ℙ function

# Notes
- Probabilities are stored as logarithms for numerical stability
- Operations maintain unnormalized distributions; normalize via ℙ() when needed
"""
struct Discrete{T} 
    logP::Vector{Float64}
end

"""
    Discrete(logP::Vector{Float64}) -> Discrete{length(logP)}

Creates a discrete distribution from a vector of log-probabilities.

The distribution is strongly typed with the number of elements, enabling
compile-time optimization and type safety.

# Arguments
- `logP::Vector{Float64}`: Log-probabilities for each state

# Returns
- `Discrete{T}`: A discrete distribution over T states

# Examples
```jldoctest
julia> Discrete([0.0, 0.0])  # Two states with equal log-prob
Discrete{2}([0.0, 0.0])

julia> Discrete([0.0, 0.0, 1.0])  # Three states
Discrete{3}([0.0, 0.0, 1.0])

julia> Discrete([log(0.3), log(0.7)])  # Explicit log probabilities
 P = [0.3, 0.7]
```

# Notes
- Log-probabilities can be unnormalized
- Use ℙ() to get normalized probabilities
- Larger log values correspond to higher probabilities
""" 
Discrete(logP::Vector{Float64}) = Discrete{length(logP)}(logP)

"""
    Discrete(n::Int64) -> Discrete{n}

Creates a uniform discrete distribution over n states.

Constructs a distribution where all states have equal probability 1/n.

# Arguments
- `n::Int64`: Number of states

# Returns
- `Discrete{n}`: A uniform distribution over states {1, ..., n}

# Examples
```jldoctest
julia> Discrete(2)  # Uniform over 2 states
 P = [0.5, 0.5]

julia> Discrete(3)  # Uniform over 3 states
 P = [0.3333, 0.3333, 0.3333]

julia> Discrete(5)
 P = [0.2, 0.2, 0.2, 0.2, 0.2]
```

# Implementation
- Uses zero log-probabilities (log(1) = 0) for all states
- After normalization: P[i] = 1/n for all i

# Use Cases
- Default initialization for message passing
- Non-informative prior distributions
- Initial marginal distributions in factor graphs
""" 
Discrete(n::Int64) = Discrete{n}(zeros(Float64, n))

"""
    *(p::Discrete{T}, q::Discrete{T}) -> Discrete{T}

Computes the element-wise product of two discrete distributions.

This operation corresponds to combining independent evidence about a variable
in factor graphs, or computing the product of factors in probabilistic inference.

# Mathematical Operation
(p * q)[i] ∝ p[i] × q[i] for all states i

In log-space: logP[i] = logP_p[i] + logP_q[i]

# Arguments
- `p::Discrete{T}`: First distribution
- `q::Discrete{T}`: Second distribution

# Returns
- `Discrete{T}`: Unnormalized product distribution

# Examples
```jldoctest
julia> p = Discrete([0.0, 2.0])  # Favors state 2
julia> q = Discrete([1.0, 0.0])  # Favors state 1
julia> p * q
 P = [0.2689, 0.7311]  # Compromise between both

julia> Discrete([0.0, 2.0, -1.0]) * Discrete([1.0, 0.0, 1.0])
 P = [0.2447, 0.6652, 0.0900]
```

# Use Cases
- Combining messages in belief propagation
- Incorporating evidence in Bayesian inference
- Factor multiplication in graphical models

# Notes
- Result is automatically unnormalized
- Use ℙ() to obtain normalized probabilities
- Commutative: p * q == q * p
""" 
function Base.:*(p::Discrete{T}, q::Discrete{T})::Discrete{T} where {T}
    Discrete(p.logP .+ q.logP)
end

"""
    /(p::Discrete{T}, q::Discrete{T}) -> Discrete{T}

Computes the element-wise quotient of two discrete distributions.

This operation is used to "divide out" old messages in belief propagation,
allowing message updates without recomputing from scratch.

# Mathematical Operation
(p / q)[i] ∝ p[i] / q[i] for all states i

In log-space: logP[i] = logP_p[i] - logP_q[i]

# Arguments
- `p::Discrete{T}`: Numerator distribution
- `q::Discrete{T}`: Denominator distribution

# Returns
- `Discrete{T}`: Unnormalized quotient distribution

# Examples
```jldoctest
julia> p = Discrete([1.0, 2.0])
julia> q = Discrete([1.0, 0.0])
julia> p / q
 P = [0.1192, 0.8808]

julia> Discrete([2.0, 0.0, -1.0]) / Discrete([1.0, 0.0, 1.0])
 P = [0.7054, 0.2595, 0.0351]
```

# Use Cases
- Removing old messages in belief propagation
- Computing marginals in factor graphs: marginal = (old_marginal / old_msg) * new_msg
- Incremental belief updates

# Notes
- Division by zero in log-space (q[i] → 0) leads to -∞
- Result is unnormalized; use ℙ() for probabilities
- Not commutative: p / q ≠ q / p

# Warning
- Ensure denominator q has non-zero probability where needed
- Numerical issues may arise with extreme log-probability differences
""" 
function Base.:/(p::Discrete{T}, q::Discrete{T})::Discrete{T} where {T}
    Discrete(p.logP .- q.logP)
end

"""
    ℙ(p::Discrete{T}) -> Vector{Float64}

Computes normalized probabilities from log-probabilities using the log-sum-exp trick.

Converts the internal log-space representation to actual probabilities that sum to 1,
while maintaining numerical stability for extreme values.

# Mathematical Operation
P[i] = exp(logP[i]) / Σⱼ exp(logP[j])

# Arguments
- `p::Discrete`: Distribution in log-space

# Returns
- `Vector{Float64}`: Normalized probabilities summing to 1.0

# Examples
```jldoctest
julia> ℙ(Discrete([0.0, 0.0]))  # Equal log-probs → equal probs
2-element Vector{Float64}:
 0.5
 0.5
 
julia> ℙ(Discrete([1000.0, 1000.0]))  # Large values handled stably
2-element Vector{Float64}:
 0.5
 0.5
 
julia> ℙ(Discrete([1.0, -1.0]))  # Different log-probs
2-element Vector{Float64}:
 0.8808
 0.1192

julia> ℙ(Discrete([0.0, 1.0, 2.0]))  # Increasing preference
 [0.0900, 0.2447, 0.6652]
```

# Numerical Stability

Uses the log-sum-exp trick to prevent overflow/underflow:
1. Compute logZ = log(Σᵢ exp(logP[i]))
2. Return exp(logP[i] - logZ) for each i

This is stable even when:
- Log-probabilities are very large (e.g., 1000.0)
- Log-probabilities are very small (e.g., -1000.0)
- There's a large range in values

# Properties
- Always returns probabilities in [0, 1]
- Sum of returned probabilities equals 1.0 (within floating-point precision)
- Preserves relative ordering of probabilities

# Performance
- O(T) time complexity where T is the number of states
- Two passes through the array (one for logsumexp, one for normalization)
""" 
function ℙ(p::Discrete)
    # Compute normalization constant using log-sum-exp trick
    logZ = logsumexp(p.logP)
    # Return normalized probabilities
    return (exp.(p.logP .- logZ))
end

"""
    show(io::IO, p::Discrete) -> Nothing

Pretty-prints a discrete distribution showing normalized probabilities.

Formats the distribution as "P = [p₁, p₂, ..., pₙ]" with probabilities
rounded to 4 decimal places for readability.

# Arguments
- `io::IO`: Output stream
- `p::Discrete`: Distribution to print

# Examples
```jldoctest
julia> Discrete([0.0, 1.0])
 P = [0.2689, 0.7311]

julia> Discrete(3)
 P = [0.3333, 0.3333, 0.3333]

julia> Discrete([log(0.1), log(0.2), log(0.7)])
 P = [0.1, 0.2, 0.7]
```

# Notes
- Automatically normalizes before printing
- Rounds to 4 decimal places for cleaner output
- Useful for debugging and interactive exploration
"""
function Base.show(io::IO, p::Discrete)
    probs = ℙ(p)
    print(io, " P = [")
    for i in eachindex(probs)
        print(io, round(probs[i], digits=4))
        if (i < length(probs))
            print(io, ", ")
        end
    end
    print(io, "]")
end

end