"""
    Factors

A module implementing factor nodes and message passing operations for discrete
factor graph inference using the sum-product (belief propagation) algorithm.

# Overview

This module provides factor types and message update operations for performing
exact inference on discrete factor graphs. It implements the sum-product message
passing algorithm for computing marginal distributions.

# Factor Graphs

A factor graph is a bipartite graph with two types of nodes:
- **Variable nodes**: Represent random variables
- **Factor nodes**: Represent local functions of variables

Edges connect factors to the variables they depend on.

# Sum-Product Algorithm

The sum-product algorithm computes marginal distributions by passing messages:
1. Each factor sends messages to its neighboring variables
2. Each variable sends messages to its neighboring factors  
3. Messages are distributions over variable domains
4. Marginals are computed by multiplying incoming messages

# Message Passing Rules

**Variable to Factor**: Product of all other incoming messages
**Factor to Variable**: Sum-product over factor's other variables

# Factors Implemented

## PriorDiscreteFactor
Represents a prior distribution: p(x)
- Connected to one variable
- Sends the prior distribution as a message

## CouplingDiscreteFactor  
Represents a conditional distribution: p(y|x)
- Connected to two variables (x and y)
- Stores transition/coupling matrix P where P[i,j] = p(y=i|x=j)
- Performs marginalization when sending messages

# Message Update Functions

- `update_msg_to_x!(factor)`: Computes and sends message to variable x
- `update_msg_to_y!(factor)`: Computes and sends message to variable y

# Mathematical Foundations

## Marginal Computation
The marginal distribution for variable x is:
```
p(x) ∝ ∏_{f ∈ neighbors(x)} msg_{f→x}(x)
```

## Message from Prior Factor
```
msg_{prior→x}(x) = p(x)
```

## Message from Coupling Factor
```
msg_{f→x}(x) = Σ_y p(y|x) × msg_{y→f}(y)
```

# Belief Propagation Algorithm

On a tree-structured graph:
1. Initialize all messages to uniform
2. Root the tree at a variable
3. Pass messages from leaves to root
4. Pass messages from root to leaves
5. Compute marginals by multiplying messages

# Author
Ralf Herbrich, Hasso-Plattner Institute, 2025

# References
- Kschischang, F. R., Frey, B. J., & Loeliger, H. A. (2001). 
  Factor graphs and the sum-product algorithm. IEEE Transactions on Information Theory.
"""
module Factors

export PriorDiscreteFactor, CouplingDiscreteFactor, update_msg_to_x!, update_msg_to_y!

using ..DistributionCollections
using ..DiscreteDistribution

"""
    PriorDiscreteFactor{T}

A factor representing a prior distribution over a discrete variable.

# Mathematical Representation
Encodes: f(x) = p(x) where p(x) is the prior distribution

# Type Parameters
- `T`: Number of discrete states for the variable

# Fields
- `db::DistributionBag{Discrete{T}}`: The shared distribution storage
- `x::Int64`: Index of the variable this factor is attached to
- `prior::Discrete{T}`: The prior distribution p(x)
- `msg_to_x::Int64`: Index where the message to x is stored in db

# Factor Graph Role
This is a degree-1 factor (connected to one variable) that injects
prior knowledge into the graph.

# Message Passing
Sends message: msg_{f→x}(x) = p(x)

Updates marginal: p(x) ← p(x) × prior(x) / old_msg_{f→x}(x)
"""
struct PriorDiscreteFactor{T}
    db::DistributionBag{Discrete{T}}
    x::Int64
    prior::Discrete{T}
    msg_to_x::Int64
end

"""
    PriorDiscreteFactor(db, x, prior) -> PriorDiscreteFactor{T}

Creates a prior factor for a discrete variable with a given prior distribution.

# Arguments
- `db::DistributionBag{Discrete{T}}`: The shared distribution storage
- `x::Int64`: Index of the variable in the distribution bag
- `prior::Discrete{T}`: The prior distribution p(x)

# Returns
- `PriorDiscreteFactor{T}`: A factor encoding prior knowledge

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(3))
julia> var = add!(db)
julia> # Prior strongly favoring state 2
julia> prior_factor = PriorDiscreteFactor(db, var, Discrete([0.0, 100.0, 0.0]))
```

# Notes
- Automatically allocates a slot for the outgoing message
- Prior should be in log-space (use Discrete constructor appropriately)
- Common use: incorporating observed data or domain knowledge
"""
PriorDiscreteFactor(db::DistributionBag{Discrete{T}}, x::Int64, prior) where {T} = 
    PriorDiscreteFactor{T}(db, x, prior, add!(db))

"""
    update_msg_to_x!(f::PriorDiscreteFactor{T}) -> Discrete{T}

Updates the message from a prior factor to its variable and updates the marginal.

# Algorithm
1. Compute incoming message: incoming = current_marginal / old_message
2. Update marginal: new_marginal = incoming × prior
3. Update stored message: message ← prior

# Arguments
- `f::PriorDiscreteFactor{T}`: The prior factor to update

# Returns
- `Discrete{T}`: The new message (also stored in the distribution bag)

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(3))
julia> var = add!(db)
julia> f = PriorDiscreteFactor(db, var, Discrete([0.0, 100.0, 0.0]))

julia> update_msg_to_x!(f)
 P = [0.0, 1.0, 0.0]  # Variable now strongly believes state 2

julia> db[var]  # Check marginal
 P = [0.0, 1.0, 0.0]
```

# Effect on Marginal
After this update, the variable's marginal incorporates the prior:
```
p(x) ∝ prior(x) × messages_from_other_factors(x)
```

# Use in Belief Propagation
- Call once per prior factor
- Typically called at the start of inference
- No need to call repeatedly (prior doesn't change)
"""
function update_msg_to_x!(f::PriorDiscreteFactor{T}) where {T}
    # Divide out the old message from the current marginal
    incoming_msg = f.db[f.x] / f.db[f.msg_to_x]
    # Update marginal: multiply incoming message by prior
    f.db[f.x] = incoming_msg * f.prior
    # Store the new message to x
    f.db[f.msg_to_x] = f.prior
end

"""
    CouplingDiscreteFactor{T}

A factor representing a conditional distribution coupling two discrete variables.

# Mathematical Representation
Encodes: f(x, y) = p(y|x) where P[i,j] = p(y=i | x=j)

# Type Parameters
- `T`: Number of discrete states for both variables

# Fields
- `db::DistributionBag{Discrete{T}}`: The shared distribution storage
- `x::Int64`: Index of first variable (typically "parent")
- `y::Int64`: Index of second variable (typically "child")
- `P::Matrix{Float64}`: Conditional probability matrix (T×T)
- `msg_to_x::Int64`: Index for message from factor to x
- `msg_to_y::Int64`: Index for message from factor to y

# Matrix Convention
P[i,j] = p(y=i | x=j)
- Columns correspond to values of x (conditioning variable)
- Rows correspond to values of y (dependent variable)
- Each column should sum to 1 (valid conditional distribution)

# Factor Graph Role
This is a degree-2 factor connecting two variables, typically used for:
- Transition probabilities in chains/HMMs
- Parent-child relationships in Bayesian networks
- Pairwise potentials in Markov Random Fields

# Message Passing
- **To x**: msg_{f→x}(x) = Σ_y p(y|x) × msg_{y→f}(y)
- **To y**: msg_{f→y}(y) = Σ_x p(y|x) × msg_{x→f}(x)
"""
struct CouplingDiscreteFactor{T}
    db::DistributionBag{Discrete{T}}
    x::Int64
    y::Int64
    P::Matrix{Float64}
    msg_to_x::Int64
    msg_to_y::Int64
end

"""
    CouplingDiscreteFactor(db, x, y, P) -> CouplingDiscreteFactor{T}

Creates a coupling factor between two discrete variables.

# Arguments
- `db::DistributionBag{Discrete{T}}`: The shared distribution storage
- `x::Int64`: Index of first variable
- `y::Int64`: Index of second variable
- `P::Matrix{Float64}`: Conditional probability matrix (T×T)

# Returns
- `CouplingDiscreteFactor{T}`: A factor coupling x and y

# Matrix Format
P[i,j] = p(y=i | x=j) in probability space (not log-space)

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(3))
julia> var1 = add!(db)
julia> var2 = add!(db)

julia> # Transition matrix: slight preference to stay in same state
julia> P = [0.5 0.25 0.25; 
           0.25 0.5 0.25; 
           0.25 0.25 0.5]

julia> coupling = CouplingDiscreteFactor(db, var1, var2, P)
```

# Common Patterns

**Identity coupling** (y likely equals x):
```julia
P = Matrix(I, 3, 3)  # [1 0 0; 0 1 0; 0 0 1]
```

**Uniform coupling** (y independent of x):
```julia
P = ones(3, 3) / 3  # Each row sums to 1
```

**Chain with noise**:
```julia
P = [0.7 0.15 0.15;
     0.15 0.7 0.15;
     0.15 0.15 0.7]
```

# Notes
- Automatically allocates slots for two outgoing messages
- Matrix is stored in probability space, not log-space
- Each column should be a valid probability distribution
"""
CouplingDiscreteFactor(db::DistributionBag{Discrete{T}}, x::Int64, y::Int64, P::Matrix{Float64}) where {T} = 
    CouplingDiscreteFactor{T}(db, x, y, P, add!(db), add!(db))

"""
    update_msg_to_x!(f::CouplingDiscreteFactor{T}) -> Discrete{T}

Updates the message from a coupling factor to variable x via marginalization.

# Algorithm
Computes: msg_{f→x}(x=j) = Σᵢ p(y=i|x=j) × msg_{y→f}(y=i)

Steps:
1. Get incoming message from y (divide out old message from marginal)
2. Marginalize over y using conditional probability matrix
3. Create new message to x
4. Update x's marginal: p(x) ← (p(x) / old_msg) × new_msg
5. Store new message

# Arguments
- `f::CouplingDiscreteFactor{T}`: The coupling factor to update

# Returns
- `Discrete{T}`: The new message to x

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(3))
julia> x = add!(db)
julia> y = add!(db)
julia> P = [0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5]'
julia> f = CouplingDiscreteFactor(db, x, y, P)

julia> # Set up some belief about y
julia> db[y] = Discrete([0.0, 10.0, 0.0])  # y likely in state 2

julia> update_msg_to_x!(f)
 P = [0.25, 0.5, 0.25]  # x's belief updated based on y
```

# Mathematical Details

The message computation performs:
```
msg[j] = Σᵢ P[i,j] × incoming_from_y[i]
```

This marginalizes out y while accounting for the coupling.

# When to Call
- After updating y's marginal
- During backward pass in belief propagation
- When propagating information from y to x
"""
function update_msg_to_x!(f::CouplingDiscreteFactor{T}) where {T}
    # Get the incoming message from y by dividing out the old message
    incoming_msg_probs = ℙ(f.db[f.y] / f.db[f.msg_to_y])
    
    # Compute new message by marginalizing over y
    # msg_to_x[j] = Σᵢ P[i,j] × incoming_msg[i]
    new_msg_to_x_probs = Vector{Float64}(undef, T)
    for i in 1:T
        # Sum over all states of y, weighted by conditional probability
        new_msg_to_x_probs[i] = sum(incoming_msg_probs .* f.P[:, i])
    end
    
    # Convert to log-space distribution
    new_msg_to_x = Discrete(log.(new_msg_to_x_probs))
    
    # Update x's marginal: divide out old message, multiply in new message
    f.db[f.x] = f.db[f.x] / f.db[f.msg_to_x] * new_msg_to_x
    
    # Store the new message
    f.db[f.msg_to_x] = new_msg_to_x
end

"""
    update_msg_to_y!(f::CouplingDiscreteFactor{T}) -> Discrete{T}

Updates the message from a coupling factor to variable y via marginalization.

# Algorithm
Computes: msg_{f→y}(y=i) = Σⱼ p(y=i|x=j) × msg_{x→f}(x=j)

Steps:
1. Get incoming message from x (divide out old message from marginal)
2. Marginalize over x using conditional probability matrix
3. Create new message to y
4. Update y's marginal: p(y) ← (p(y) / old_msg) × new_msg
5. Store new message

# Arguments
- `f::CouplingDiscreteFactor{T}`: The coupling factor to update

# Returns
- `Discrete{T}`: The new message to y

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(3))
julia> x = add!(db)
julia> y = add!(db)
julia> P = [0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5]'
julia> f = CouplingDiscreteFactor(db, x, y, P)

julia> # Set up some belief about x
julia> db[x] = Discrete([10.0, 0.0, 0.0])  # x likely in state 1

julia> update_msg_to_y!(f)
 P = [0.5, 0.25, 0.25]  # y's belief updated based on x
```

# Mathematical Details

The message computation performs:
```
msg[i] = Σⱼ P[i,j] × incoming_from_x[j]
```

This propagates information from x to y through the coupling.

# When to Call
- After updating x's marginal
- During forward pass in belief propagation
- When propagating information from x to y

# Symmetry with update_msg_to_x!
These two functions are symmetric, differing only in which variable
is marginalized over and which direction the message is sent.
"""
function update_msg_to_y!(f::CouplingDiscreteFactor{T}) where {T}
    # Get the incoming message from x by dividing out the old message
    incoming_msg_probs = ℙ(f.db[f.x] / f.db[f.msg_to_x])
    
    # Compute new message by marginalizing over x
    # msg_to_y[i] = Σⱼ P[i,j] × incoming_msg[j]
    new_msg_to_y_probs = Vector{Float64}(undef, T)
    for i in 1:T
        # Sum over all states of x, weighted by conditional probability
        new_msg_to_y_probs[i] = sum(incoming_msg_probs .* f.P[i, :])
    end
    
    # Convert to log-space distribution
    new_msg_to_y = Discrete(log.(new_msg_to_y_probs))
    
    # Update y's marginal: divide out old message, multiply in new message
    f.db[f.y] = f.db[f.y] / f.db[f.msg_to_y] * new_msg_to_y
    
    # Store the new message
    f.db[f.msg_to_y] = new_msg_to_y
end

end