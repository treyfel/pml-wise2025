"""
    DistributionCollections

A module for managing collections of probability distributions in factor graph inference.

# Overview

This module provides the `DistributionBag` data structure, which maintains a collection
of probability distributions (marginals) for variables in a factor graph. It serves as
a centralized storage and management system for belief propagation algorithms.

# DistributionBag Concept

The `DistributionBag` acts as a repository that:
- Stores marginal distributions for random variables
- Stores messages between factors and variables
- Provides indexed access to distributions
- Maintains a uniform distribution as default initialization

# Factor Graph Context

In factor graphs:
- Each **variable node** has an associated marginal distribution
- Each **edge** (factor-to-variable or variable-to-factor) has a message
- The DistributionBag stores all these distributions in a single container
- Indices are used to reference specific distributions

# Design Benefits

1. **Centralized storage**: All distributions in one place
2. **Automatic initialization**: New distributions start uniform
3. **Efficient updates**: In-place modification via indexing
4. **Type safety**: Parameterized by distribution type
5. **Iterable**: Supports standard Julia array operations

# Typical Usage Pattern

```julia
# Create bag with uniform default
db = DistributionBag(Discrete(5))

# Add marginal distributions for variables
x1 = add!(db)  # Returns index 1
x2 = add!(db)  # Returns index 2

# Add message slots for factors
msg_f1_to_x1 = add!(db)  # Returns index 3

# Access and update
db[x1] = some_distribution
current = db[x1]

# Reset all to uniform
reset!(db)
```

# Author
Ralf Herbrich, Hasso-Plattner Institute, 2025
"""
module DistributionCollections

export DistributionBag, add!, reset!

"""
    DistributionBag{T} <: AbstractArray{T, 1}

A dynamically-sized collection of probability distributions with uniform initialization.

# Type Parameters
- `T`: The type of distribution stored (e.g., `Discrete{5}`, `Gaussian`)

# Fields
- `uniform::T`: The default uniform distribution used for initialization
- `bag::Vector{T}`: The actual vector storing distributions

# Array Interface

DistributionBag implements the AbstractArray interface, supporting:
- Indexing: `db[i]`, `db[begin]`, `db[end]`
- Assignment: `db[i] = distribution`
- Iteration: `for d in db ... end`
- Size queries: `size(db)`, `length(db)`

# Invariants
- All distributions in the bag are of type T
- New distributions are initialized to `uniform`
- Indices are consecutive integers starting from 1

# Use Cases
- Storing variable marginals in factor graphs
- Storing messages in belief propagation
- Managing distributions for graphical model inference
"""
struct DistributionBag{T} <: AbstractArray{T, 1}
    uniform::T      # the uniform distribution of distribution type T
    bag::Vector{T}  # the actual resizeable vector that stores the distributions
    
    # Inner constructor ensures empty initialization
    DistributionBag{T}(uniform::T) where {T} = new(uniform, Vector{T}(undef, 0))
end

"""
    DistributionBag(uniform::T) -> DistributionBag{T}

Outer constructor for creating a distribution bag with a specified uniform distribution.

# Arguments
- `uniform::T`: The default distribution used when adding new elements

# Returns
- `DistributionBag{T}`: An empty collection ready to store distributions

# Examples
```julia-repl
julia> DistributionBag(Discrete(5))
0-element DistributionBag{Discrete{5}}

julia> db = DistributionBag(Discrete(3))
julia> typeof(db)
DistributionBag{Discrete{3}}
```

# Notes
- The bag starts empty; use `add!()` to add distributions
- All added distributions are initially set to `uniform`
- The type parameter T is inferred from the uniform distribution
"""
DistributionBag(uniform::T) where {T} = DistributionBag{T}(uniform)

"""
    show(io::IO, db::DistributionBag{T}) -> Nothing

Pretty-prints a distribution bag showing the uniform distribution and all stored distributions.

# Format
```
Uniform: <uniform distribution>
  [1]: <first distribution>
  [2]: <second distribution>
  ...
```

# Arguments
- `io::IO`: Output stream
- `db::DistributionBag{T}`: The distribution bag to print

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(5))
0-element DistributionBag{Discrete{5}}

julia> print(db)
Uniform:  P = [0.2, 0.2, 0.2, 0.2, 0.2]
  empty bag

julia> add!(db)
1

julia> print(db)
Uniform:  P = [0.2, 0.2, 0.2, 0.2, 0.2]
  [1]:  P = [0.2, 0.2, 0.2, 0.2, 0.2]
```

# Notes
- Shows "empty bag" when no distributions have been added
- Each distribution is numbered for easy reference
- Useful for debugging belief propagation algorithms
"""
function Base.show(io::IO, db::DistributionBag{T}) where {T}
    println(io, "Uniform: ", db.uniform)
    if (length(db.bag) == 0)
        println(io, "  empty bag")
    else
        for i in eachindex(db.bag)
            println(io, "  [", i, "]: ", db.bag[i])
        end
    end
end

"""
    add!(db::DistributionBag{T}) -> Int

Adds a new distribution to the bag and returns its index.

The new distribution is initialized to the uniform distribution specified
when creating the bag.

# Arguments
- `db::DistributionBag{T}`: The distribution bag to modify

# Returns
- `Int`: The index of the newly added distribution (1-based)

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(5))
0-element DistributionBag{Discrete{5}}

julia> x1 = add!(db)  # Add first distribution
1
  
julia> x2 = add!(db)  # Add second distribution
2

julia> length(db)
2
```

# Usage Pattern in Factor Graphs

```julia
# Create bag
db = DistributionBag(Discrete(3))

# Add marginals for variables
var1 = add!(db)
var2 = add!(db)

# Add message slots for a factor
msg_to_var1 = add!(db)
msg_to_var2 = add!(db)

# Now can reference distributions by index
db[var1] = some_belief
```

# Notes
- Always initializes to uniform distribution
- Indices are consecutive starting from 1
- Commonly used at the start of inference to allocate space
"""
function add!(db::DistributionBag{T}) where {T}
    push!(db.bag, db.uniform)
    return (length(db.bag))
end

"""
    reset!(db::DistributionBag{T}) -> Nothing

Resets all distributions in the bag to the uniform distribution.

Useful for re-initializing the factor graph while keeping the same
variable and message structure.

# Arguments
- `db::DistributionBag{T}`: The distribution bag to reset

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(5))
julia> idx = add!(db)
julia> db[idx] = Discrete([0.0, 1.0, 1.0, 2.0, -1.0])
 P = [0.0705, 0.1915, 0.1915, 0.5206, 0.0259]

julia> reset!(db)

julia> db[idx]
 P = [0.2, 0.2, 0.2, 0.2, 0.2]  # Back to uniform
```

# Use Cases
- Restarting belief propagation with same graph structure
- Clearing evidence while maintaining variable indices
- Testing different inference scenarios

# Notes
- Preserves the number of distributions
- Does not change indices
- All distributions become identical (uniform)
"""
function reset!(db::DistributionBag{T}) where {T}
    for i in eachindex(db.bag)
        db.bag[i] = db.uniform
    end
    return
end

"""
    getindex(db::DistributionBag{T}, i::Int64) -> T

Retrieves the distribution at the specified index.

Part of the AbstractArray interface implementation.

# Arguments
- `db::DistributionBag{T}`: The distribution bag
- `i::Int64`: The index (1-based)

# Returns
- `T`: The distribution at index i

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(5))
julia> idx = add!(db)
julia> db[idx]
 P = [0.2, 0.2, 0.2, 0.2, 0.2]

julia> db[1]  # Same as db[idx]
 P = [0.2, 0.2, 0.2, 0.2, 0.2]
```

# Notes
- Bounds checking is automatic via Vector
- Supports `begin` and `end` keywords
"""
Base.getindex(db::DistributionBag{T}, i::Int64) where {T} = db.bag[i]

"""
    setindex!(db::DistributionBag{T}, d::T, i::Int64) -> T

Sets the distribution at the specified index.

Part of the AbstractArray interface implementation. Used to update
marginals and messages during belief propagation.

# Arguments
- `db::DistributionBag{T}`: The distribution bag
- `d::T`: The new distribution to store
- `i::Int64`: The index (1-based)

# Returns
- `T`: The distribution that was set

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(5))
julia> idx = add!(db)
julia> db[idx] = Discrete([0.0, 1.0, 1.0, 2.0, -1.0])
 P = [0.0705, 0.1915, 0.1915, 0.5206, 0.0259]
```

# Use Cases
- Updating marginal distributions during inference
- Storing computed messages
- Incorporating evidence into the graph

# Notes
- Type must match the bag's type parameter T
- In-place modification of the collection
"""
function Base.setindex!(db::DistributionBag{T}, d::T, i::Int64) where {T}
    db.bag[i] = d
end

"""
    firstindex(db::DistributionBag{T}) -> Int

Returns the index of the first element (always 1).

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(3))
julia> add!(db); add!(db)
julia> db[firstindex(db)]
 P = [0.3333, 0.3333, 0.3333]
```
"""
Base.firstindex(db::DistributionBag{T}) where {T} = return (1)

"""
    lastindex(db::DistributionBag{T}) -> Int

Returns the index of the last element.

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(3))
julia> add!(db); add!(db)
julia> lastindex(db)
2
```
"""
Base.lastindex(db::DistributionBag{T}) where {T} = return (length(db.bag))

"""
    size(db::DistributionBag{T}) -> Tuple{Int}

Returns the size of the distribution bag as a 1-tuple.

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(3))
julia> add!(db); add!(db)
julia> size(db)
(2,)
```
"""
Base.size(db::DistributionBag{T}) where {T} = (length(db.bag),)

"""
    IndexStyle(::Type{<:DistributionBag{T}}) -> IndexLinear

Indicates that DistributionBag uses linear indexing.
"""
Base.IndexStyle(::Type{<:DistributionBag{T}}) where {T} = IndexLinear()

"""
    eltype(::Type{<:DistributionBag{T}}) -> Type{T}

Returns the element type (distribution type) of the bag.

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(5))
julia> eltype(db)
Discrete{5}
```
"""
Base.eltype(::Type{<:DistributionBag{T}}) where {T} = T

"""
    iterate(db::DistributionBag, i=1) -> Union{Tuple{T, Int}, Nothing}

Implements iteration over distributions in the bag.

Enables use of DistributionBag in loops and comprehensions.

# Examples
```julia-repl
julia> db = DistributionBag(Discrete(3))
julia> add!(db); add!(db)
julia> db[1] = Discrete([0.0, 1.0, 1.0])

julia> for d in db
           println(d)
       end
 P = [0.0900, 0.2447, 0.6652]
 P = [0.3333, 0.3333, 0.3333]
```

# Notes
- Follows standard Julia iterator protocol
- Returns (element, state) or nothing
"""
Base.iterate(db::DistributionBag{T}, i=1) where {T} = (i > length(db.bag)) ? nothing : (db.bag[i], i+1)

end