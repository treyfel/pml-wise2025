"""
Chain Factor Graph Examples

Demonstrates belief propagation (sum-product algorithm) on chain-structured
factor graphs with discrete variables.

# Overview

This file provides educational examples of exact inference on linear chain
factor graphs using the sum-product message passing algorithm.

# Chain Structure

A chain graph has the structure:
```
Y₁ --- Y₂ --- Y₃ --- ... --- Yₙ
```

Where:
- Yᵢ are random variables (nodes)
- Edges represent conditional dependencies
- Each variable is connected to at most 2 neighbors

# Factor Graph Representation

```
[Prior₁] --- Y₁ --- [Coupling₁₂] --- Y₂ --- [Coupling₂₃] --- Y₃ --- [Priorₙ]
```

Factors:
- **Prior factors**: Provide boundary conditions at endpoints
- **Coupling factors**: Encode transition probabilities between adjacent variables

# Inference Task

Compute marginal distributions p(Yᵢ) for each variable given:
- Prior distributions on endpoints
- Conditional distributions p(Yᵢ₊₁|Yᵢ) between neighbors

# Belief Propagation Algorithm

## On Chains (Tree-Structured Graphs):

1. **Forward Pass**: Send messages from Y₁ to Yₙ
   - Prior₁ → Y₁
   - Y₁ → Coupling₁₂ → Y₂
   - Y₂ → Coupling₂₃ → Y₃
   - ... continue to Yₙ

2. **Backward Pass**: Send messages from Yₙ to Y₁
   - Priorₙ → Yₙ
   - Yₙ → Coupling_{n-1,n} → Yₙ₋₁
   - ... continue to Y₁

3. **Compute Marginals**: Each node's marginal is the product of incoming messages

# Examples Provided

1. `example_3_nodes()`: Step-by-step demonstration with 3 variables
   - Shows state of marginals after each message update
   - Educational: see how beliefs propagate through the chain

2. `example_n_nodes(n)`: Automated inference on n-node chain
   - Shows before/after marginals
   - Demonstrates scalability

# Author
Ralf Herbrich, Hasso-Plattner Institute, 2025
"""

module ChainFactorGraphExamples
# Load required modules
include("discrete.jl")
include("distributionbag.jl")
include("factors.jl")

using .DistributionCollections 
using .DiscreteDistribution
using .Factors

"""
    example_3_nodes() -> Nothing

Step-by-step demonstration of belief propagation on a 3-node chain.

# Graph Structure
```
[Prior: state 2] --- Y₁ --- Y₂ --- Y₃ --- [Prior: state 3]
```

# Setup
- All variables have 3 possible states
- Y₁ has strong prior favoring state 2
- Y₃ has strong prior favoring state 3
- Coupling matrices favor staying in the same state

# Message Passing Sequence

Demonstrates each message update individually:
1. Prior₁ → Y₁: Y₁ believes it's in state 2
2. Prior₃ → Y₃: Y₃ believes it's in state 3
3. Y₁ → Y₂: Y₂ receives signal from Y₁
4. Y₃ → Y₂: Y₂ receives signal from Y₃
5. Y₂ → Y₃: Y₃ updated based on Y₂'s belief
6. Y₂ → Y₁: Y₁ updated based on Y₂'s belief

# Educational Value
- See how beliefs propagate through the chain
- Observe the "meeting in the middle" effect
- Understand influence of strong priors on neighbors
- Watch marginals evolve step by step

# Output
Prints marginal distributions after each message update, showing
how evidence from the boundaries propagates inward.
"""
function example_3_nodes()    
    # Create distribution bag with 3-state discrete distributions
    db = DistributionBag(Discrete(3))
    
    # Add marginal distributions for three variables
    y1 = add!(db)
    y2 = add!(db)
    y3 = add!(db)

    # Helper function to display current marginals
    function print_marginals()
        println("   Y1: ", db[y1])
        println("   Y2: ", db[y2])
        println("   Y3: ", db[y3])
    end

    # Create factors
    # Prior strongly favoring state 2 for Y1
    f1 = PriorDiscreteFactor(db, y1, Discrete([0.0, 100.0, 0.0]))
    
    # Coupling Y1-Y2: prefer same state, some transition probability
    f2 = CouplingDiscreteFactor(db, y1, y2, Matrix([0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5]'))
    
    # Coupling Y2-Y3: same transition structure
    f3 = CouplingDiscreteFactor(db, y2, y3, Matrix([0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5]'))
    
    # Prior strongly favoring state 3 for Y3
    f4 = PriorDiscreteFactor(db, y3, Discrete([0.0, 0.0, 100.0]))

    # Step-by-step message passing with commentary
    println("Before all message updates")
    println("(All variables uniform)")
    print_marginals()
    
    println("\nAfter f1->y1 message update")
    println("(Y1 now strongly believes state 2 due to prior)")
    update_msg_to_x!(f1)
    print_marginals()

    println("\nAfter f4->y3 message update")
    println("(Y3 now strongly believes state 3 due to prior)")
    update_msg_to_x!(f4)
    print_marginals()

    println("\nAfter f2->y2 message update")
    println("(Y2 receives information from Y1)")
    update_msg_to_y!(f2)
    print_marginals()

    println("\nAfter f3->y3 message update")  
    println("(Y3 receives information from Y2, conflicts with prior)")
    update_msg_to_y!(f3)
    print_marginals()

    println("\nAfter f3->y2 message update")
    println("(Y2 receives information from Y3, now sees evidence from both sides)")
    update_msg_to_x!(f3)
    print_marginals()

    println("\nAfter f2->y1 message update")
    println("(Y1 receives information from Y2, conflicts with prior)")
    update_msg_to_x!(f2)
    print_marginals()
    
    println("\nFinal marginals show compromise between boundary priors")
end

"""
    example_n_nodes(n=3) -> Nothing

Automated belief propagation on an n-node chain.

# Graph Structure
```
[Prior: state 1] --- Y₁ --- Y₂ --- ... --- Yₙ --- [Prior: state n]
```

# Arguments
- `n::Int=3`: Number of variables in the chain

# Setup
- All variables have 3 possible states
- Y₁ has strong prior favoring state 1
- Yₙ has strong prior favoring state n (clamped to 3)
- Coupling matrices favor staying in the same state

# Algorithm
1. Initialize all marginals to uniform
2. Send messages from both prior factors
3. **Forward pass**: Update messages Y₁→Y₂→...→Yₙ
4. **Backward pass**: Update messages Yₙ→...→Y₂→Y₁

# Complexity
- Time: O(n × T²) where T is number of states
- Space: O(n × T)
- Exact inference (no approximation)

# Examples
```julia
example_n_nodes(3)   # Small chain
example_n_nodes(9)   # Longer chain
example_n_nodes(50)  # Demonstrate scalability
```

# Expected Behavior
- Variables near Y₁ favor state 1
- Variables near Yₙ favor state 3  
- Middle variables show more uncertainty
- Longer chains → more gradual transitions

# Output
Prints marginals before and after inference, showing how
boundary conditions propagate through the chain.
"""
function example_n_nodes(n = 3)    
    # Create distribution bag
    db = DistributionBag(Discrete(3))
    
    # Add n variables
    y = [add!(db) for i in 1:n]

    # Helper function to display all marginals
    function print_marginals()
        for i in 1:n
            println("   Y", i, ": ", db[y[i]])
        end
    end

    # Create prior factors at boundaries
    # Y₁ strongly favors state 1
    f1 = PriorDiscreteFactor(db, y[begin], Discrete([100.0, 0.0, 0.0]))
    # Yₙ strongly favors state 3
    f2 = PriorDiscreteFactor(db, y[end], Discrete([0.0, 0.0, 100.0]))
    
    # Create coupling factors between adjacent variables
    # All use same transition matrix: prefer same state
    f = [CouplingDiscreteFactor(db, y[i], y[i+1], 
            Matrix([0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5]')) 
         for i in 1:n-1]

    println("Before all message updates")
    println("(All variables start uniform)")
    print_marginals()

    # Initialize with prior messages
    update_msg_to_x!(f1)  # Y₁ receives its prior
    update_msg_to_x!(f2)  # Yₙ receives its prior

    # Forward pass: propagate information from Y₁ to Yₙ
    for i in 1:n-1
        update_msg_to_y!(f[i])
    end

    # Backward pass: propagate information from Yₙ to Y₁
    for i in n-1:-1:1
        update_msg_to_x!(f[i])
    end
    
    println("\nAfter all message updates")
    println("(Marginals computed via belief propagation)")
    print_marginals()
    
    println("\nNote: Gradient from state 1 (left) to state 3 (right)")
end


# ============================================================================
# Run Examples
# ============================================================================
function main()
    println("\n" * "="^70)
    println("Step-by-step example of message passing in a chain of 3 nodes")
    println("="^70)
    example_3_nodes()

    println("\n" * "="^70)
    println("Example of a chain with 9 nodes")
    println("="^70)
    example_n_nodes(9)
end

end