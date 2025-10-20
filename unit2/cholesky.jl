"""
    Cholesky Decomposition Module

This module provides implementations of Cholesky decomposition algorithms for
symmetric positive definite matrices.

# Overview

The Cholesky decomposition factorizes a symmetric positive definite matrix A into
the product of a lower triangular matrix L and its transpose:

    A = L Lᵀ

where L is a lower triangular matrix with positive diagonal entries.

# Algorithms Implemented

1. **Cholesky-Crout Algorithm**: Column-oriented computation
2. **Cholesky-Banachiewicz Algorithm**: Row-oriented computation

Both algorithms produce the same result but differ in their order of computation,
which may affect numerical stability and cache performance in practice.

# Mathematical Background

For a symmetric positive definite matrix A ∈ ℝⁿˣⁿ, the Cholesky decomposition
exists and is unique. The elements of L are computed as:

    L[i,j] = (A[i,j] - Σₖ₌₁ʲ⁻¹ L[i,k]L[j,k]) / L[j,j]  for i > j
    L[j,j] = √(A[j,j] - Σₖ₌₁ʲ⁻¹ L[j,k]²)

# Applications

- Solving systems of linear equations efficiently
- Computing matrix inverses
- Generating samples from multivariate Gaussian distributions
- Numerical optimization algorithms
- Probabilistic modeling and inference

# Complexity

Time complexity: O(n³/3) for an n×n matrix
Space complexity: O(n²) for storing the result

# Author
Ralf Herbrich, Hasso-Plattner Institute, 2025
"""

"""
    cholesky_crout(A::Matrix) -> Matrix{Float64}

Computes the Cholesky-Crout decomposition of a symmetric positive definite matrix A.

The Cholesky-Crout algorithm computes the lower triangular matrix L column by column,
where A = L Lᵀ. This is a column-oriented algorithm that may have better cache
performance for certain matrix layouts.

# Arguments
- `A::Matrix`: An n×n symmetric positive definite matrix

# Returns
- `Matrix{Float64}`: An n×n lower triangular matrix L such that A = L Lᵀ

# Algorithm

For each column j from 1 to n:
1. Compute the diagonal element:
   L[j,j] = √(A[j,j] - Σₖ₌₁ʲ⁻¹ L[j,k]²)

2. Compute the below-diagonal elements:
   L[i,j] = (A[i,j] - Σₖ₌₁ʲ⁻¹ L[i,k]L[j,k]) / L[j,j]  for i > j

# Examples
```jldoctest
julia> A = [1.0 2.0 0.0; 2.0 20.0 4.0; 0.0 4.0 17.0]
3×3 Matrix{Float64}:
 1.0   2.0   0.0
 2.0  20.0   4.0
 0.0   4.0  17.0

julia> L = cholesky_crout(A)
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 2.0  4.0  0.0
 0.0  1.0  4.0

julia> L * L'  # Verify: should equal A
3×3 Matrix{Float64}:
 1.0   2.0   0.0
 2.0  20.0   4.0
 0.0   4.0  17.0
```

# Errors
- Throws an error if A is not a square matrix
- May produce NaN or complex values if A is not positive definite
- Numerical instability may occur for ill-conditioned matrices

# Notes
- The matrix A should be symmetric and positive definite
- Only the lower triangle of A is accessed during computation
- The upper triangle of the result L contains zeros
- For numerical stability, the matrix should be well-conditioned

# References
- Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.)
- Cholesky, A.-L. (1924). Sur la résolution numérique des systèmes d'équations linéaires
"""
function cholesky_crout(A::Matrix)
    # Check that the matrix is square
    if (size(A)[1] != size(A)[2])
        error("matrix must be square")
    end
    n = size(A)[1]

    # Initialize result matrix with zeros
    L = zeros(n, n)

    # Column-oriented Cholesky decomposition (Crout algorithm)
    for j = 1:n
        # Compute diagonal element L[j,j]
        # L[j,j] = √(A[j,j] - Σₖ₌₁ʲ⁻¹ L[j,k]²)
        sum = 0
        for k = 1:(j-1)
            sum += L[j, k] * L[j, k]
        end
        L[j, j] = sqrt(A[j, j] - sum)

        # Compute below-diagonal elements in column j
        # L[i,j] = (A[i,j] - Σₖ₌₁ʲ⁻¹ L[i,k]L[j,k]) / L[j,j]
        for i = (j+1):n
            sum = 0
            for k = 1:(j-1)
                sum += L[i, k] * L[j, k]
            end
            L[i, j] = (A[i, j] - sum) / L[j, j]
        end
    end

    return (L)
end

"""
    cholesky_banachiewicz(A::Matrix) -> Matrix{Float64}

Computes the Cholesky-Banachiewicz decomposition of a symmetric positive definite matrix A.

The Cholesky-Banachiewicz algorithm computes the lower triangular matrix L row by row,
where A = L Lᵀ. This is a row-oriented algorithm that processes elements in a different
order compared to the Crout algorithm.

# Arguments
- `A::Matrix`: An n×n symmetric positive definite matrix

# Returns
- `Matrix{Float64}`: An n×n lower triangular matrix L such that A = L Lᵀ

# Algorithm

For each row i from 1 to n:
  For each column j from 1 to i:
    If i = j (diagonal element):
      L[i,i] = √(A[i,i] - Σₖ₌₁ⁱ⁻¹ L[i,k]²)
    
    Else (below-diagonal element):
      L[i,j] = (A[i,j] - Σₖ₌₁ʲ⁻¹ L[i,k]L[j,k]) / L[j,j]

# Examples
```jldoctest
julia> A = [4.0 12.0 -16.0; 12.0 37.0 -43.0; -16.0 -43.0 98.0]
3×3 Matrix{Float64}:
   4.0   12.0  -16.0
  12.0   37.0  -43.0
 -16.0  -43.0   98.0

julia> L = cholesky_banachiewicz(A)
3×3 Matrix{Float64}:
  2.0  0.0  0.0
  6.0  1.0  0.0
 -8.0  5.0  3.0

julia> L * L'  # Verify: should equal A
3×3 Matrix{Float64}:
   4.0   12.0  -16.0
  12.0   37.0  -43.0
 -16.0  -43.0   98.0
```

# Comparison with Cholesky-Crout

While both algorithms produce identical results (up to floating-point rounding),
they differ in:
- **Computational order**: Crout is column-oriented, Banachiewicz is row-oriented
- **Cache performance**: May differ depending on matrix storage layout
- **Implementation complexity**: Banachiewicz uses a single conditional expression

# Errors
- Throws an error if A is not a square matrix
- May produce NaN or complex values if A is not positive definite
- Numerical instability may occur for ill-conditioned matrices

# Notes
- The matrix A should be symmetric and positive definite
- Only the lower triangle of A is accessed during computation
- The upper triangle of the result L contains zeros
- The choice between Crout and Banachiewicz is mainly for pedagogical purposes

# References
- Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.)
- Banachiewicz, T. (1938). Méthode de résolution numérique des équations linéaires
"""
function cholesky_banachiewicz(A::Matrix)
    # Check that the matrix is square
    if (size(A)[1] != size(A)[2])
        error("matrix must be square")
    end
    n = size(A)[1]

    # Initialize result matrix with zeros
    L = zeros(n, n)

    # Row-oriented Cholesky decomposition (Banachiewicz algorithm)
    for i = 1:n
        for j = 1:i
            # Compute dot product of previous elements in row i and row j
            # This represents: Σₖ₌₁ʲ⁻¹ L[i,k]L[j,k]
            sum = 0
            for k = 1:(j-1)
                sum += L[i, k] * L[j, k]
            end

            # Compute L[i,j] based on whether it's a diagonal or off-diagonal element
            # Diagonal: L[i,i] = √(A[i,i] - Σₖ₌₁ⁱ⁻¹ L[i,k]²)
            # Off-diagonal: L[i,j] = (A[i,j] - Σₖ₌₁ʲ⁻¹ L[i,k]L[j,k]) / L[j,j]
            L[i, j] = (i == j) ? sqrt(A[j, j] - sum) : (A[i, j] - sum) / L[j, j]
        end
    end

    return (L)
end
