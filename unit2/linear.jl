"""
    LinearAlgebraPlots

A comprehensive module for visualizing linear algebra concepts and operations using
various basis functions and transformations, with special emphasis on MNIST digit
manipulation and visualization.

# Features
- Multiple basis function implementations (polynomial, Fourier, Gaussian, sigmoid)
- Linear mapping visualizations via SVD decomposition
- MNIST digit manipulation (linear combinations, SVD decomposition, rotations)
- Interactive plotting utilities for educational purposes

# Basis Functions
The module supports four types of basis functions:
- Polynomial: Powers of x
- Fourier: Cosine basis with periodic structure
- Gaussian: Normal distribution-based basis
- Sigmoid: Logistic function-based basis

# MNIST Operations
- Linear combinations of digit images
- Singular Value Decomposition (SVD) for dimensionality reduction
- Image rotation via linear transformations
- Animated rotation sequences

# Author
Ralf Herbrich, Hasso-Plattner Institute, 2025
"""
module LinearAlgebraPlots

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using MLDatasets
using Plots

# ============================================================================
# Basis Functions
# ============================================================================

"""
    polynomial_basis(x::Float64, j) -> Float64

Computes the j-th basis function value at x for polynomial basis functions.

The polynomial basis is defined as φⱼ(x) = xʲ, where j is the basis function index.

# Arguments
- `x::Float64`: The input value at which to evaluate the basis function
- `j`: The index of the basis function (non-negative integer)

# Returns
- `Float64`: The value xʲ

# Examples
```julia
polynomial_basis(2.0, 3)  # Returns 8.0 (2³)
polynomial_basis(1.5, 0)  # Returns 1.0 (1.5⁰)
```

# Notes
- For j=0, this returns 1 (constant basis)
- Higher values of j produce increasingly non-linear functions
"""
function polynomial_basis(x::Float64, j)
    return x^j
end

"""
    fourier_basis(x::Float64, j) -> Float64

Computes the j-th basis function value at x for the Fourier basis functions.

The Fourier basis is defined as φⱼ(x) = cos(πjx), providing a periodic
orthogonal basis suitable for representing periodic signals.

# Arguments
- `x::Float64`: The input value at which to evaluate the basis function
- `j`: The frequency index (non-negative integer)

# Returns
- `Float64`: The value cos(πjx)

# Examples
```julia
fourier_basis(0.5, 1)  # Returns cos(π/2) ≈ 0
fourier_basis(1.0, 2)  # Returns cos(2π) = 1
```

# Notes
- For j=0, this returns 1 (constant basis)
- Higher j values represent higher frequency components
- The basis functions are periodic with period 2/j
"""
function fourier_basis(x::Float64, j)
    return cos(π * j * x)
end

"""
    gauss_basis(x::Float64, j) -> Float64

Computes the j-th basis function value at x for the Gaussian basis functions.

The Gaussian basis uses normal distributions centered at integer positions:
φⱼ(x) = pdf(N(j, 1), x), where N(μ, σ²) is a normal distribution.

# Arguments
- `x::Float64`: The input value at which to evaluate the basis function
- `j`: The center position of the Gaussian (typically an integer)

# Returns
- `Float64`: The probability density function value at x for N(j, 1)

# Examples
```julia
gauss_basis(3.0, 3)  # Maximum value at center
gauss_basis(4.0, 3)  # Lower value, one standard deviation away
```

# Notes
- All basis functions have unit variance (σ = 1)
- Centers are positioned at integer values j
- Provides smooth, localized basis functions
- Useful for radial basis function (RBF) networks
"""
function gauss_basis(x::Float64, j)
    return pdf(Normal(j, 1), x)
end

"""
    sigmoid_basis(x::Float64, j) -> Float64

Computes the j-th basis function value at x for the sigmoid basis functions.

The sigmoid basis is defined as φⱼ(x) = exp(x-j)/(1+exp(x-j)), creating
S-shaped curves centered at position j.

# Arguments
- `x::Float64`: The input value at which to evaluate the basis function
- `j`: The center position of the sigmoid transition

# Returns
- `Float64`: The sigmoid function value, in range (0, 1)

# Examples
```julia
sigmoid_basis(0.0, 0)  # Returns 0.5 (center of sigmoid)
sigmoid_basis(5.0, 0)  # Returns ≈1.0 (saturated upper region)
sigmoid_basis(-5.0, 0) # Returns ≈0.0 (saturated lower region)
```

# Notes
- Output is always in the range (0, 1)
- Transition occurs around x = j
- Useful for neural network-like representations
"""
function sigmoid_basis(x::Float64, j)
    return exp(x - j) / (1 + exp(x - j))
end

# ============================================================================
# Linear Basis Function Evaluation
# ============================================================================

"""
    f(x::Float64, w, basis) -> Float64

Evaluates a linear combination of basis functions at a given point.

Computes f(x) = Σᵢ wᵢ φᵢ₋₁(x), where φⱼ is the j-th basis function.

# Arguments
- `x::Float64`: The input value at which to evaluate the function
- `w`: Weight vector (array-like) of length d
- `basis`: Basis function of the form (x::Float64, j) -> Float64

# Returns
- `Float64`: The weighted sum of basis function evaluations

# Examples
```julia
w = [1.0, 2.0, 3.0]
f(1.0, w, polynomial_basis)  # Returns 1 + 2*1 + 3*1² = 6.0
```

# Notes
- Basis functions are indexed from 0, so φⱼ uses j-1 internally
- This allows for flexible function approximation with different bases
- Used for constructing function samples in plotting
"""
function f(x::Float64, w, basis)
    y = 0
    for j in eachindex(w)
        y += w[j] * basis(x, j - 1)
    end
    return y
end

# ============================================================================
# Plotting Functions
# ============================================================================

"""
    plot_function_sample(basis, n=99, d=5; min_x=0, max_x=5, color1=:blue, color2=:red) -> Plot

Generates a visualization of random function samples using linear basis functions.

Creates a plot showing multiple random functions, each generated by sampling
weight vectors from a standard normal distribution and combining them with
the specified basis functions.

# Arguments
- `basis`: Basis function to use (e.g., polynomial_basis, fourier_basis)
- `n::Int=99`: Number of thin sample functions to plot
- `d::Int=5`: Dimensionality (number of basis functions to use)

# Keywords
- `min_x::Real=0`: Minimum x-value for the plot domain
- `max_x::Real=5`: Maximum x-value for the plot domain
- `color1::Symbol=:blue`: Color for the first (thick) sample line
- `color2::Symbol=:red`: Color for the remaining (thin) sample lines

# Returns
- `Plot`: A Plots.jl plot object showing the function samples

# Examples
```julia
# Plot polynomial basis functions
p = plot_function_sample(polynomial_basis, 50, 4, min_x=-2, max_x=2)

# Plot Fourier basis functions
p = plot_function_sample(fourier_basis, 100, 10, min_x=-π, max_x=π)
```

# Notes
- The first function is plotted with a thicker line for emphasis
- Weight vectors are sampled from N(0, 1)
- Useful for visualizing the function space induced by different bases
- X-axis is sampled at 1000 points for smooth curves
"""
function plot_function_sample(
    basis,
    n = 99,
    d = 5;
    min_x = 0,
    max_x = 5,
    color1 = :blue,
    color2 = :red,
)
    # Generate fine-grained x-axis for smooth plotting
    xs = range(min_x, max_x, 1000)
    
    # Sample first weight vector and plot with emphasis
    v = randn(d, 1)
    p = plot(
        xs,
        map(x -> f(x, v, basis), xs),
        legend = false,
        linewidth = 2,
        color = color1,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    
    # Add n additional random function samples
    for i = 1:n
        v = randn(d, 1)
        plot!(
            xs,
            map(x -> f(x, v, basis), xs),
            legend = false,
            linewidth = 0.3,
            color = color2,
        )
    end
    
    # Add axis labels
    xlabel!(L"x")
    ylabel!(L"f(x)")

    return p
end

"""
    plot_mapping(A) -> Tuple{Plot, Plot, Plot, Plot, Plot}

Visualizes the step-by-step decomposition of a 2D linear transformation using SVD.

Shows how a 2×2 matrix A transforms a unit circle through its SVD decomposition:
A = U Σ Vᵀ. The five plots show the progressive transformation stages.

# Arguments
- `A`: A 2×2 matrix representing the linear transformation

# Returns
- `Tuple{Plot, Plot, Plot, Plot, Plot}`: Five plots showing:
  1. Original unit circle with basis vectors
  2. After rotation by Vᵀ
  3. After scaling by Σ (singular values)
  4. After rotation by U
  5. Final transformation by A (equivalent to step 4)

# Examples
```julia
# Visualize a shear transformation
(p1, p2, p3, p4, p5) = plot_mapping([1 1; 0 1])

# Visualize a rotation and scaling
(p1, p2, p3, p4, p5) = plot_mapping([2*cos(π/4) -sin(π/4); 2*sin(π/4) cos(π/4)])
```

# Notes
- Blue arrow represents the first basis vector
- Red arrow represents the second basis vector (at index 26 of the circle)
- All plots maintain equal aspect ratio for geometric accuracy
- Useful for understanding SVD geometrically
"""
function plot_mapping(A)
    # Helper function to plot points with basis vectors
    function plot_points(pts, i1, i2)
        p = plot(
            pts[1, :],
            pts[2, :],
            legend = false,
            linewidth = 3,
            color = :black,
            xtickfontsize = 14,
            ytickfontsize = 14,
            xguidefontsize = 16,
            yguidefontsize = 16,
            aspect_ratio = :equal,
        )
        # Add blue arrow for first basis vector
        plot!([0, pts[1, i1]], [0, pts[2, i1]], arrow = true, linewidth = 5, color = :blue)
        # Add red arrow for second basis vector
        plot!([0, pts[1, i2]], [0, pts[2, i2]], arrow = true, linewidth = 5, color = :red)
        xlabel!(L"x_1")
        ylabel!(L"x_2")
        return p
    end

    # Compute SVD decomposition: A = U Σ Vᵀ
    U, S, V = svd(A, alg = LinearAlgebra.QRIteration())

    # Choose indices for basis vectors (roughly orthogonal on circle)
    i1, i2 = 1, 26
    
    # Generate unit circle points
    θs = range(0, stop = 2π, length = 100)
    X = hcat(sin.(θs), cos.(θs))'
    
    # Create plots showing each transformation step
    p1 = plot_points(X, i1, i2)                         # Original
    p2 = plot_points(V' * X, i1, i2)                    # Rotate by Vᵀ
    p3 = plot_points(Diagonal(S) * V' * X, i1, i2)      # Scale by Σ
    p4 = plot_points(U * Diagonal(S) * V' * X, i1, i2)  # Rotate by U
    p5 = plot_points(A * X, i1, i2)                     # Direct transformation

    return (p1, p2, p3, p4, p5)
end

# ============================================================================
# MNIST Visualization Functions
# ============================================================================

"""
    plot_MNIST_digit(digit; inset=:nothing, subplot=:nothing) -> Plot

Renders a single MNIST digit as a grayscale heatmap.

# Arguments
- `digit`: A 784-element vector (28×28 flattened) representing the digit image

# Keywords
- `inset`: Position specification for inset plot (default: :nothing for standalone plot)
- `subplot`: Subplot index for multi-plot layouts (default: :nothing)

# Returns
- `Plot`: A Plots.jl heatmap showing the digit

# Examples
```julia
data = MNIST(split=:train)
digit = data.features[:, :, 1][:]
plot_MNIST_digit(digit)
```

# Notes
- Image is automatically flipped to correct orientation
- Uses inverted grayscale (1 - intensity) for proper display
- Can be used as standalone plot or as inset in larger visualization
- Border color is white for clean appearance
"""
function plot_MNIST_digit(digit; inset=:nothing, subplot=:nothing)
    # Reshape and flip the image to correct orientation
    img = hcat(map(r -> r[28:-1:1], eachrow(reshape(digit[(28*28):-1:1], 28, 28)'))...)'

    if inset == :nothing
        # Create standalone plot
        return heatmap(1 .- img,
            colormap=:grays,
            legend=false,
            aspect_ratio=:equal,
            xaxis=nothing,
            yaxis=nothing,
            bordercolor=:white,
            xlim=(0, 28),
            ylim=(0, 28),
            clim=(0, 1),
        )
    else
        # Create inset plot within existing figure
        return heatmap!(1 .- img,
            colormap=:grays,
            legend=false,
            aspect_ratio=:equal,
            xaxis=nothing,
            yaxis=nothing,
            bordercolor=:white,
            inset=inset,
            subplot=subplot,
            xlim=(0, 28),
            ylim=(0, 28),
            clim=(0, 1),
        )
    end
end

"""
    plot_MNIST_grid(; idx1=1, idx2=4, N=10) -> Nothing

Creates a grid visualization showing linear combinations of two MNIST digits.

Displays N×N grid where each cell shows α₁·digit₁ + α₂·digit₂ for different
combinations of coefficients α₁ and α₂ ranging from 0 to 1.

# Keywords
- `idx1::Int=1`: Index of the first MNIST digit from training set
- `idx2::Int=4`: Index of the second MNIST digit from training set
- `N::Int=10`: Number of samples along each axis (creates N×N grid)

# Examples
```julia
# Create 10×10 grid of linear combinations
plot_MNIST_grid(idx1=1, idx2=5, N=10)

# Create finer 15×15 grid
plot_MNIST_grid(idx1=2, idx2=7, N=15)
```

# Notes
- X-axis represents coefficient α₁ for first digit
- Y-axis represents coefficient α₂ for second digit
- Each inset shows the weighted combination at that point
- Useful for visualizing the linear span of two digit images
- Automatically displays the resulting plot
"""
function plot_MNIST_grid(; idx1=1, idx2=4, N=10)
    # Load the two MNIST digits
    x1 = MNIST(split=:train).features[:, :, idx1]
    x2 = MNIST(split=:train).features[:, :, idx2]
    
    # Create base plot with labeled axes
    p = plot(
        xlim=(-0.1, 1.1),
        ylim=(-0.1, 1.1),
        legend=false,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
    )
    xlabel!(L"\alpha_1")
    ylabel!(L"\alpha_2")
    
    # Create grid of linear combinations
    idx = 2  # Subplot index (1 is the main plot)
    for α1 in range(0, 1.0 - 8.0 / 5.0N, N)
        for α2 in range(0, 1.0 - 8.0 / 5.0N, N)
            # Plot weighted combination as inset at (α₁, α₂)
            plot_MNIST_digit(α1 * x1 + α2 * x2,
                inset=(1, bbox(α1 + 1.0 / 2.0N, α2 + 1.0 / 2.0N, 5.0 / 8.0N, 5.0 / 8.0N, :bottom, :left)),
                subplot=idx,
            )
            idx += 1
        end
    end
    display(p)
end

"""
    plot_MNIST_SVD(; N=1000, k=5, idx=2, base_filename="~/Downloads/mnist_svd") -> Nothing

Performs and visualizes SVD decomposition of MNIST digit dataset.

Computes the Singular Value Decomposition of N MNIST digits and shows the
top k principal components (basis images) as well as reconstructions.

# Keywords
- `N::Int=1000`: Number of MNIST images to include in the analysis
- `k::Int=5`: Number of singular values/vectors to retain
- `idx::Int=2`: Index of example digit to reconstruct
- `base_filename::String`: Base path for saving output files

# Output Files
- `{base_filename}_base_{i}.svg`: The i-th basis image (i=1..k)
- `{base_filename}_original_{idx}.svg`: Original digit at index idx
- `{base_filename}_reconstruction_{idx}.svg`: k-component reconstruction

# Examples
```julia
# Analyze first 500 digits with 10 components
plot_MNIST_SVD(N=500, k=10, idx=5)

# High-resolution analysis
plot_MNIST_SVD(N=5000, k=20, idx=100, base_filename="/tmp/svd_analysis")
```

# Notes
- Higher N provides better basis estimation but slower computation
- Higher k retains more information in reconstructions
- Basis images reveal principal variations in digit appearance
- Useful for understanding dimensionality reduction via SVD
"""
function plot_MNIST_SVD(; N=1000, k=5, idx=2, base_filename="~/Downloads/mnist_svd")
    # Load MNIST training data
    data = MNIST(split=:train)
    
    # Stack N digit images as rows in matrix X
    X = hcat(map(i -> data.features[:, :, i][:], 1:N)...)'
    
    # Compute SVD: X = U Σ Vᵀ
    U, Σ, V = svd(X)

    # Create plot canvas
    p = plot(
        xlim=(-0.1, 1.1),
        ylim=(-0.1, 1.1),
        legend=false,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
    )

    # Compute and save the top k basis images (principal components)
    A = Diagonal(Σ[1:k]) * V[:, 1:k]'
    println(size(A))
    for i in 1:k
        display(plot_MNIST_digit(A[i, :]))
        savefig(base_filename * "_base_$i.svg")
    end

    # Save original and k-component reconstruction of example digit
    display(plot_MNIST_digit(X[idx, :]))
    savefig(base_filename * "_original_$idx.svg")
    display(plot_MNIST_digit(U[idx, 1:k]' * A))
    savefig(base_filename * "_reconstruction_$idx.svg")
end

"""
    compute_MNIST_rotation_matrix(; α=π/2) -> Matrix

Constructs a 784×784 matrix that rotates 28×28 MNIST digit images.

Uses bilinear interpolation to create a smooth rotation transformation
that can be applied via matrix multiplication to flattened MNIST digits.

# Keywords
- `α::Real=π/2`: Rotation angle in radians (positive = counterclockwise)

# Returns
- `Matrix{Float64}`: A 784×784 transformation matrix

# Examples
```julia
# Create 45-degree rotation matrix
R = compute_MNIST_rotation_matrix(α=π/4)

# Rotate a digit
digit = MNIST(split=:train).features[:, :, 1][:]
rotated = R * digit
```

# Algorithm
For each target pixel:
1. Compute source position via inverse rotation around image center
2. Use bilinear interpolation from four neighboring source pixels
3. Handle edge cases where source is outside image bounds

# Notes
- Rotation is around the center point (14.5, 14.5)
- Uses bilinear interpolation for smooth results
- Pixels outside the original image contribute zero
- Matrix is sparse but stored as dense for efficiency
"""
function compute_MNIST_rotation_matrix(; α=π / 2)
    N = 28
    A = zeros(N * N, N * N)

    # Helper function to set matrix element with bounds checking
    function set(idx, row, col, value)
        if row > 0 && row <= N && col > 0 && col <= N
            A[idx, round(Int, (row - 1) * 28 + col)] = value
        end
    end

    # Compute bilinear interpolation weights for each target pixel
    for row in 1:N
        for col = 1:N
            idx = (row - 1) * N + col
            
            # Convert to centered coordinates
            x, y = col - 14.5, row - 14.5
            
            # Compute source position via inverse rotation
            γ = atan(y, x)  # Angle from center
            r = sqrt(x^2 + y^2)  # Distance from center
            src_col, src_row = 14.5 + cos(γ + α) * r, 14.5 + sin(γ + α) * r

            # Set bilinear interpolation weights from four neighboring pixels
            set(idx, floor(src_row), floor(src_col), (ceil(src_row) - src_row) * (ceil(src_col) - src_col))
            set(idx, floor(src_row), ceil(src_col), (ceil(src_row) - src_row) * (src_col + 1.0 - ceil(src_col)))
            set(idx, ceil(src_row), floor(src_col), (src_row + 1.0 - ceil(src_row)) * (ceil(src_col) - src_col))
            set(idx, ceil(src_row), ceil(src_col), (src_row + 1.0 - ceil(src_row)) * (src_col + 1.0 - ceil(src_col)))
        end
    end

    return A
end

"""
    plot_MNIST_rotation_matrix(; idx=5, base_filename="~/Downloads/") -> Nothing

Generates comprehensive visualizations of MNIST digit rotation transformations.

Creates static images of rotated digits, visualizations of the rotation matrices,
and an animated movie showing continuous rotation.

# Keywords
- `idx::Int=5`: Index of the MNIST digit to use for demonstrations
- `base_filename::String`: Directory path for saving output files

# Output Files
- `original_{idx}.svg`: The original unrotated digit
- `rotated_pi_16_{idx}.svg`: Digit rotated by +22.5° (π/16 radians)
- `rotated_minus_pi_16_{idx}.svg`: Digit rotated by -22.5°
- `rotation_matrix_plus_pi_16.svg`: Heatmap of +22.5° rotation matrix
- `rotation_matrix_minus_pi_16.svg`: Heatmap of -22.5° rotation matrix
- `rotation_movie.mp4`: Animated 360° rotation (30 fps, 4 seconds)

# Examples
```julia
# Generate all rotation visualizations for digit 7
plot_MNIST_rotation_matrix(idx=7, base_filename="~/Desktop/rotation/")
```

# Notes
- Movie shows full 360° rotation in 120 frames at 30 fps
- Rotation matrices are visualized as 784×784 heatmaps
- Demonstrates linear transformation properties geometrically
- Useful for teaching linear algebra concepts with visual examples
"""
function plot_MNIST_rotation_matrix(; idx=5, base_filename="~/Downloads/")
    # Load the example digit
    img = MNIST(split=:train).features[:, :, idx][:]

    # Save original and ±22.5° rotated versions
    display(plot_MNIST_digit(img))
    savefig(base_filename * "original_$idx.svg")
    display(plot_MNIST_digit(compute_MNIST_rotation_matrix(α=π / 16) * img))
    savefig(base_filename * "rotated_pi_16_$idx.svg")
    display(plot_MNIST_digit(compute_MNIST_rotation_matrix(α=-π / 16) * img))
    savefig(base_filename * "rotated_minus_pi_16_$idx.svg")

    # Visualize and save +22.5° rotation matrix
    A = compute_MNIST_rotation_matrix(α=π / 16)
    p = heatmap(1 .- A,
        colormap=:grays,
        legend=false,
        aspect_ratio=:equal,
        xaxis=nothing,
        yaxis=nothing,
        bordercolor=:white,
        xlim=(0, 28 * 28),
        ylim=(0, 28 * 28),
        clim=(0, 1),
    )
    display(p)
    savefig(base_filename * "rotation_matrix_plus_pi_16.svg")

    # Visualize and save -22.5° rotation matrix
    A = compute_MNIST_rotation_matrix(α=-π / 16)
    p = heatmap(1 .- A,
        colormap=:grays,
        legend=false,
        aspect_ratio=:equal,
        xaxis=nothing,
        yaxis=nothing,
        bordercolor=:white,
        xlim=(0, 28 * 28),
        ylim=(0, 28 * 28),
        clim=(0, 1),
    )
    display(p)
    savefig(base_filename * "rotation_matrix_minus_pi_16.svg")

    # Create animated rotation movie (full 360° in 120 frames)
    anim = Animation()
    for α in range(0, 2π, length=120)
        p = plot_MNIST_digit(compute_MNIST_rotation_matrix(α=α) * img)
        frame(anim, p)
    end

    # Save as MP4 video at 30 fps
    mp4(anim, base_filename * "rotation_movie.mp4", fps=30)
end

# ============================================================================
# Main Demonstration Function
# ============================================================================

"""
    main() -> Nothing

Runs all visualization demonstrations and saves output files.

This function executes a comprehensive suite of visualizations demonstrating:
- Different basis function families
- MNIST digit linear combinations
- SVD decomposition of digit data
- Rotation transformations
- SVD geometric interpretation

All outputs are saved to ~/Downloads/ by default.

# Examples
```julia
LinearAlgebraPlots.main()
```

# Output Files
Generated in ~/Downloads/:
- poly.png: Polynomial basis function samples
- fourier.png: Fourier basis function samples  
- gauss.png: Gaussian basis function samples
- sigmoid.png: Sigmoid basis function samples
- mnist_linear.svg: Grid of digit linear combinations
- mnist_svd_*.svg: SVD basis images and reconstructions
- original_*.svg, rotated_*.svg: Rotation examples
- rotation_matrix_*.svg: Rotation matrix visualizations
- rotation_movie.mp4: Animated rotation sequence
- circle_*.svg: SVD decomposition steps

# Notes
- Random seed is set to 42 for reproducibility
- Requires write access to ~/Downloads/
- Total execution time varies with system performance
- Some operations (SVD with 1000 images) may take several seconds
"""
function main()
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Generate basis function visualizations
    p = plot_function_sample(polynomial_basis, 99, 5, min_x = -5, max_x = 5)
    savefig(p, "~/Downloads/poly.png")
    display(p)

    p = plot_function_sample(fourier_basis, 99, 5, min_x = -2, max_x = 2)
    savefig(p, "~/Downloads/fourier.png")
    display(p)

    p = plot_function_sample(gauss_basis, 99, 5, min_x = -5, max_x = 10)
    savefig(p, "~/Downloads/gauss.png")
    display(p)

    p = plot_function_sample(sigmoid_basis, 99, 5, min_x = -5, max_x = 10)
    savefig(p, "~/Downloads/sigmoid.png")
    display(p)

    # Generate MNIST visualizations
    plot_MNIST_grid()
    savefig("~/Downloads/mnist_linear.svg")

    plot_MNIST_SVD()

    plot_MNIST_rotation_matrix()    

    # Generate SVD mapping visualizations (using shear transformation)
    (p1, p2, p3, p4, p5) = plot_mapping([[1 1]; [0 1]])
    savefig(p1, "~/Downloads/circle_X.svg")
    display(p1)
    savefig(p2, "~/Downloads/circle_V'*X.svg")
    display(p2)
    savefig(p3, "~/Downloads/circle_S*V'*X.svg")
    display(p3)
    savefig(p4, "~/Downloads/circle_U*S*V'*X.svg")
    display(p4)
    savefig(p5, "~/Downloads/circle_A*X.svg")
    display(p5)
end

end