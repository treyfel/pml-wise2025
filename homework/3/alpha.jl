using Distributions
using Plots


"""
    α_divergence(p::Vector{Float64}, q::Vector{Float64}; α=0.5)

Computes the α-divergence between two discretized probability distributions.

The α-divergence is a family of divergences parameterized by α:
- α = 1: Forward KL divergence KL(p || q)
- α = 0: Reverse KL divergence KL(q || p)
- α = 0.5: Hellinger distance
- α → ±∞: Different limiting behaviors

D_α(p || q) = (1 - Σᵢ q(xᵢ) (p(xᵢ)/q(xᵢ))^α) / (α(1-α))

# Arguments
- `p::Vector{Float64}`: Discretized target distribution (should sum to 1)
- `q::Vector{Float64}`: Discretized approximating distribution (should sum to 1)
- `α::Float64`: Divergence parameter (default: 0.5)

# Returns
- `Float64`: The α-divergence value
"""
function α_divergence(p::Vector{Float64}, q::Vector{Float64}; α=0.5)
    if α == 1
        return KL(p, q)
    elseif α == 0
        return KL_reverse(p, q)
    else
        return (1 - sum(q .* (p ./ q).^α)) / (α * (1 - α))
    end
end


"""
    KL(p::Vector{Float64}, q::Vector{Float64})

Computes the Kullback-Leibler divergence KL(p || q) between two discretized probability distributions.

KL(p || q) = Σᵢ p(xᵢ) log(p(xᵢ) / q(xᵢ))

This is the forward KL divergence, which penalizes q for having mass where p has little mass.
It tends to produce mode-seeking behavior (the approximation focuses on a single mode).

# Arguments
- `p::Vector{Float64}`: Discretized target distribution (should sum to 1)
- `q::Vector{Float64}`: Discretized approximating distribution (should sum to 1)

# Returns
- `Float64`: The KL divergence value (always non-negative)
"""
function KL(p::Vector{Float64}, q::Vector{Float64})
    return sum(p .* log.(p ./ q))
end

"""
    KL_reverse(p::Vector{Float64}, q::Vector{Float64})

Computes the reverse Kullback-Leibler divergence KL(q || p) between two discretized distributions.

KL(q || p) = Σᵢ q(xᵢ) log(q(xᵢ) / p(xᵢ))

This is the reverse KL divergence, which penalizes q for missing mass where p has mass.
It tends to produce moment-matching behavior (the approximation covers all modes).

# Arguments
- `p::Vector{Float64}`: Discretized target distribution (should sum to 1)
- `q::Vector{Float64}`: Discretized approximating distribution (should sum to 1)

# Returns
- `Float64`: The reverse KL divergence value (always non-negative)
"""
function KL_reverse(p::Vector{Float64}, q::Vector{Float64})
    return sum(q .* log.(q ./ p))
end


p = [0.1 - 0.1 * (x / 20) for x = 1:19]
function q(λ)
    v = [pdf(Exponential(λ), x) for x in 1:19]
    v_norm = v ./ sum(v)
    v_norm .= max.(v_norm, eps())  # avoid nans
    return v_norm
end


# KL alphas
alphas = [(λ, KL(p, q(λ))) for λ in range(0.01, step = 0.01, stop=25)]

λ_min_idx = argmin(last.(alphas))
λ_min = first(alphas[λ_min_idx])
KL_min = last(alphas[λ_min_idx])
q_kl = q(λ_min)

plt = plot(first.(alphas), last.(alphas),
     xlabel="λ",
     ylabel="KL(p || q(λ))",
     title="KL divergence between p and Exponential(λ)",
     linewidth=2,
     label="KL divergence")

scatter!(plt, [λ_min], [KL_min], color=:red, label="Minimum")
savefig(plt, "kl_vs_lambda.png")
display(plt)


# Reverse KL alphas
reverse_alphas = [(λ, KL_reverse(p, q(λ))) for λ in range(0.01, step = 0.01, stop=25)]

reverse_λ_min_idx = argmin(last.(reverse_alphas))
reverse_λ_min = first(reverse_alphas[reverse_λ_min_idx])
reverse_KL_min = last(reverse_alphas[reverse_λ_min_idx])
q_rkl = q(reverse_λ_min)


plt2 = plot(first.(reverse_alphas), last.(reverse_alphas),
     xlabel="λ",
     ylabel="KL_reverse(p || q(λ))",
     title="Reverse KL divergence between p and Exponential(λ)",
     linewidth=2,
     label="Reverse KL divergence")

scatter!(plt2, [reverse_λ_min], [reverse_KL_min], color=:red, label="Minimum")

savefig(plt2, "reverse_kl_vs_lambda.png")
display(plt2)


# compute mean and var
x = 1:19
function mean_var(x, q)
    μ = sum(x .* q)
    σ2 = sum((x .- μ).^2 .* q)
    return μ, σ2
end

mean_kl, var_kl = mean_var(x, q_kl)
mean_rkl, var_rkl = mean_var(x, q_rkl)

println("KL-minimizing distribution: λ = ", λ_min)
println("Mean = ", mean_kl, ", Variance = ", var_kl)

println("Reverse KL-minimizing distribution: λ = ", reverse_λ_min)
println("Mean = ", mean_rkl, ", Variance = ", var_rkl)