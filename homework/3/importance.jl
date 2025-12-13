using Distributions
using StatsBase, Plots

# Constants
μ_Z = 20
σ²_Z = 10
μ_X = 6
σ²_X = 8


g(x) = x * exp(0.1 * x)
f(x, z) = Dirac(z - g(x))

msg_Z_f(z) = pdf(Normal(μ_Z, sqrt(σ²_Z)), z)
msg_X_f(x) = pdf(Normal(μ_X, sqrt(σ²_X)), x)

# target
p(x, z) = msg_Z_f(z) .* f(x, z)

# proposals
q_1(x, z) = msg_Z_f(z) * msg_X_f(x)
q_2(x, z) = msg_X_f(x) * f(x, z)

# sample
N = 1000
x1s = rand(Normal(μ_X, sqrt(σ²_X)), N)
z1s = rand(Normal(μ_Z, sqrt(σ²_Z)), N)

w1hat = [p(x, z)/q_1(x, z) for (x, z) in zip(x1s, z1s)]

# w1 = w1hat ./ sum(w1hat)
#println(w1)  nonsensical as q1 is not a proper distribution

x2s = rand(Normal(μ_X, sqrt(σ²_X)), N)
z2s = g.(x2s)

w2hat = [msg_Z_f(z) / msg_X_f(x) for (x, z) in zip(x2s, z2s)]

w2 = w2hat ./ sum(w2hat)

msg_X_f_estimate = w2
msg_X_f_true(x) = msg_Z_f(g(x))

# bins for x
bins = range(minimum(x2s), maximum(x2s), length=60)

# weighted histogram
hist = fit(Histogram, x2s, Weights(w2), bins)

# bin centers
x_centers = (hist.edges[1][1:end-1] .+ hist.edges[1][2:end]) ./ 2
msg_X_f_est = hist.weights ./ sum(hist.weights)

msg_X_f_true_vals = msg_X_f_true.(x_centers)
msg_X_f_true_vals ./= sum(msg_X_f_true_vals)

# plot proposal vs target
p1 = plot(
    x_centers,
    msg_X_f_true_vals,
    label="True msg_f→X",
    linewidth=3
)

plot!(
    p1,
    x_centers,
    msg_X_f_est,
    seriestype=:step,
    label="IS estimate (q₂)",
    linewidth=2
)

savefig(p1, "msg_fX_true_vs_est_q2.png")

# plot error
error = msg_X_f_est .- msg_X_f_true_vals

p2 = plot(
    x_centers,
    error,
    label="Estimate − True",
    xlabel="x",
    ylabel="Error",
    linewidth=2
)

savefig(p2, "msg_fX_error_q2.png")

