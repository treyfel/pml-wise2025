using Distributions
using FStrings
using Plots

include("../unit4/discrete.jl")
include("../unit4/distributionbag.jl")
include("../unit4/factors.jl")


N = 20
μ_1 = 8
μ_2 = 12
σ²_1 = 2
σ²_2 = 2
β² = 3

f1(s1) = pdf(Normal(μ_1, sqrt(σ²_1)), s1)

f2(s1, p1) = pdf(Normal(s1, sqrt(β²)), p1)

f3(s2) = pdf(Normal(μ_2, sqrt(σ²_2)), s2)

f4(s2, p2) = pdf(Normal(s2, sqrt(β²)), p2)

f5(p1, p2, d) = (d == p1-p2) ? 1 : 0

f6(d) = (d > 0) ? 1 : 0

function marginals()
    start = time_ns()
    # marginal for s1
    marginal_s1 = [0.0 for i = 1:N]
    for s2 = 1:N
        for p1 = 1:N
            for p2 = 1:N
                for d = -N:N
                    for s1 = 1:N
                        marginal_s1[s1] += f1(s1) * f2(s1, p1) * f3(s2) * f4(s2, p2) * f5(p1, p2, d) * f6(d)
                    end
                end
            end
        end
    end


    # marginal for s2
    marginal_s2 = [0.0 for i = 1:N]
    for s1 = 1:N
        for p1 = 1:N
            for p2 = 1:N
                for d = -N:N
                    for s2 = 1:N
                        marginal_s2[s2] += f1(s1) * f2(s1, p1) * f3(s2) * f4(s2, p2) * f5(p1, p2, d) * f6(d)
                    end
                end
            end
        end
    end

    # marginal for p1
    marginal_p1 = [0.0 for i = 1:N]
    for s1 = 1:N
        for s2 = 1:N
            for p2 = 1:N
                for d = -N:N
                    for p1 = 1:N
                        marginal_p1[p1] += f1(s1) * f2(s1, p1) * f3(s2) * f4(s2, p2) * f5(p1, p2, d) * f6(d)
                    end
                end
            end
        end
    end

    # marginal for p2
    marginal_p2 = [0.0 for i = 1:N]
    for s1 = 1:N
        for s2 = 1:N
            for p1 = 1:N
                for d = -N:N
                    for p2 = 1:N
                        marginal_p2[p2] += f1(s1) * f2(s1, p1) * f3(s2) * f4(s2, p2) * f5(p1, p2, d) * f6(d)
                    end
                end
            end
        end
    end

    # marginal for d
    marginal_d = [0.0 for i = -N:N]
    for s1 = 1:N
        for p1 = 1:N
            for p2 = 1:N
                for s2 = 1:N
                    for d = -N:N
                        marginal_d[d + N + 1] += f1(s1) * f2(s1, p1) * f3(s2) * f4(s2, p2) * f5(p1, p2, d) * f6(d)
                    end
                end
            end
        end
    end

    println(f"Time: {(time_ns() - start) / 1e9}")

    marginal_s1 = marginal_s1 ./ sum(marginal_s1)
    marginal_s2 = marginal_s2 ./ sum(marginal_s2)
    marginal_p1 = marginal_p1 ./ sum(marginal_p1)
    marginal_p2 = marginal_p2 ./ sum(marginal_p2)
    marginal_d = marginal_d ./ sum(marginal_d)

    return [marginal_s1, marginal_s2, marginal_p1, marginal_p2, marginal_d]
end


# f1      f3
# -       -
# s1      s2
# -       -
# f2      f4:
# -       -
# p1      p2
#   -    -
#     f5
#     -
#     d
#     -
#     f6

# Message passing
function message_passing()
    start = time_ns()
    ## forward pass
    m_f1_s1 = [f1(i) for i = 1:N]
    m_f3_s2 = [f3(i) for i = 1:N]

    m_f2_s1 = [1/N for i = 1:N]
    m_f4_s2 = [1/N for i = 1:N]

    p_s1 = m_f1_s1 .* m_f2_s1
    p_s2 = m_f3_s2 .* m_f4_s2

    m_f2_p1 = [sum(f2(s1, p1) * p_s1[s1] for s1 = 1:N) for p1 = 1:N]
    m_f4_p2 = [sum(f4(s2, p2) * p_s2[s2] for s2 = 1:N) for p2 = 1:N]

    p_p1 = m_f2_p1
    p_p2 = m_f4_p2

    m_p1_f5 = p_p1
    m_p2_f5 = p_p2

    m_f5_d = [sum(sum(f5(p1, p2, d) * m_p1_f5[p1] * m_p2_f5[p2] for p2 = 1:N) for p1 = 1:N) for d = -N + 1:N - 1]

    p_d = m_f5_d

    ## backward pass
    #m_f6_d = [f6(d) for d = -N + 1:N - 1]
    #
    #p_d = m_f5_d .* m_f6_d
    #
    #m_d_f5 = p_d ./ m_f5_d # .* [(m_f5_d[d + N + 1] > 0) ? 1 : 0 for d = -N:N]
    ## print(m_d_f5)
#
    #m_f5_p1 = [sum(sum(f5(p1, p2, d) * m_d_f5[d + N] * m_p2_f5[p2] for p2 = 1:N) for d = -N + 1:N - 1) for p1 = 1:N]
    #m_f5_p2 = [sum(sum(f5(p1, p2, d) * m_d_f5[d + N] * m_p1_f5[p1] for p1 = 1:N) for d = -N + 1:N - 1) for p2 = 1:N]
#
    #p_p1 = m_f2_p1 .* m_f5_p1
    #p_p2 = m_f4_p2 .* m_f5_p2
#
    #m_f2_s1 = [sum(f2(s1, p1) * (p_p1[p1] / m_f2_p1[p1]) for p1 = 1:N) for s1 = 1:N]
    #m_f4_s2 = [sum(f4(s2, p2) * (p_p2[p2] / m_f4_p2[p2]) for p2 = 1:N) for s2 = 1:N]
#
    #p_s1 = m_f1_s1 .* m_f2_s1
    #p_s2 = m_f3_s2 .* m_f4_s2

    # p_s1 /= sum(p_s1)
    # p_s2 /= sum(p_s2)
    # p_p1 /= sum(p_p1)
    # p_p2 /= sum(p_p2)
    # p_d /= sum(p_d)

    println(f"Time: {(time_ns() - start) / 1e9}")
    return [p_s1, p_s2, p_p1, p_p2, p_d]
end

function plot_marginals(marginals, prefix="marginal")
    labels = ["s1", "s2", "p1", "p2", "d"]
    for (i, marginal) in enumerate(marginals)
        p = plot(marginal, label=labels[i], title="Marginal of $(labels[i])", xlabel="Index", ylabel="Probability")
        
        # Save the plot as a PNG file with a prefix for the name
        filename = "$(prefix)_$(labels[i]).png"
        savefig(p, filename)
        println("Saved plot for $(labels[i]) as $(filename)")
    end
end


# marg = marginals()
messages = message_passing()


# println("Marginals distributions Brute Force")
# println(f"marginal of s1: {marg[1]}")
# println(f"marginal of s2: {marg[2]}")
# println(f"marginal of p1: {marg[3]}")
# println(f"marginal of p2: {marg[4]}")
# println(f"marginal of d: {marg[5][2:end - 1]}")

println(f"Marginals distributions Sum Product")
println(f"sum: {sum(messages[1])}")
println(f"sum: {sum(messages[2])}")
println(f"sum: {sum(messages[3])}")
println(f"sum: {sum(messages[4])}")
println(f"sum: {sum(messages[5])}")

# println(f"Marginals distributions differences")
# println(f"\ndiff: {marg[1] .- messages[1]}\n")
# println(f"\ndiff: {marg[2] .- messages[2]}\n")
# println(f"\ndiff: {marg[3] .- messages[3]}\n")
# println(f"\ndiff: {marg[4] .- messages[4]}\n")
# println(f"\ndiff: {marg[5][2:end - 1] .- messages[5]}\n")
# 
# plot_marginals(marg)

plot_marginals(messages, "messages")