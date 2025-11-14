using Distributions
using FStrings


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

marginal_s1 = marginal_s1 ./ sum(marginal_s1)
marginal_s2 = marginal_s2 ./ sum(marginal_s2)
marginal_p1 = marginal_p1 ./ sum(marginal_p1)
marginal_p2 = marginal_p2 ./ sum(marginal_p2)
marginal_d = marginal_d ./ sum(marginal_d)


println(f"marginal of s1: {marginal_s1}, {sum(marginal_s1)}")
