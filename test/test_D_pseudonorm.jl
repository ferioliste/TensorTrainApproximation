function pseudoinverse(A, tol=eps(Float64))
    U, S, V = svd(A)

    local_tol = tol * maximum(S) 
    significant_idxs = S .> local_tol
    S_inv = zeros(size(S))
    S_inv[significant_idxs] = 1.0 ./ S[significant_idxs]

    return (V .* S_inv') * U'
end

function ceillog2(x::Int)
    return 2^ceil(Int, log2(x))
end

function expected_fnorm2(n, g, m, s, size)
    Q = randomQ(g,m)
    sum = 0.
    for _ in 1:size
        sum += norm(pseudoinverse(sketch(Q, n, "srht_hash", s=s)))^2
    end
    return sum/size
end

function randomQ(n,m)
    A = randn(n, m)
    Q, _ = qr(A)
    return Matrix(Q[:, 1:m])
end

times = 30

n = 128
p = 256
m = 64
s = 10

nn = collect(m+2:25:1300)
pp = collect(128:25:1000)
mm = collect(1:3:(n-2))
ss = collect(1:20)

results_n = Float64[]
for n_ in nn
    println("n = ", n_)
    result = expected_fnorm2(n_, p, m, s, times)
    push!(results_n, result)
end

results_p = Float64[]
for p_ in pp
    println("p = ", p_)
    result = expected_fnorm2(n, p_, m, s, times)
    push!(results_p, result)
end

results_m = Float64[]
for m_ in mm
    println("m = ", m_)
    result = expected_fnorm2(n, p, m_, s, times)
    push!(results_m, result)
end

results_s = Float64[]
for s_ in ss
    println("s = ", s_)
    result = expected_fnorm2(n, p, m, s_, times)
    push!(results_s, result)
end

println("n = ", n)
println("p = ", p)
println("m = ", m)
println("s = ", s)
println("nn = np.array(", nn, ")")
println("pp = np.array(", pp, ")")
println("mm = np.array(", mm, ")")
println("ss = np.array(", ss, ")")
println("res_n = np.array(", results_n, ")")
println("res_p = np.array(", results_p, ")")
println("res_m = np.array(", results_m, ")")
println("res_s = np.array(", results_s, ")")