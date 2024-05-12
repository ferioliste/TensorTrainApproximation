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

if false
    m = 128
    g = 256
    nn = collect(m+1:15:1300)
    s = 10

    results = Float64[]
    for n in nn
        println(n)
        result = expected_fnorm2(n, g, m, s, 10)
        push!(results, result)
    end

    plot(nn, results, yscale=:log10, xscale=:log10)
    plot!(nn, (m.*nn.^2)./((nn.-m).*(s.^2).*ceillog2.(g)), linestyle=:dash, color=:blue, label="x")
    #plot!(nn, nn.^2, linestyle=:dash, color=:blue, label="x^2")
    plot!(legend=false)

    println(nn)
    println(results)
    savefig("../plots/plot3.png")
end

if true
    n = 32
    g = 256
    mm = collect(1:2:n)
    s = 10

    results = Float64[]
    for m in mm
        println(m)
        result = expected_fnorm2(n, g, m, s, 10)
        push!(results, result)
    end

    plot(mm, results, yscale=:log10, xscale=:log10)
    plot!(mm, (mm.*n.^2)./((n.-mm).*(s.^2).*ceillog2.(g)), linestyle=:dash, color=:blue, label="x")
    # plot!(mm, mm, linestyle=:dash, color=:red, label="x")
    # plot!(mm, mm.^2, linestyle=:dash, color=:red, label="x")
    plot!(mm, s*(mm .* (n+1))./(n .- mm))
    plot!(legend=false)

    println(mm)
    println(results)
    savefig("../plots/plot3.png")
end

if false
    m = 64
    g = 256
    n = 128
    ss = collect(1:20)

    results = Float64[]
    for s in ss
        println(s)
        result = expected_fnorm2(n, g, m, s, 10)
        push!(results, result)
    end

    plot(ss, results, yscale=:log10, xscale=:log10)
    plot!(ss, ss.^-1, linestyle=:dash, color=:red, label="x")
    plot!(ss, ss.^-2, linestyle=:dash, color=:blue, label="x^2")
    plot!(legend=false)

    println(ss)
    println(results)
    savefig("../plots/plot4.png")
end

if false
    m = 128
    gg = collect(128:15:1000)
    n = 256
    s = 10

    results = Float64[]
    for g in gg
        println(g)
        result = expected_fnorm2(n, g, m, s, 10)
        push!(results, result)
    end

    plot(gg, results, yscale=:log10, xscale=:log10)
    #plot!(gg, s*(m .* (nn+1))./(nn .- m))
    plot!(gg, gg.^2, linestyle=:dash, color=:blue, label="x^2")
    plot!(legend=false)

    println(gg)
    println(results)
    savefig("../plots/plot3.png")
end