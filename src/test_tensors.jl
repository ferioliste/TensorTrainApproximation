function hilbert_tensor(n, d)
    H = zeros(fill(n,d)...)

    for indices in CartesianIndices(H)
        H[indices] = 1 / (sum(indices.I) + d - 1)
    end
    
    return H
end

function square_root_sum_tensor(n, d, a, b)
    S = zeros(fill(n,d)...)

    for indices in CartesianIndices(S)
        idxs = indices.I
        S[indices] = sqrt(abs(sum([a*(n-idxs[j])/(n-1) + b*(idxs[j]-1)/(n-1) for j in 1:d])))
    end
    
    return S
end

function square_root_sum_tensor_v2(n, d, a, b)
    S = zeros(fill(n,d)...)

    for indices in CartesianIndices(S)
        idxs = indices.I
        S[indices] = sqrt(sum([a*(n-idxs[j])/(n-1) + b*(idxs[j]-1)/(n-1) for j in 1:d]))
    end
    
    return S
end

function random_tensor(n, d)
    return randn(fill(n,d)...)
end