function test_tensor(tensor_type, n, d; a=-0.2, b=2)
    if tensor_type == "hilbert"
        return hilbert_tensor(n, d)
    elseif tensor_type == "sqrt_sum"
        return square_root_sum_tensor(n, d, a, b)
    elseif tensor_type == "random"
        return random_tensor(n, d)
    else
        error("Unsupported tensor type: $tensor_type")
    end
end

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

function random_tensor(n, d)
    return randn(fill(n,d)...)
end