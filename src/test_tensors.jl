function hilbert_tensor(n, d)
    H = zeros(fill(n,d)...)

    for indices in CartesianIndices(H)
        H[indices] = 1 / (sum(indices.I) + d - 1)
    end
    
    return H
end