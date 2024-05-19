n = 5
d = 7
T = random_tensor(n, d)
#hilbert_tensor(n, d)
norm_T = norm(T)

dims = size(T)
ranks = fix_ranks(fill(10000,d-1), dims)
TTT = tt_svd(T,ranks)

approx = tt_rsvd_rounding(TTT, ranks)
# approx = tt_STTA_rounding(TTT, ranks, ranks.*2; sketch_type = "srht_hash", s = 1, seed = nothing)
T_ = tt2full(approx)
println(norm(T-T_)/norm_T)