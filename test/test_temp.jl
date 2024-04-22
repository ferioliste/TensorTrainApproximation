function generate_tt(n,d)
    dims=ntuple(_ -> n, d)
    v=randn(ntuple(_ -> n, d)...)
    tol=1e-12
    mps= tt_STTA(v, fix_ranks(fill(10000, d-1), dims), fix_ranks(fill(10000, d-1), dims)*2) #tt_rsvd(v,100000)
    return v, mps
end

v, mps = generate_tt(5,7)
# tt_rounding_olese!(mps; tt_ranks = tt_ranks(mps), tol=1e-12, truncation=false)
# tt_rounding_rand_ort!(mps; tt_ranks = tt_ranks(mps), tol=1e-12, truncation=false)
# tt_rounding_kres_left!(mps; tt_l_ranks = ceil.(Int, tt_ranks(mps) .* 1.1), tt_r_ranks = tt_ranks(mps))
#println(tt_ranks(mps))
#tt_rounding_fwht_left!(mps; tt_l_ranks = ceil.(Int, tt_ranks(mps) .* 2), tt_r_ranks = tt_ranks(mps))
println(tt_ranks(mps))

v_ = tt2full(mps)
println(norm(v-v_)/norm(v))