using FLAME

using Test

@testset "FLAME" begin

    numdata1 = 249
    numdata2 = 249
    data1 = 0.15 .* randn(2, numdata1) .- 0.5
    data2 = 0.15 .* randn(2, numdata2) .+ 0.5
    outliers = [-1.0 1.0; 1.0 -1.0]
    data = [data1 data2 outliers]
    numdata = size(data,2)

    k = 100
    threshold = 2.0
    idxs, dists = knn(data, k)
    csos, outliers, rests = extractstructure(idxs, dists; threshold=threshold)
    @test length(csos) == 2
    @test outliers == [size(data,2)-1, size(data,2)]
    
    memberships = initializemembership(numdata, csos, outliers, rests)
    numdata = size(data,2)
    numcsos = length(csos)
    for cso in csos
        x = zeros(length(csos)+1)
        x[findall(cso .== csos)] .= 1.0
        @test memberships[cso,:] == x
    end
    for outlier in outliers
        @test memberships[outlier,end] == 1.0
    end
    fixed = ones(numcsos+1) ./ (numcsos+1)
    for rest in rests
        @test memberships[rest,:] == fixed
    end

    res = flame(data, k; threshold=threshold)
    vs = vec(argmax(res.memberships[rests,1:end-1]; dims=2))
    labels = [v[2] for v in vs]
    @test labels == [ones(Int, numdata1-1); 2 * ones(Int, numdata1-1)]

    ds = [0.5,0.1,1.3,1.1]
    sims = distances2similarities(ds)
    @test sims ≈ [0.8333,0.9667,0.5667,0.6333] atol=0.0001
end