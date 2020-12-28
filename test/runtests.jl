using FLAME

using Distances, Test

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
    for algorithm ∈ ["brute","kd","ball"]
        idxs, dists = FLAME._knn(data, k; algorithm=algorithm, metric=Euclidean())
        csos, outliers, rests = extractstructure(idxs, dists; threshold=threshold)
        @test length(csos) == 2
        @test outliers == [size(data,2)-1, size(data,2)]
    end
    @test_throws ArgumentError FLAME._knn(data, k; algorithm="test", metric=Euclidean())
    
    idxs, dists = FLAME._knn(data, k; algorithm="brute", metric=Euclidean())
    csos, outliers, rests = extractstructure(idxs, dists; threshold=threshold)
    memberships = zeros(numdata, length(csos)+1)
    initializemembership!(memberships, csos, outliers, rests)
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
    @test construct_clusters(res) == [ones(Int, numdata1); 2 .* ones(Int, numdata2); [0,0]]

    ds = [0.5,0.1,1.3,1.1]
    sims = distances2similarities(ds)
    @test sims ≈ [0.8333,0.9667,0.5667,0.6333] atol=0.0001

end