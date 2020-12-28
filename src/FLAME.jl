module FLAME

using Clustering: display_level
using Distances
using NearestNeighbors
using Printf

export

    FLAMEResult,
    extractstructure, 
    initializemembership!, 
    distances2similarities,
    flame,
    construct_clusters

const _flame_default_maxiter = 100
const _flame_default_tol = 1.0e-6
const _flame_default_display = :none


struct FLAMEResult{T<:AbstractFloat,OT<:Integer}
    memberships::Matrix{T}
    csos::Vector{OT}
    outliers::Vector{OT}
    rests::Vector{OT}
    iterations::Int             
    converged::Bool             
end

"""
Create a tree from `data` using the given `metric`.
"""
function gettree(data, algorithm, metric)
    if algorithm == "brute"
        BruteTree(data, metric)
    elseif  algorithm == "kd"
        KDTree(data, metric)
    elseif algorithm == "ball"
        BallTree(data, metric)
    else
        throw(ArgumentError("Invalid algorithm."))
    end
end

"""
K-nearest neighbors without the querying points.
"""
function _knn(data::AbstractMatrix{T}, k::Int; algorithm::String="brute", metric::MT=Euclidean()) where {T,MT}
    tree = gettree(data, algorithm, metric)
    idxs, dists = knn(tree, data, k+1, true)
    [idx[2:end] for idx in idxs], [dist[2:end] for dist in dists]
end

"""
Extract structure information from the data.
"""
function extractstructure(idxs::AbstractVector{IVT}, 
                          dists::AbstractVector{FVT}; 
                          threshold::T) where {IVT,FVT,T<:AbstractFloat}
    totaldists = sum.(dists)
    maxdist = maximum(totaldists)
    densities = maxdist ./ totaldists
    csos = Int[]
    outliers = Int[]
    rests = Int[]
    for (i, (idx, density)) in enumerate(zip(idxs, densities))
        if (density < threshold) && (density < minimum(densities[idx]))
            push!(outliers, i)
        elseif density > maximum(densities[idx])
            push!(csos, i)
        else
            push!(rests, i)
        end
    end
    csos, outliers, rests
end

"""
Initialize fuzzy memberships.
"""
function initializemembership!(memberships::AbstractMatrix{PT}, csos::OT, outliers::OT, rests::OT) where {PT,OT<:AbstractVector}
    numdata = size(memberships, 1)
    numcsos = length(csos)
    #memberships = zeros(numdata, numcsos+1)
    fixed = ones(numcsos+1) ./ (numcsos+1)
    for i in 1:numdata
        if i in csos
            memberships[i,findall(i .== csos)] .= 1.0
        elseif i in outliers
            memberships[i,end] = 1.0
        elseif i in rests
            memberships[i,:] = fixed
        else
            throw(ArgumentError("Index not found in Cluster Supporting Object, Cluster Outliers or the rest"))
        end 
    end
    memberships
end

"""
Transform distances to similarities.
"""
function distances2similarities(ds::AbstractVector{T}) where T<:AbstractFloat
    normds = ds ./ sum(ds)
    1 .- normds
end

function update_memberships(ps::AbstractMatrix{T}, ds) where T<:AbstractFloat
    sims = distances2similarities(ds)
    weights = sims ./ sum(sims)
    sum(weights .* ps, dims=1)
end

"""
Fuzzy clustering by Local Approximation of MEmbership (FLAME).

Fu, L., Medico, E. FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data. BMC Bioinformatics 8, 3 (2007).
"""
function flame(data::AbstractMatrix{T}, 
               k::Integer; 
               algorithm::String="brute", 
               metric::MT=Euclidean(),
               threshold::PT=2.0, 
               maxiter::Integer=_flame_default_maxiter, 
               tol::PT=_flame_default_tol,
               display::Symbol=_flame_default_display) where {T<:Real,MT,PT}
    numdata = size(data,2)
    idxs, dists = _knn(data, k; algorithm=algorithm, metric=metric)
    csos, outliers, rests = extractstructure(idxs, dists; threshold=threshold)
    memberships = zeros(PT, numdata, length(csos)+1)
    memberships = initializemembership!(memberships, csos, outliers, rests)
    displevel = display_level(display)
    prev_objv = δ = typemax(T)
    iter = 0
    converged = false
    while !converged && iter < maxiter
        iter += 1
        newmemberships = copy(memberships)
        for rest in rests
            newmemberships[rest,:] = update_memberships(memberships[idxs[rest],:], dists[rest])
        end
        objv = sqrt.(sum(abs2, newmemberships .- memberships) / length(memberships))
        objv_change = objv - prev_objv
        δ = abs(objv_change)
        if δ <= tol 
            converged = true
        else
            prev_objv = objv
            memberships = newmemberships
        end
        if displevel >= 2
            @printf("%7d %18.6e\n", iter, δ)
        end
    end

    if displevel >= 1
        if converged
            println("FLAME converged with $iter iterations (δ = $δ)")
        else
            println("FLAME terminated without convergence after $iter iterations (δ = $δ)")
        end
    end

    FLAMEResult(memberships, csos, outliers, rests, iter, converged)
end

"""
Get cluster assignment vectors based on the highest membership score.
"""
function construct_clusters(result::FLAMEResult)
    m = size(result.memberships, 2)
    indices = argmax(result.memberships; dims=2)
    xs = Int[]
    for index ∈ indices
        ci = last(index.I)
        if ci != m
            push!(xs, ci)
        else
            push!(xs, 0)
        end
    end
    xs
end

end # module