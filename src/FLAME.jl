module FLAME

using Clustering: display_level
using Distances
using NearestNeighbors
using NearestNeighbors: copy_svec
using Printf

export

    FLAMEResult,
    extractstructure, 
    knn, 
    initializemembership, 
    distances2similarities,
    flame

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
K-nearest neighbors without the querying points.
"""
function NearestNeighbors.knn(data::AbstractVecOrMat{T}, k::Integer=3; metric::M=Euclidean()) where {T<:AbstractFloat,M<:Metric}
    tree = BruteTree(data, metric)
    idxs, dists = knn(tree, data, k+1, true)
    [idx[2:end] for idx in idxs], [dist[2:end] for dist in dists]
end

"""
Extract structure information from the data.
"""
function extractstructure(idxs::AbstractVector{IVT}, dists::AbstractVector{FVT}; threshold::T) where {IVT,FVT,T<:AbstractFloat}
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
function initializemembership(numdata::Integer, csos::OT, outliers::OT, rests::OT) where OT<:AbstractVector
    numcsos = length(csos)
    memberships = zeros(numdata, numcsos+1)
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
function flame(data::AbstractVecOrMat{T}, 
               k::Integer; 
               metric::M=Euclidean(), 
               threshold::T=2.0, 
               maxiter::Integer=_flame_default_maxiter, 
               tol::T=_flame_default_tol,
               display::Symbol=_flame_default_display) where {T<:AbstractFloat,M<:Metric}
    numdata = size(data,2)
    idxs, dists = knn(data, k; metric=metric)
    csos, outliers, rests = extractstructure(idxs, dists; threshold=threshold)
    memberships = initializemembership(numdata, csos, outliers, rests)
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

end # module