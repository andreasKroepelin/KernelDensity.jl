"""
$(TYPEDEF)

Store both grid and density for KDE over the real line.

Reading the fields directly is part of the API, and

```julia
sum(density) * prod(step.(ranges)) ≈ 1
```

# Fields

$(FIELDS)
"""
struct MultivariateKDE{N, R <: NTuple{N, AbstractRange}} <: AbstractKDE
    "Coordinates of gridpoints for evaluating density"
    ranges::R
    "Kernel density at corresponding gridpoints."
    density::Array{Float64, N}
end

function kernel_dist(::Type{D}, w::NTuple{N, Real}) where {N, D <: UnivariateDistribution}
    kernel_dist.(D, w)
end

@generated function kernel_dist(::Type{Ds}, w::NTuple{N, Real}) where {N, Ds <: NTuple{N, UnivariateDistribution}}
    quote
        kernel_dist.(
            # Ds.parameters is of type svec which does not have a compile time
            # known length. By splatting it into a tuple at compile time, proper
            # type inference is possible.
            ($(Ds.parameters...),),
            w
        )
    end
end

const DataTypeOrUnionAll = Union{DataType, UnionAll}

# this function provided for backwards compatibility, though it doesn't have the type restrictions
# to ensure that the given tuple only contains univariate distributions
function kernel_dist(d::NTuple{N, DataTypeOrUnionAll}, w::NTuple{N, Real}) where N
    kernel_dist.(d, w)
end

# TODO: there are probably better choices.
function default_bandwidth(data::NTuple{N, RealVector}) where N
    default_bandwidth.(data)
end

@generated function interpolate_in_hypercube!(
    grid::Array{Float64, N},
    midpoints::NTuple{N, AbstractRange},
    coords::NTuple{N, Real},
    high_idcs::NTuple{N, Int},
    low_idcs::NTuple{N, Int},
    weight::Real
) where N

    interpolate_in_hypercube!_impl(Val(N))

end

function interpolate_in_hypercube!_impl(::Val{N}) where N

    # We want to interpolate the data point at `coords` on the hypercube around it,
    # defined by the grid spanned by `midpoints`.
    # `high_idcs` and `low_idcs` tell us where in `midpoints` we have to look to
    # find the vertices of that hypercube, of which there are 2^N.
    # `sides_of_vertices` will store 2^N arrays of zeros and ones of length N,
    # each representing one vertex.
    # A zero means that the vertex has the lower of the two possible coordinates
    # in that specific dimension, a one means it's the larger one.
    # The concrete process of constructing `slides_of_vertices` makes sure that
    # the first entry varies the slowest and the last entry varies the fastest.
    # This leads to the proper "column"-major iteration order later.
    sides_of_vertices = [[]]
    for _ in 1:N
        new_sides_of_vertices = []
        for sides in sides_of_vertices
            push!(new_sides_of_vertices, vcat(sides, [0]))
            push!(new_sides_of_vertices, vcat(sides, [1]))
        end
        sides_of_vertices = new_sides_of_vertices
    end
    
    # `low_idcs` stores the lower indices and `high_idcs` stores the higher
    # indices and this function connects that to the 0/1-representation in
    # `sides_of_vertices`.
    side_to_symbol(side) = side == 0 ? :low_idcs : :high_idcs
    
    # `indices_of_vertices` stores the array of indices that will be used to
    # access a vertex of the hypercube later, for each vertex.
    indices_of_vertices = [
        [:($(side_to_symbol(side))[$i]) for (side, i) in zip(sides, 1:N)]
        for sides in sides_of_vertices
    ]
    
    # `factors_of_vertices` is concerned with interpolation coefficients.
    # The idea is that the contribution of the data point to a specific vertex
    # in a specific dimension is proportional to the distance of the data point
    # to the opposite face of the hypercube in that dimension.
    # This distance can be obtained as
    # `midpoints[dimension][other_side[dimension]] - coords[dimension]`.
    # The pseudo variable `other_side` can be computed simply by `1 - side`,
    # thanks to the 0/1-representation.
    # Using all that information we can then construct the factors for interpolation.
    factors_of_vertices = [
        [
            :(midpoints[$i][$(side_to_symbol(1-side))[$i]] - coords[$i])
            for (side, i) in zip(sides, 1:N)
        ]
        for sides in sides_of_vertices
    ]
    
    # When calculating the factors before, we actually used the signed distance
    # to the opposite side, which is always wrong when the opposite side has the
    # lower possible coordinate in one dimension.
    # Therefore, for every side represented by 1, we introduce a multiplicative
    # error of -1.
    # By counting the ones for a vertex and raising -1 to that power we can find
    # a correction forthat error.
    sign_corrections = (-1) .^ map(sum, sides_of_vertices)
    
    # Using this preprocessing, we can now create accesses to `grid`.
    updates = [
        :(@inbounds grid[$(indices...)] += *($sign_correction, weight, $(factors...)))
        for (indices, factors, sign_correction)
        in zip(indices_of_vertices, factors_of_vertices, sign_corrections)
    ]
    
    code = quote end
    for update in updates
        code = quote
            $code
            $update
        end
    end
    
    code
end

# tabulate data for kde
function tabulate(
    data::NTuple{N, RealVector},
    midpoints::NTuple{N, AbstractRange},
    weights::Weights = default_weights(data)
) where N

    ndata = length(first(data))
    all(==(ndata) ∘ length, data) || error("data vectors must be of same length")

    range_lengths = length.(midpoints)
    range_steps = step.(midpoints)

    # Set up a grid for discretized data
    grid = zeros(Float64, range_lengths...)
    ainc = 1.0 / (sum(weights) * prod(range_steps)^2)

    # weighted discretization (cf. Jones and Lotwick)
    for i in 1:ndata
        coords = getindex.(data, i)
        high_idcs = searchsortedfirst.(midpoints, coords)
        low_idcs = high_idcs .- 1
        if all(1 .<= low_idcs .<= range_lengths .- 1)
            interpolate_in_hypercube!(grid, midpoints, coords, high_idcs, low_idcs, weights[i])
        end
    end
    # Normalize for interpolation coefficients and weights
    grid .*= ainc

    # returns an un-convolved KDE
    MultivariateKDE(midpoints, grid)
end

# convolution with product distribution of N univariate distributions
function conv(k::MultivariateKDE{N, R}, dists::NTuple{N, UnivariateDistribution}) where {N, R}
    # Transform to Fourier basis
    ft = rfft(k.density)

    # Convolve fft with characteristic function of kernel
    cs = -twoπ ./ (maximum.(k.ranges) .- minimum.(k.ranges))
    for idx in CartesianIndices(ft)
        pos = Tuple(idx) .- 1
        pos = min.(pos, size(k.density) .- pos)
        ft[idx] *= prod(cf.(dists, pos .* cs))
    end

    # Invert the Fourier transform to get the KDE
    density = irfft(ft, size(k.density, 1))

    density .= max.(0., density)

    MultivariateKDE(k.ranges, density)
end

default_weights(data::NTuple{N, RealVector}) where N = UniformWeights(length(data[1]))

function kde(
    data::NTuple{N, RealVector},
    weights::Weights,
    midpoints::NTuple{N, AbstractRange},
    dist::NTuple{N, UnivariateDistribution}
) where N

    k = tabulate(data, midpoints, weights)
    conv(k, dist)
end

function kde(
    data::NTuple{N, RealVector},
    dist::NTuple{N, UnivariateDistribution};
    boundary::NTuple{N, Tuple{Real, Real}} =
        kde_boundary.(data, std.(dist)),
    npoints::NTuple{N, Int} = ntuple(_ -> 256, Val(N)),
    weights::Weights = default_weights(data)
) where N

    midpoints = kde_range.(boundary, npoints)
    kde(data, weights, midpoints, dist)
end

function kde(
    data::NTuple{N, RealVector},
    midpoints::NTuple{N, AbstractRange};
    bandwidth = default_bandwidth(data),
    kernel = Normal,
    weights::Weights = default_weights(data)
) where N

    dist = kernel_dist(kernel, bandwidth)
    kde(data, weights, midpoints, dist)
end

function kde(
    data::NTuple{N, RealVector};
    bandwidth = default_bandwidth(data),
    kernel = Normal,
    boundary::NTuple{N, Tuple{Real, Real}} = kde_boundary.(data, bandwidth),
    npoints::NTuple{N, Int} = ntuple(_ -> 256, Val(N)),
    weights::Weights = default_weights(data)
) where N

    dist = kernel_dist(kernel, bandwidth)
    midpoints = kde_range.(boundary, npoints)

    kde(data, weights, midpoints, dist)
end

# matrix data
function kde(data::RealMatrix, args...; kwargs...)
    kde(
        ntuple(i -> view(data, :, i), size(data, 2)),
        args...; kwargs...
    )
end
