const BivariateKDE{Rx <: AbstractRange, Ry <: AbstractRange} =
    MultivariateKDE{2, Tuple{Rx, Ry}}


const BivariateDistribution = Union{Tuple{UnivariateDistribution, UnivariateDistribution}}
