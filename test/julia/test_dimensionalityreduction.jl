module TestDimensionalityReductionWrapper

using Test
import MultivariateStats: reconstruct
using CombineML.Transformers.DimensionalityReductionWrapper

#include(joinpath("..", "fixture_learners.jl"))
#using .FixtureLearners
#fcp = FeatureClassification()


@testset "DimensionalityReduction transformers" begin
  @testset "PCA transforms features" begin

    instances = [
      5.0 10.0;
      -5.0 0.0;
      0.0 5.0;
    ]
    labels = ["x"; "y"; "z"]

    options = Dict(:pratio => 1.0,:maxoutdim => 2)
    pca = PCA(options)
    fit!(pca, instances, labels)
    transformed = transform!(pca, instances)
    @test reconstruct(pca.model,transformed')' ≈ instances
    @test  transformed * pca.model.proj' .+ pca.model.mean' ≈ instances

  end
end

end # module
