module TestMLBaseWrapper

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
fcp = FeatureClassification()

using Base.Test

importall CombineML.Transformers.MLBaseWrapper

@testset "MLBase transformers" begin

  @testset "StandardScaler transforms features" begin
    instances = [
      5 10;
      -5 0;
      0 5;
    ]
    labels = [
      "x";
      "y";
      "z";
    ]
    expected_transformed = [
      1.0 1.0;
      -1.0 -1.0;
      0.0 0.0;
    ]
    standard_scaler = StandardScaler()
    fit!(standard_scaler, instances, labels)
    transformed = transform!(standard_scaler, instances)
    @test transformed == expected_transformed
  end

end

end # module
