# System tests.
module TestSystem

using CombineML.Types
using CombineML.System
using CombineML.Transformers
importall CombineML.Util

include("fixture_learners.jl")
using .FixtureLearners
nfcp = NumericFeatureClassification()
fcp = FixtureLearners.FeatureClassification()

function all_concrete_subtypes(a_type::Type)
  a_subtypes = Type[]
  for a_subtype in subtypes(a_type)
    if isleaftype(a_subtype)
      push!(a_subtypes, a_subtype)
    else
      append!(a_subtypes, all_concrete_subtypes(a_subtype))
    end
  end
  return a_subtypes
end

concrete_learner_types = setdiff(
  all_concrete_subtypes(Learner),
  all_concrete_subtypes(TestLearner)
)

using Base.Test

@testset "CombineML system" begin

  @testset "All learners train and predict on fixture data." begin
    for concrete_learner_type in concrete_learner_types
      learner = concrete_learner_type()
      fit_and_transform!(learner, nfcp)
      @test learner.model != Void
    end
  end

  @testset "All learners train and predict on iris dataset." begin
    # Get data
    dataset = readcsv(joinpath(Pkg.dir("CombineML"),"test", "iris.csv"))
    features = dataset[:,1:(end-1)]
    labels = dataset[:, end]
    (train_ind, test_ind) = holdout(size(features, 1), 0.3)
    train_features = features[train_ind, :]
    test_features = features[test_ind, :]
    train_labels = labels[train_ind]
    test_labels = labels[test_ind]
    # Test all learners
    for concrete_learner_type in concrete_learner_types
      learner = concrete_learner_type()
      fit!(learner, train_features, train_labels)
      transform!(learner, test_features)
      @test learner.model != Void
    end
  end

  @testset "Ensemble with learners from different libraries work." begin 
    learners = Learner[]
    push!(learners, RandomForest())
    push!(learners, StackEnsemble())
    if LIB_SKL_AVAILABLE
      push!(learners, SKLLearner())
    end
    if LIB_CRT_AVAILABLE
      push!(learners, CRTLearner())
    end
    ensemble = VoteEnsemble(Dict(:learners => learners))
    predictions = fit_and_transform!(ensemble, nfcp)
    @test predictions == Any["a","a","b","b","a","a","d","d"]
  end

  @testset "Pipeline works with fixture data." begin
    transformers = [
      OneHotEncoder(),
      Imputer(),
      StandardScaler(),
      BestLearner()
    ]
    pipeline = Pipeline(Dict(:transformers => transformers))
    predictions = fit_and_transform!(pipeline, fcp)
    @test predictions == Any["a","a","b","b","a","a","d","d"] || prediction == Any["a","a","b","b","c","c","d","d"]
  end

end

end # module
