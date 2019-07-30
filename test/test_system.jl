# System tests.
module TestSystem

using DataFrames
using RDatasets
#using DelimitedFiles
import CombineML
using InteractiveUtils
using CombineML.Types
using CombineML.System
using CombineML.Transformers

if LIB_SKL_AVAILABLE
  using CombineML.Transformers.ScikitLearnWrapper
end
if LIB_CRT_AVAILABLE
  using CombineML.Transformers.CaretWrapper
end
 

using CombineML.Util
using Test

include("fixture_learners.jl")
using .FixtureLearners
nfcp = NumericFeatureClassification()
fcp = FixtureLearners.FeatureClassification()

function all_concrete_subtypes(a_type::Type)
  a_subtypes = Type[]
  for a_subtype in subtypes(a_type)
    if isconcretetype(a_subtype)
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


@testset "CombineML system" begin

  @testset "All learners train and predict on fixture data." begin
    for concrete_learner_type in concrete_learner_types
      learner = concrete_learner_type()
      fit_and_transform!(learner, nfcp)
      @test learner.model != Nothing
    end
  end

  @testset "All learners train and predict on iris dataset." begin
    # Get data
    #mdataset = readdlm(joinpath(dirname(pathof(CombineML)),"../test", "iris.csv"),',')
    mdataset = dataset("datasets","iris")
    features = mdataset[:,1:(end-1)]
    labels = (mdataset[:, end]) |> collect
    (train_ind, test_ind) = holdout(size(features, 1), 0.3)
    train_features = features[train_ind, :] |> Matrix
    test_features = features[test_ind, :] |> Matrix
    train_labels = labels[train_ind] |> Vector
    test_labels = labels[test_ind] |> Vector
    #m = CRTLearner()
    #fit!(m,train_features,train_labels)
    # Test all learners
    for concrete_learner_type in concrete_learner_types
      learner = concrete_learner_type()
      fit!(learner, train_features, train_labels)
      transform!(learner, test_features)
      @test learner.model != Nothing
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
      #OneHotEncoder(),
      #Imputer(),
      #StandardScaler(),
      BestLearner()
    ]
    mpipeline = Pipeline(Dict(:transformers => transformers))
    predictions = fit_and_transform!(mpipeline, fcp)
    @test predictions == Any["a","a","b","b","a","a","d","d"] || predictions == Any["a","a","b","b","c","c","d","d"]
  end

end

end # module
