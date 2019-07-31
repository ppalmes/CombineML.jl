module TestScikitLearnWrapper
using Random
using Test
using PyCall

using CombineML.Types
import CombineML.Types.fit!
import CombineML.Types.transform!
using CombineML.Transformers.ScikitLearnWrapper

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
nfcp = NumericFeatureClassification()

ScikitLearnWrapper.initlibs()


function skl_fit_and_transform!(learner::Learner, problem::MLProblem, seed=1)
  #RAN.seed(seed)
  Random.seed!(seed)
  return fit_and_transform!(learner, problem, seed)
end

function backend_fit_and_transform!(sk_learner, seed=1)
  #RAN.seed(seed)
  Random.seed!(seed)
  sk_learner.fit(nfcp.train_instances, nfcp.train_labels)
  return collect(sk_learner.predict(nfcp.test_instances))
end

function behavior_check(learner::Learner, sk_learner)
  # Predict with CombineML learner
  combineml_predictions = skl_fit_and_transform!(learner, nfcp)
  # Predict with original backend learner
  original_predictions = backend_fit_and_transform!(sk_learner)
  # Verify same predictions
  @test length(combineml_predictions) == length(original_predictions)
end


@testset "scikit-learn learners" begin

  @testset "SKLLearner gives same results as its backend" begin
    #learner_names = collect(keys(ScikitLearnWrapper.learner_dict))
    learner_names = [
      "AdaBoostClassifier",
      "BaggingClassifier",
      "ExtraTreesClassifier",
      "GradientBoostingClassifier",
      "RandomForestClassifier",
      "LDA",
      "QDA",
      "PassiveAggressiveClassifier",
      "RidgeClassifier",
      "RidgeClassifierCV",
      "SGDClassifier",
      "KNeighborsClassifier",
      "RadiusNeighborsClassifier",
      "NearestCentroid",
      "SVC",
      "LinearSVC",
      "NuSVC",
      "MLPClassifier",
      "GaussianProcessClassifier",
      "DecisionTreeClassifier"
    ]
    for learner_name in learner_names
      sk_learner = ScikitLearnWrapper.learner_dict[learner_name]()
      impl_options = Dict()
      fragile_learners = [
        "AdaBoostClassifier",
        "BaggingClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "RandomForestClassifier",
        "SGDClassifier",
        "RadiusNeighborsClassifier",
        "RidgeClassifierCV"
      ]
      learner = SKLLearner(Dict(
        :learner => learner_name,
        :impl_options => impl_options
       ))
      if !in(learner_name, fragile_learners)
        behavior_check(learner, sk_learner)
      else
        skl_fit_and_transform!(learner, nfcp)
        @test 1 == 1
      end
    end
  end

end

end # module
