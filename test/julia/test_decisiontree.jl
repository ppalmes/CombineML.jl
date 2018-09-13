module TestDecisionTreeWrapper

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
nfcp = NumericFeatureClassification()

using Test
using Random
using CombineML.Transformers.DecisionTreeWrapper
using DecisionTree

@testset "DecisionTree learners" begin

  @testset "PrunedTree gives same results as its backend" begin
    # Predict with CombineML learner
    learner = PrunedTree()
    combineml_predictions = fit_and_transform!(learner, nfcp)
    # Predict with original backend learner
    Random.seed!(1)
    model = build_tree(
      nfcp.train_labels,
      nfcp.train_instances,
      0,  # num_subfeatures
      -1, # max_depth
      1,  # min_samples_leaf
      2,  # min_samples_split
      0.0 # min_purity_increase
    )
    model = prune_tree(model, 1.0)
    original_predictions = apply_tree(model, nfcp.test_instances)
    # Verify same predictions
    @test combineml_predictions == original_predictions
  end

  @testset "RandomForest gives same results as its backend" begin
    # Predict with CombineML learner
    learner = RandomForest()
    combineml_predictions = fit_and_transform!(learner, nfcp)
    # Predict with original backend learner
    Random.seed!(1)
    model = build_forest(
      nfcp.train_labels,
      nfcp.train_instances,
      size(nfcp.train_instances, 2),
      10,
      0.7,
      -1
    )
    original_predictions = apply_forest(model, nfcp.test_instances)
    # Verify same predictions
    @test combineml_predictions == original_predictions
  end

  @testset "DecisionStumpAdaboost gives same results as its backend" begin
    # Predict with CombineML learner
    learner = DecisionStumpAdaboost()
    combineml_predictions = fit_and_transform!(learner, nfcp)
    # Predict with original backend learner
    Random.seed!(1)
    model, coeffs = build_adaboost_stumps(
      nfcp.train_labels,
      nfcp.train_instances,
      7
    )
    original_predictions = apply_adaboost_stumps(
      model, coeffs, nfcp.test_instances
    )
    # Verify same predictions
    @test combineml_predictions == original_predictions
  end

  @testset "RandomForest handles training-dependent options" begin
    # Predict with CombineML learner
    learner = RandomForest(Dict(:impl_options => Dict(:num_subfeatures => 2)))
    combineml_predictions = fit_and_transform!(learner, nfcp)
    # Verify RandomForest didn't die
    @test combineml_predictions == Any["a","a","b","b","a","a","d","d"] 
  end

end

end # module
