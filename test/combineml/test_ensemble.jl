module TestEnsembleMethods

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
nfcp = NumericFeatureClassification()

using Test
using Random

using CombineML.Transformers.EnsembleMethods
import CombineML.Transformers.DecisionTreeWrapper: fit!, transform!
import CombineML.Transformers.DecisionTreeWrapper: PrunedTree
import CombineML.Transformers.DecisionTreeWrapper: RandomForest
import CombineML.Transformers.DecisionTreeWrapper: DecisionStumpAdaboost

@testset "Ensemble learners" begin

  @testset "VoteEnsemble predicts according to majority" begin
    always_a_options = Dict( :label => :a )
    always_b_options = Dict( :label => :b ) 
    learner = VoteEnsemble(Dict(
      :learners => [
        AlwaysSameLabelLearner(always_a_options),
        AlwaysSameLabelLearner(always_a_options),
        AlwaysSameLabelLearner(always_b_options)
      ]
     ))
    fit!(learner, nfcp.train_instances, nfcp.train_labels)
    predictions = transform!(learner, nfcp.test_instances)
    expected_predictions = fill(:a, size(nfcp.test_instances, 1))
    @test  predictions == expected_predictions
  end

#  @testset "StackEnsemble predicts with CombineMLd learners" begin
#    # Fix random seed, due to stochasticity in stacker.
#    Random.seed!(2)
#    always_a_options = Dict( :label => "a" )
#    learner = StackEnsemble(Dict(
#      :learners => [
#        AlwaysSameLabelLearner(always_a_options),
#        AlwaysSameLabelLearner(always_a_options),
#        PerfectScoreLearner()
#      ],
#      :keep_original_features => true
#     ))
#    fit!(learner, nfcp.train_instances, nfcp.train_labels)
#    predictions = transform!(learner, nfcp.test_instances)
#    unexpected_predictions = fill("a", size(nfcp.test_instances, 1))
#    @test predictions == not(unexpected_predictions)
#  end

#  @testset "BestLearner picks the best learner" begin
#    always_a_options = Dict( :label => "a" )
#    always_b_options = Dict( :label => "b" )
#    learner = BestLearner(Dict(
#      :learners => [
#        AlwaysSameLabelLearner(always_a_options),
#        PerfectScoreLearner(),
#        AlwaysSameLabelLearner(always_b_options)
#      ]
#     ))
#    fit!(learner, nfcp.train_instances, nfcp.train_labels)
#
#    @test learner.model[:best_learner_index] == 2
#  end

#  testset "BestLearner conducts grid search" begin
#    learner = BestLearner(Dict(
#      :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()],
#      :learner_options_grid => [
#       Dict( 
#          :impl_options => Dict(
#            :purity_threshold => [0.5, 1.0]
#           ) 
#         ),
#        Dict(),
#        Dict( 
#          :impl_options => Dict(
#            :num_trees => [5, 10, 20], 
#            :partial_sampling => [0.5, 0.7]
#           )
#         ) 
#      ]
#     ))
#    fit!(learner, nfcp.train_instances, nfcp.train_labels)
#
#    @test length(learner.model[:learners]) == 8
#  end
end

end # module
