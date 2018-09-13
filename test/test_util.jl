module TestUtil

using Test

#using FactCheck

using CombineML.Util

include("fixture_learners.jl")
using .FixtureLearners
nfcp = NumericFeatureClassification()

@testset "CombineML util functions" begin

  @testset "holdout returns proportional partitions" begin
    n = 10
    right_prop = 0.3
    (left, right) = holdout(n, right_prop)
    @test size(left, 1) == n - (n * right_prop)
    @test size(right, 1) == n * right_prop
    @test intersect(left, right) |> isempty
    @test size(union(left, right), 1) == n
  end

  @testset "kfold returns k partitions" begin
    num_instances = 10
    num_partitions = 3
    partitions = kfold(num_instances, num_partitions)
    @test size(partitions, 1) == num_partitions
    [@test length(partition) >= 6 for partition in partitions]
  end

#  context("score calculates accuracy") do
#    learner = PerfectScoreLearner(Dict(:problem => nfcp))
#    predictions = fit_and_transform!(learner, nfcp)
#
#    @fact score(
#      :accuracy, nfcp.test_labels, predictions
#    ) --> 100.0
#  end

#  context("score throws exception on unknown metric") do
#    learner = PerfectScoreLearner(Dict(:problem => nfcp))
#    predictions = fit_and_transform!(learner, nfcp)
#
#    @fact_throws score(
#      :fake, nfcp.test_labels, predictions
#    )
#  end

  @testset "infer_eltype returns inferred elements type" begin
    vector = [1,2,3,"a"]
    @test infer_eltype(vector[1:3]) == Int
  end

  @testset "nested_dict_to_tuples produces list of tuples" begin
    nested_dict = Dict(
      :a => [1,2],
      :b => Dict(
        :c => [3,4,5]
       )
     )
    expected_set = Set([
      ([:a], [1,2]),
      ([:b,:c], [3,4,5])
     ])
    set = nested_dict_to_tuples(nested_dict)
    @test set == expected_set
  end

  @testset "nested_dict_set! assigns values" begin
    nested_dict = Dict(
      :a => 1,
      :b => Dict(
        :c => 2
       )
     )
    expected_dict = Dict(
      :a => 1,
      :b => Dict(
        :c => 3
       )
     )
    nested_dict_set!(nested_dict, [:b,:c], 3)
    @test nested_dict == expected_dict
  end

  @testset "nested_dict_merge merges two nested dictionaries" begin
    first = Dict(
      :a => 1,
      :b => Dict(
        :c => 2,
        :d => 3
       )
     )
    second = Dict(
      :a => 4,
      :b => Dict(
        :d => 5
       )
     )
    expected = Dict(
      :a => 4,
      :b => Dict(
        :c => 2,
        :d => 5
      )
     )
    actual = nested_dict_merge(first, second)
    @test actual == expected
  end

  @testset "create_transformer produces new transformer" begin
    learner = AlwaysSameLabelLearner(Dict(:label => :a))
    new_options = Dict(:label => :b)
    new_learner = create_transformer(learner, new_options)
    @test learner.options[:label] == :a
    @test new_learner.options[:label] == :b
    @test true == !isequal(learner, new_learner)
  end

end

end # module
