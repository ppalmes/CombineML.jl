module TestCaretWrapper
using Random
using DataFrames

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
nfcp = NumericFeatureClassification()

using Test

using MLBase
using CombineML.Types
import CombineML.Types.fit!
import CombineML.Types.transform!
using CombineML.Transformers.CaretWrapper
CW = CaretWrapper

using RCall
R"library(caret)"
R"library(randomForest)"
R"library(e1071)"
R"library(gam)"
#R"library(nnet)"
#R"library(kernlab)"
#R"library(grid)"



function behavior_check(caret_learner::String, impl_options=Dict())
  # Predict with CombineML learner
  Random.seed!(1)
  R"set.seed(1)"
  learner = CRTLearner(Dict(
                        :learner => caret_learner, 
                        :fitControl=>"trainControl(method='cv')",
                        :impl_options => impl_options
                       ))
  combineml_predictions = fit_and_transform!(learner, nfcp)


  # Predict with backend learner
  Random.seed!(1)
  R"set.seed(1)"
  xtr=nfcp.train_instances |> DataFrame
  xts=nfcp.test_instances |> DataFrame
  yy =  nfcp.train_labels
  fcontrol=R"trainControl(method='none')"
  mdl = rcall(:train,xtr,yy,method=caret_learner,trControl=fcontrol)
  original_predictions = rcopy(rcall(:predict,mdl,xts)) # extract robject

  ## Verify same predictions
  @test combineml_predictions == original_predictions
end

@testset "CARET learners" begin
  @testset "CRTLearner gives same results as its backend" begin
    caret_learners = ["rf"] #,"svmLinear", "nnet", "earth"]
    for caret_learner in caret_learners
      behavior_check(caret_learner)
    end
  end
  @testset "CRTLearner with options gives same results as its backend" begin
    behavior_check("rf", Dict(:ntree=>200))
  end
#
#  @testset "CRTLearner throws on incompatible feature" begin
#    instances = [
#                 1 "a";
#                 2 3;
#                ]
#    labels = [
#              "a";
#              "b";
#             ]
#
#    learner = CRTLearner(Dict(:learner=>"rf"))
#    @fact_throws fit!(learner, instances, labels)
#  end
end

end # module
