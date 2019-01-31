# Wrapper module for scikit-learn machine learners.
module ScikitLearnWrapper

export skkrun
using RDatasets

using CombineML.Types
import CombineML.Types.fit!
import CombineML.Types.transform!
using CombineML.Util

using PyCall

@pyimport sklearn.ensemble as ENS
@pyimport sklearn.linear_model as LM
@pyimport sklearn.discriminant_analysis as DA
@pyimport sklearn.neighbors as NN
@pyimport sklearn.svm as SVM
@pyimport sklearn.tree as TREE
@pyimport sklearn.neural_network as ANN
@pyimport sklearn.gaussian_process as GP
@pyimport sklearn.kernel_ridge as KR
@pyimport sklearn.naive_bayes as NB
@pyimport sklearn.isotonic as ISO


export SKLLearner,
       fit!,
       transform!

# Available scikit-learn learners.
learner_dict = Dict(
  "AdaBoostClassifier" => ENS.AdaBoostClassifier,
  "BaggingClassifier" => ENS.BaggingClassifier,
  "ExtraTreesClassifier" => ENS.ExtraTreesClassifier,
  "GradientBoostingClassifier" => ENS.GradientBoostingClassifier,
  "RandomForestClassifier" => ENS.RandomForestClassifier,
  "LDA" => DA.LinearDiscriminantAnalysis,
  "QDA" => DA.QuadraticDiscriminantAnalysis,
  "PassiveAggressiveClassifier" => LM.PassiveAggressiveClassifier,
  "RidgeClassifier" => LM.RidgeClassifier,
  "RidgeClassifierCV" => LM.RidgeClassifierCV,
  "SGDClassifier" => LM.SGDClassifier,
  "KNeighborsClassifier" => NN.KNeighborsClassifier,
  "RadiusNeighborsClassifier" => NN.RadiusNeighborsClassifier,
  "NearestCentroid" => NN.NearestCentroid,
  "SVC" => SVM.SVC,
  "LinearSVC" => SVM.LinearSVC,
  "NuSVC" => SVM.NuSVC,
  "MLPClassifier" => ANN.MLPClassifier,
  "GaussianProcessClassifier" => GP.GaussianProcessClassifier,
  "DecisionTreeClassifier" => TREE.DecisionTreeClassifier,
  #"VotingClassifier" => ENS.VotingClassifier,
  "GaussianNB" => NB.GaussianNB,
  "MultinomialNB" => NB.MultinomialNB,
  "ComplementNB" => NB.ComplementNB,
  "BernoulliNB" => NB.BernoulliNB,
  "SVR" => SVM.SVR,
  "Ridge" => LM.Ridge,
  "RidgeCV" => LM.RidgeCV,
  "Lasso" => LM.Lasso,
  "ElasticNet" => LM.ElasticNet,
  "Lars" => LM.Lars,
  "LassoLars" => LM.LassoLars,
  "OrthogonalMatchingPursuit" => LM.OrthogonalMatchingPursuit,
  "BayesianRidge" => LM.BayesianRidge,
  "LogisticRegression" => LM.LogisticRegression,
  "ARDRegression" => LM.ARDRegression,
  "SGDRegressor" => LM.SGDRegressor,
  "PassiveAggressiveRegressor" => LM.PassiveAggressiveRegressor,
  "KernelRidge" => KR.KernelRidge,
  "KNeighborsRegressor" => NN.KNeighborsRegressor,
  "RadiusNeighborsRegressor" => NN.RadiusNeighborsRegressor,
  "GaussianProcessRegressor" => GP.GaussianProcessRegressor,
  "DecisionTreeRegressor" => TREE.DecisionTreeRegressor,
  "RandomForestRegressor" => ENS.RandomForestRegressor,
  "ExtraTreesRegressor" => ENS.ExtraTreesRegressor,
  "AdaBoostRegressor" => ENS.AdaBoostRegressor,
  "GradientBoostingRegressor" => ENS.GradientBoostingRegressor,
  "IsotonicRegression" => ISO.IsotonicRegression,
  "MLPRegressor" => ANN.MLPRegressor
)


# Wrapper for scikit-learn that provides access to most learners.
# 
# Options for the specific scikit-learn learner is to be passed
# in `options[:impl_options]` dictionary.
# 
# Available learners:
#
#   - "AdaBoostClassifier"
#   - "BaggingClassifier"
#   - "ExtraTreesClassifier"
#   - "GradientBoostingClassifier"
#   - "RandomForestClassifier"
#   - "LDA"
#   - "LogisticRegression"
#   - "PassiveAggressiveClassifier"
#   - "RidgeClassifier"
#   - "RidgeClassifierCV"
#   - "SGDClassifier"
#   - "KNeighborsClassifier"
#   - "RadiusNeighborsClassifier"
#   - "NearestCentroid"
#   - "QDA"
#   - "SVC"
#   - "LinearSVC"
#   - "NuSVC"
#   - "DecisionTreeClassifier"
#
mutable struct SKLLearner <: Learner
  model
  options
  
  function SKLLearner(options=Dict())
    default_options = Dict(
      # Output to train against
      # (:class).
      :output => :class,
      :learner => "LinearSVC",
      # Options specific to this implementation.
      :impl_options => Dict()
    )
    new(nothing, nested_dict_merge(default_options, options)) 
  end
end

function fit!(sklw::SKLLearner, instances::T, labels::Vector) where {T <: Union{Matrix,Vector}}
  impl_options = copy(sklw.options[:impl_options])
  learner = sklw.options[:learner]
  py_learner = learner_dict[learner]

  # Assign CombineML-specific defaults if required
  if learner == "RadiusNeighborsClassifier"
    if get(impl_options, :outlier_label, nothing) == nothing
      impl_options[:outlier_label] = labels[rand(1:size(labels, 1))]
    end
  end

  # Train
  sklw.model = py_learner(;impl_options...)
  sklw.model[:fit](instances, labels)
end

function transform!(sklw::SKLLearner, instances::T) where {T <: Union{Matrix,Vector}}
  return collect(sklw.model[:predict](instances))
end

function skkrun()
    iris=dataset("datasets","iris")
    instances=iris[:,1:4] |> Matrix
    labels=iris[:,5] |> Vector
    model1 = SKLLearner(Dict(:learner=>"LinearSVC",:impl_args=>Dict(:max_iter=>5000)))
    model2 = SKLLearner(Dict(:learner=>"QDA"))
    model3 = SKLLearner(Dict(:learner=>"MLPClassifier"))
    model = SKLLearner(Dict(:learner=>"BernoulliNB"))
    fit!(model,instances,labels)
    println(sum(transform!(model,instances).==labels)/length(labels)*100)

    x=iris[:,1:3] |> Matrix
    y=iris[:,4] |> Vector
    #regmodel = SKLLearner(Dict(:learner => "SVR",:impl_args=>Dict(:gamma=>"scale")))
    #regmodel = SKLLearner(Dict(:learner => "RidgeCV"))
    regmodel = SKLLearner(Dict(:learner => "GradientBoostingRegressor"))
    #regmodel = SKLLearner(Dict(:learner => "MLPRegressor"))
    fit!(regmodel,x,y)
    println(sum(transform!(regmodel,x).-y)/length(labels)*100)
    xx=iris[:,1] |> Vector
    regmodel = SKLLearner(Dict(:learner => "IsotonicRegression"))
    fit!(regmodel,xx,y)
    println(sum(transform!(regmodel,xx).-y)/length(labels)*100)
end



end # module
