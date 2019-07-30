# Wrapper module for scikit-learn machine learners.
module ScikitLearnWrapper

export skkrun
using DataFrames

using CombineML.Types
import CombineML.Types.fit!
import CombineML.Types.transform!
using CombineML.Util

using PyCall

function initlibs()
  global ENS=pyimport("sklearn.ensemble")
  global LM=pyimport("sklearn.linear_model")
  global DA=pyimport("sklearn.discriminant_analysis")
  global NN=pyimport("sklearn.neighbors")
  global SVM=pyimport("sklearn.svm")
  global TREE=pyimport("sklearn.tree")
  global ANN=pyimport("sklearn.neural_network")
  global GP=pyimport("sklearn.gaussian_process")
  global KR=pyimport("sklearn.kernel_ridge")
  global NB=pyimport("sklearn.naive_bayes")
  global ISO=pyimport("sklearn.isotonic")

 # Available scikit-learn learners.
  global learner_dict = Dict(
       "AdaBoostClassifier" => ENS.AdaBoostClassifier,
       "BaggingClassifier" => ENS.BaggingClassifier,
       "ExtraTreesClassifier" => ENS.ExtraTreesClassifier,
       "VotingClassifier" => ENS.VotingClassifier,
       "GradientBoostingClassifier" => ENS.GradientBoostingClassifier,
       "RandomForestClassifier" => ENS.RandomForestClassifier,
       "LDA" => DA.LinearDiscriminantAnalysis,
       "QDA" => DA.QuadraticDiscriminantAnalysis,
       "LogisticRegression" => LM.LogisticRegression,
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
end
export SKLLearner,
       fit!,
       transform!

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
    initlibs()
    new(nothing, nested_dict_merge(default_options, options)) 
  end
end

function fit!(sklw::SKLLearner, xinstances::T, labels::Vector) where {T <: Union{Vector,Matrix,DataFrame}}
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
  features = convert(Matrix,xinstances)
  sklw.model.fit(features, labels)
end

function transform!(sklw::SKLLearner, xinstances::T) where {T <: Union{Vector,Matrix,DataFrame}}
  features = convert(Matrix,xinstances)
  return collect(sklw.model.predict(features))
end

end # module
