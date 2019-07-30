# Wrapper to CARET library.
module CaretWrapper

export caretrun
using RDatasets
using DataFrames

using CombineML.Types
import CombineML.Types.fit!
import CombineML.Types.transform!
using CombineML.Util

using RCall
R"library(caret)"
R"library(e1071)"
R"library(gam)"
R"library(randomForest)"
R"library(nnet)"
R"library(kernlab)"
R"library(grid)"
R"library(MASS)"
R"library(pls)"
#R"library(xgboost)"


export CRTLearner,
       fit!,
       transform!


# CARET wrapper that provides access to all learners.
# 
# Options for the specific CARET learner is to be passed
# in `options[:impl_options]` dictionary.
mutable struct CRTLearner <: Learner
  model
  options
  
  function CRTLearner(options=Dict())
    #fitControl="trainControl(method = 'cv',number = 5,repeats = 5)"
    fitControl="trainControl(method = 'none')"
    default_options = Dict(
      # Output to train against
      # (:class).
      :output => :class,
      :learner => "rf",
      :fitControl => fitControl,
      :impl_options => Dict()
    )
    new(nothing, nested_dict_merge(default_options, options)) 
  end
end

function fit!(crt::CRTLearner,x::T,y::Vector) where  {T<:Union{Vector,DataFrame,Matrix}}
    xx = x |> DataFrame
    yy = y |> Vector
    rres = rcall(:train,xx,yy,method=crt.options[:learner],trControl = reval(crt.options[:fitControl]))
    #crt.model = R"$rres$finalModel"
    crt.model = rres
end

function transform!(crt::CRTLearner,x::T) where  {T<:Union{Vector,Matrix,DataFrame}}
    xx = x |> DataFrame
    res = rcall(:predict,crt.model,xx)
    return rcopy(res)
end

function caretrun()
    crt = CRTLearner(Dict(:learner=>"rf",:fitControl=>"trainControl(method='cv')"))
    iris=dataset("datasets","iris")
    x=iris[:,1:4]  |> Matrix
    y=iris[:,5] |> Vector
    fit!(crt,x,y)
    print(crt.model)
    transform!(crt,x)
end


end # module
