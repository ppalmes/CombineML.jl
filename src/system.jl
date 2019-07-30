# System module.
module System


export LIB_SKL_AVAILABLE,
       LIB_CRT_AVAILABLE

using RCall
using Conda

import PyCall: pyimport_conda, pycall

function check_py_dep()
  is_available = true
  try
    pyimport_conda("sklearn", "scikit-learn")
  catch
    @info "scikitlearn not available"
    is_available = false
  end
  return is_available
end

function check_r_dep()
  is_available = true
  try
    R"library(caret)"
  catch
    @info "caret not available"
    is_available = false
  end
  return is_available
end

## Check system for python dependencies.
LIB_SKL_AVAILABLE = check_py_dep()
LIB_CRT_AVAILABLE = check_r_dep()

end # module
