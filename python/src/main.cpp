#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <iostream>
#include <numeric>
#include <string>
#include <sstream>


// our headers
#include "opengm_python/opengm_python.hpp"
#include "opengm_python/opengm_python_config.hpp"

namespace py = pybind11;



namespace opengm {

    // implementation in def_myclass.cpp
    void def_build_config(py::module & m);

    // implementation in def_space.cpp
    void def_space(py::module & m);

    // implementation in def_tensor.cpp
    void def_tensor(py::module & m);

    // implementation in def_gm.cpp
    void def_gm(py::module & m);

    // implementation in def_minimizer.cpp
    void def_minimizer(py::module & m);

    // implementation in def_minimizers.cpp
    void def_minimizers(py::module & m);
}


// Python Module and Docstrings
PYBIND11_MODULE(_opengm , module)
{
    xt::import_numpy();

    module.doc() = R"pbdoc(
        _opengm  python bindings

        .. currentmodule:: _opengm 

        .. autosummary::
           :toctree: _generate

           BuildConfiguration
           MyClass
    )pbdoc";

    opengm::def_build_config(module);
    opengm::def_space(module);
    opengm::def_tensor(module);
    opengm::def_gm(module);
    opengm::def_minimizer(module);
    opengm::def_minimizers(module);
    // make version string
    std::stringstream ss;
    ss<<OPENGM_PYTHON_VERSION_MAJOR<<"."
      <<OPENGM_PYTHON_VERSION_MINOR<<"."
      <<OPENGM_PYTHON_VERSION_PATCH;
    module.attr("__version__") = ss.str().c_str();
}