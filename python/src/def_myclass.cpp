#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <iostream>
#include <numeric>


// our headers
#include "opengm_python/opengm_python.hpp"

namespace py = pybind11;



namespace opengm {

    void def_class(py::module & m)
    {
        py::class_<MyClass>(m, "MyClass")
            .def(py::init<uint64_t>(),py::arg("size"))
            .def("hello_world", &MyClass::hello_world)
        ;
    }

}
