#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <iostream>
#include <numeric>


// our headers
#include "opengm_python/opengm_python.hpp"

#include <opengm/opengm_config.hpp>
#include <opengm/space.hpp>

namespace py = pybind11;



namespace opengm {

    template<class space_type>
    auto def_space(
        py::module & m,
        const std::string & space_name
    )
    {
        auto py_cls = py::class_<space_type>(m, space_name.c_str())
            //.def(py::init<uint64_t>(),py::arg("size"))
            .def("__len__", &space_type::size)
            .def("__getitem__", &space_type::operator[],py::arg("vi"))
            .def("max_num_labels", &space_type::max_num_labels)
            .def("is_uniform_space", &space_type::is_uniform_space)
            .def("name", [space_name](const space_type & ){return space_name;})
        ;
        return py_cls;
    }

    void def_space(py::module & m)
    {
        def_space<UniformSpace<label_type>>(m, "UniformSpace")
            .def(py::init<std::size_t, label_type>(),
                py::arg("num_variables"),
                py::arg("num_labels")
            )
        ;

        def_space<StaticNumLabelsSpace<label_type, 2>>(m, "BinarySpace")
            .def(py::init<std::size_t>(),
                py::arg("num_variables")
            )
        ;

        def_space<ExplicitSpace<label_type>>(m, "ExplicitSpace")
            .def(py::init<std::size_t, label_type>(),
                py::arg("num_variables"),
                py::arg("num_labels")
            )
        ;
    }

}
