#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <iostream>
#include <numeric>

#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/pyarray.hpp>

// our headers
#include <opengm/space.hpp>
#include <opengm/graphical_model.hpp>


#include "opengm_python/opengm_python.hpp"
#include "opengm_python/opengm_python_config.hpp"

namespace py = pybind11;



namespace opengm {

    template<class T>
    inline auto iterable_to_arity_vector(py::handle iterable)
    {
        arity_vector<T> vec;
        for(auto element_obj : iterable){
            auto element = element_obj.cast<T>();
            vec.push_back(element);
        }
        return vec;
    }

    template<class gm_type>
    auto def_gm(
        py::module & m,
        py::class_<gm_type> & py_gm_cls,
        const std::string & gm_name
    )
    {
        py_gm_cls
            .def("num_variables", &gm_type::num_variables)
            .def("num_factors", &gm_type::num_factors)
            .def("num_labels", &gm_type::num_labels, py::arg("variable"))
            .def("space", [](gm_type & gm){return gm.space();}, py::return_value_policy::reference_internal)
            .def("max_arity", &gm_type::max_arity)
            .def("arity_upper_bound", &gm_type::arity_upper_bound)
            .def("evaluate", [](gm_type & gm, const xt::pytensor<label_type, 1> & labels){
                return gm.evaluate_at(labels.begin(), labels.end());
            })
            .def("__len__", &gm_type::size)
        ;
        return py_gm_cls;
    }


    template<class space_type>
    auto def_vgm(py::module & m, const std::string space_name){
        using gm_type = GraphicalModel<space_type, py_value_type>;
        using tensor_base_type = typename gm_type::virtual_tensor_base_type;
        const auto gm_name = space_name + std::string("GraphicalModel");
        auto py_gm_cls =  py::class_<gm_type>(m, gm_name.c_str());

        py_gm_cls

            .def("add_tensor",[](gm_type & gm, const xt::pyarray<py_value_type> & numpy_tensor)
            {
                if(numpy_tensor.dimension() == 1)
                {
                    return gm.add_tensor(std::make_unique<UnaryTensor<py_value_type>>(
                        numpy_tensor.begin(),
                        numpy_tensor.end()
                    ));
                }
                else{
                    return gm.add_tensor(std::make_unique<XArrayTensor<py_value_type>>(numpy_tensor));
                }
            },py::arg("tensor"))

            .def("add_factor",[](gm_type & gm, const xt::pyarray<py_value_type> & numpy_tensor, py::handle obj)
            {
                const auto vis = iterable_to_arity_vector<std::size_t>(obj);
                if(numpy_tensor.dimension() == 1)
                {
                    return gm.add_factor(std::make_unique<UnaryTensor<py_value_type>>(
                        numpy_tensor.begin(),
                        numpy_tensor.end()
                    ), vis.begin(), vis.end());
                }
                else{
                    return gm.add_factor(std::make_unique<XArrayTensor<py_value_type>>(numpy_tensor), vis.begin(), vis.end());
                }
            },py::arg("tensor"), py::arg("variables"))



            .def("add_tensor",[](gm_type & gm, tensor_base_type * tensor)
            {
                return gm.add_tensor(tensor->clone());
            },
            py::arg("tensor"))
            .def("add_factor", [](gm_type & gm, std::size_t tid, py::handle obj)
            {
                const auto vis = iterable_to_arity_vector<std::size_t>(obj);
                return gm.add_factor(tid, vis.begin(), vis.end());
            },
            py::arg("ti"), py::arg("variables"))
            .def("add_factor", [](gm_type & gm, tensor_base_type * tensor, py::handle obj)
            {
                const auto vis = iterable_to_arity_vector<std::size_t>(obj);
                return gm.add_factor(tensor->clone(), vis.begin(), vis.end());
            },
            py::arg("tensor"), py::arg("variables"))
        ;

        def_gm<gm_type>(m, py_gm_cls, gm_name);
        return py_gm_cls;
    }

    void def_gm(py::module & m)
    {

        def_vgm<UniformSpace<label_type>>(m, "UniformSpace")
            .def(py::init<std::size_t, label_type>(),
                py::arg("num_variables"),
                py::arg("num_labels")
            )
        ;

        def_vgm<StaticNumLabelsSpace<label_type, 2>>(m, "BinarySpace")
            .def(py::init<std::size_t>(),
                py::arg("num_variables")
            )
        ;

        def_vgm<ExplicitSpace<label_type>>(m, "ExplicitSpace")
            .def(py::init<std::size_t, label_type>(),
                py::arg("num_variables"),
                py::arg("num_labels")
            )
        ;
    }

}
