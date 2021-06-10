#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <iostream>
#include <numeric>


// our headers
#include "opengm_python/opengm_python.hpp"
#include "opengm_python/opengm_python_config.hpp"

#include <opengm/opengm_config.hpp>
#include <opengm/tensors.hpp>



namespace py = pybind11;



namespace opengm {




    template<class tensor_type>
    auto def_tensor(
        py::module & m,
        const std::string & tensor_name
    )
    {
        using base_type = TensorBase<py_value_type>;

        auto py_cls = py::class_<tensor_type, base_type>(m, tensor_name.c_str())
        ;
        return py_cls;
    }

    void def_tensor(py::module & m)
    {

        // export base class
        using base_type = TensorBase<py_value_type>;
        py::class_<base_type>(m, "TensorBase")
            .def_property_readonly("arity", &base_type::arity)
            .def_property_readonly("dimension", &base_type::arity)
            //.def("arity", base_type::arity)
            .def("__getitem__", [](base_type * tensor, py::tuple tuple){
                if(tensor->arity() != tuple.size())
                {
                    throw std::out_of_range("__getitem__ called with wrong number of arguments");
                }
                arity_vector<label_type> labels;
                for(auto t : tuple){
                    auto l = t.cast<label_type>();
                    labels.push_back(l);
                }
                return tensor->at(labels.data(), labels.data() + tensor->arity());
            })
            .def("__getitem__", [](base_type * tensor, label_type l){
                if(tensor->arity() != 1)
                {
                    throw std::out_of_range("__getitem__ called with wrong number of arguments");
                }
                return tensor->at(&l, &l+1);
            })
            .def_property_readonly("shape", [](base_type * tensor){
                py::list list;
                for(auto s : tensor->shape()){
                    list.append(py::cast(s));
                }
                return list;
            })
        ;




        // concrete classes
        def_tensor<Potts2Tensor<py_value_type>>(m, "Potts2Tensor")
            .def(py::init<std::size_t, py_value_type>(),
                py::arg("num_labels"),
                py::arg("beta")
            )
        ;

        def_tensor<UnaryTensor<py_value_type>>(m, "UnaryTensor")
            .def(py::init<label_type, py_value_type>(),
                py::arg("num_labels"),
                py::arg("value") = py_value_type(0.0)
            )
            .def("__setitem__", [](UnaryTensor<py_value_type> * tensor, label_type l, py_value_type v){
                tensor->data()[l] = v;
            })
        ;

        def_tensor<OptimizedBinaryUnary<py_value_type>>(m, "OptimizedBinaryUnary")
            .def(py::init<py_value_type, py_value_type>(),
                py::arg("val0"),
                py::arg("val1") = py_value_type(0.0)
            )
        ;



    }

}
