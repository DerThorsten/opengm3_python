#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

#include <iostream>
#include <numeric>
#include <memory>

#include <xtensor-python/pytensor.hpp>

// our headers
#include "opengm_python/opengm_python.hpp"
#include "opengm_python/opengm_python_config.hpp"
#include "opengm/minimizer/utils/label_fuser.hpp"

#include <opengm/space.hpp>
#include <opengm/graphical_model.hpp>
#include <opengm/minimizer/minimizer_base.hpp>


namespace py = pybind11;



namespace opengm {



    // defines:
    // - minimizer base
    // - minimizer factory base
    // - minimizer callback base
    template<class gm_type>
    void def_minimizer_bases(py::module & m, const std::string gm_name)
    {
        using callback_base_type = MinimizerCallbackBase<gm_type>;
        {
            auto cls_name = std::string("MinimizerCallbackBase")+gm_name;
            py::class_<callback_base_type>(m, cls_name.c_str());
        }
        {
            using minimizer_base_type = MinimizerBase<gm_type>;
            using labels_vector_type = typename  minimizer_base_type::labels_vector_type;

            auto cls_name = std::string("MinimizerBase")+gm_name;
            py::class_<minimizer_base_type>(m, cls_name.c_str())
                .def("can_start_from_starting_point", &minimizer_base_type::can_start_from_starting_point)
                .def("current_energy", &minimizer_base_type::current_energy)
                .def("best_energy", &minimizer_base_type::best_energy)
                .def("best_labels", [](minimizer_base_type * minimizer){
                    const auto & best_labels = minimizer->best_labels();
                    auto numpy_labels = xt::pytensor<label_type, 1>::from_shape({int(best_labels.size())});
                    std::copy(best_labels.begin(), best_labels.end(), numpy_labels.begin());
                    return numpy_labels;
                })
                .def("current_labels", [](minimizer_base_type * minimizer){
                    const auto & current_labels = minimizer->current_labels();
                    auto numpy_lables = xt::pytensor<label_type, 1>::from_shape({int(current_labels.size())});
                    std::copy(current_labels.begin(), current_labels.end(), numpy_lables.begin());
                    return numpy_lables;
                })
                .def("set_starting_point", [](minimizer_base_type * minimizer, xt::pytensor<label_type, 1> & numpy_labels){
                    if(numpy_labels.size() != minimizer->gm().num_variables())
                    {
                        throw std::runtime_error("labels.shape[0] != gm.num_variables()");
                    }
                    // TODO this can be speed up when the data ptr is used in case the array in contigous
                    labels_vector_type labels(numpy_labels.begin(), numpy_labels.end());
                    minimizer->set_starting_point(labels);
                })
                .def("minimize",[](minimizer_base_type * minimizer, callback_base_type * callback){
                    minimizer->minimize(callback);
                })

            ;
        }
        {
            using minimizer_factory_base_type = MinimizerFactoryBase<gm_type>;
            auto cls_name = std::string("MinimizerFactoryBase")+gm_name;
            py::class_<minimizer_factory_base_type, std::shared_ptr<minimizer_factory_base_type>>(m, cls_name.c_str())
                .def("create", &minimizer_factory_base_type::create,
                    py::keep_alive<0, 1>(),
                    py::arg("gm"))
            ;
        }

    }


    // defines:
    // - label fuser settings
    // - label fuser
    template<class gm_type>
    void def_label_fuser(py::module & m, const std::string & gm_name)
    {
        using label_fuser_type = LabelFuser<gm_type>;
        using settings_type = typename label_fuser_type::settings_type;

        auto settings_cls_name = std::string("LabelFuserSettings")+gm_name;
        py::class_<settings_type>(m, settings_cls_name.c_str())
            .def_readwrite("minimizer_factory", &settings_type::minimizer_factory)
            .def_readwrite("eps", &settings_type::eps)
        ;

        auto cls_name = std::string("LabelFuser")+gm_name;
        py::class_<label_fuser_type>(m, cls_name.c_str())
        ;
    }


    template<class callback_type>
    auto def_concret_callback(py::module & m, const std::string & callback_name, const std::string & gm_name)
    {
        using gm_type = typename callback_type::gm_type;
        using callback_base_type = MinimizerCallbackBase<gm_type>;
        auto cls_name = callback_name+gm_name;
        return py::class_<callback_type, callback_base_type>(m, cls_name.c_str());
    }



    // defines all callbacks
    template<class gm_type>
    void def_callacks(py::module & m, const std::string gm_name)
    {
        def_concret_callback<VerboseMinimizerCallback<gm_type>>(m, "VerboseCallback", gm_name)
            .def(py::init<std::size_t>(), py::arg("visit_nth")=1)
        ;
    }

    // defines all inference related methods for a given gm_type
    template<class gm_type>
    void def_minimizer_for_gm(py::module & m, const std::string gm_name)
    {
        def_minimizer_bases<gm_type>(m, gm_name);
        def_label_fuser<gm_type>(m, gm_name);
        def_callacks<gm_type>(m, gm_name);
    }

    // defines all inference related methods for all relevant graphical model types
    void def_minimizer(py::module & m)
    {

        def_minimizer_for_gm<GraphicalModel<UniformSpace<label_type>, py_value_type>>(
            m,"UniformSpaceGraphicalModel");

        def_minimizer_for_gm<GraphicalModel<StaticNumLabelsSpace<label_type, 2>, py_value_type>>(
            m,"BinarySpaceGraphicalModel");

        def_minimizer_for_gm<GraphicalModel<ExplicitSpace<label_type>, py_value_type>>(
            m,"ExplicitSpaceGraphicalModel");

    }

}
