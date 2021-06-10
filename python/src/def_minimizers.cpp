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

#include <opengm/minimizer/bp.hpp>
#include <opengm/minimizer/icm.hpp>
#include <opengm/minimizer/factor_icm.hpp>
#include <opengm/minimizer/self_fusion.hpp>
#include <opengm/minimizer/chained_minimizers.hpp>

namespace py = pybind11;



namespace opengm {




    // defines:
    // - settings of minimizer
    // - minimizer
    // - minimizer factory
    template<class minimizer_type>
    auto def_concret_minimizer(py::module & m, const std::string minimizer_name, const std::string minimizer_fname, const std::string & gm_name)
    {
        using gm_type = typename minimizer_type::gm_type;
        using settings_type = typename minimizer_type::settings_type;
        using minimizer_base_type = MinimizerBase<gm_type>;
        auto cls_name = minimizer_name + gm_name;

        // settings type
        auto settings_cls_name = minimizer_name + std::string("Settings") + gm_name;
        auto py_minimizer_settings_cls =  py::class_<settings_type>(m, settings_cls_name.c_str());
        py_minimizer_settings_cls
            .def(py::init<>())
        ;

        // minimizer itself
        py::class_<minimizer_type, minimizer_base_type>(m, cls_name.c_str())
            .def(py::init<const gm_type &, const  settings_type &>(),
                py::keep_alive<1, 2>(),
                py::arg("gm"),
                py::arg("settings") = settings_type())
        ;

        // factory type
        using minimizer_factory_base_type = MinimizerFactoryBase<gm_type>;
        using minimizer_factory =  MinimizerFactory<minimizer_type>;
        auto factory_cls_name = minimizer_name + std::string("Factory") + gm_name;
        auto py_minimizer_factory_cls = py::class_<minimizer_factory, minimizer_factory_base_type, std::shared_ptr<minimizer_factory>>(m, factory_cls_name.c_str())
            .def(py::init<const  settings_type &>(),
                py::arg("settings") = settings_type())
        ;

        return py_minimizer_settings_cls;
    }



    // defines all inference related methods for a given gm_type
    template<class gm_type>
    void def_minimizer(py::module & m, const std::string gm_name)
    {
        // bp
        {
            using minimizer_type = BeliefPropergation<gm_type>;
            using settings_type = typename minimizer_type::settings_type;
            def_concret_minimizer<minimizer_type>(m, "BeliefPropergation","belief_propergation", gm_name)
                .def_readwrite("damping", &settings_type::damping)
                .def_readwrite("num_iterations", &settings_type::num_iterations)
                .def_readwrite("convergence", &settings_type::convergence)
            ;
        }
        // icm
        {
            using minimizer_type = Icm<gm_type>;
            using settings_type = typename minimizer_type::settings_type;
            def_concret_minimizer<minimizer_type>(m, "Icm","icm", gm_name)
            ;
        }
        // factory icm
        {
            using minimizer_type = FactorIcm<gm_type>;
            using settings_type = typename minimizer_type::settings_type;
            def_concret_minimizer<minimizer_type>(m, "FactorIcm","factor_icm", gm_name)
            ;
        }
        // self fusion
        {
            using minimizer_type = SelfFusion<gm_type>;
            using settings_type = typename minimizer_type::settings_type;
            def_concret_minimizer<minimizer_type>(m, "SelfFusion","self_fusion", gm_name)
                .def_readwrite("minimizer_factory", &settings_type::minimizer_factory)
                .def_readwrite("fuse_minimizer_factory", &settings_type::fuse_minimizer_factory)
            ;
        }
        // chained minimizers
        {
            using minimizer_type = ChainedMinimizers<gm_type>;
            using settings_type = typename minimizer_type::settings_type;
            def_concret_minimizer<minimizer_type>(m, "ChainedMinimizers","chained_minimizers", gm_name)
                .def_readwrite("minimizer_factories", &settings_type::minimizer_factories)
            ;
        }
    }

    // defines all inference related methods for all relevant graphical model types
    void def_minimizers(py::module & m)
    {

        def_minimizer<GraphicalModel<UniformSpace<label_type>, py_value_type>>(
            m,"UniformSpaceGraphicalModel");

        def_minimizer<GraphicalModel<StaticNumLabelsSpace<label_type, 2>, py_value_type>>(
            m,"BinarySpaceGraphicalModel");

        def_minimizer<GraphicalModel<ExplicitSpace<label_type>, py_value_type>>(
            m,"ExplicitSpaceGraphicalModel");

    }

}
