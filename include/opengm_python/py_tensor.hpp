#pragma once

namespace opengm
{


    class PyAnimal : public Animal {
    public:
        /* Inherit the constructors */
        using Animal::Animal;

        /* Trampoline (need one for each virtual function) */
        std::string go(int n_times) override {
            PYBIND11_OVERRIDE_PURE(
                std::string, /* Return type */
                Animal,      /* Parent class */
                go,          /* Name of function in C++ (must match Python name) */
                n_times      /* Argument(s) */
            );
        }
    };



    template<class T>
    class PyTensor{
    public:
         d
    };
}