#include <doctest.h>

#include "opengm_python/opengm_python.hpp"
#include "opengm_python/opengm_python_config.hpp"



TEST_SUITE_BEGIN("core");

TEST_CASE("check version"){

    #ifndef OPENGM_PYTHON_VERSION_MAJOR
        #error "OPENGM_PYTHON_VERSION_MAJOR is undefined"
    #endif
    

    #ifndef OPENGM_PYTHON_VERSION_MINOR
        #error "OPENGM_PYTHON_VERSION_MINOR is undefined"
    #endif


    #ifndef OPENGM_PYTHON_VERSION_PATCH
        #error "OPENGM_PYTHON_VERSION_PATCH is undefined"
    #endif

    CHECK_EQ(OPENGM_PYTHON_VERSION_MAJOR , 0);
    CHECK_EQ(OPENGM_PYTHON_VERSION_MINOR , 1);
    CHECK_EQ(OPENGM_PYTHON_VERSION_PATCH , 0);
}



TEST_SUITE_END(); // end of testsuite core
