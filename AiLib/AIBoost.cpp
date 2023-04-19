#include "Model.hpp"
#include "Enums.hpp"
#include "DataSet.hpp"
#include "Layer.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <python3.10/Python.h>
#include "/usr/include/x86_64-linux-gnu/openblas-pthread/cblas.h"
namespace py = pybind11;
PYBIND11_MODULE(AIBoost,m){
    init_my_module_DataSet(m);
    init_my_module_Enumss(m);
    init_my_module_Layer(m);
    init_my_module_Modell(m);
}