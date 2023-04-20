#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <pybind11/stl.h>
#include <python3.10/Python.h>
namespace py = pybind11;
using namespace std;
enum LossFType
{
	BCELoss = 1,
	CELoss = 2,
	L1Loss = 3,
	L2Loss = 4
};

enum LayerFunction
{
	Tanh = 5,
	Sigmoid = 6,
	Relu = 7
};

enum RegFType
{
	None = 8,
	L1 = 9,
	L2 = 10
};

struct EnumValue
{
	LossFType getBCELoss();
	LossFType getCELoss();
	LossFType getL1Loss();
	LossFType getL2Loss();
	LayerFunction getTanh();
	LayerFunction getSigmoid();
	LayerFunction getRelu();
	RegFType getRegNone();
	RegFType getRegL1();
	RegFType getRegL2();
};

vector<vector<float>> numpytoVector2D(pybind11::array_t<float> input_array);
void init_my_module_Enumss(py::module& m);