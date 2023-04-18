#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <python3.10/Python.h>
using namespace std;
enum LossFType
{
	BCELoss = 1,
	CELoss = 2,
	L1Loss = 3,
	L2Loss = 4
};

enum LayerF
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
	LossFType getBCELoss() {
		return LossFType(BCELoss);
	}
	LossFType getCELoss() {
		return LossFType(CELoss);
	}
	LossFType getL1Loss() {
		return LossFType(L1Loss);
	}
	LossFType getL2Loss() {
		return LossFType(L2Loss);
	}
	LayerF getTanh() {
		return LayerF(Tanh);
	}
	LayerF getSigmoid() {
		return LayerF(Sigmoid);
	}
	LayerF getRelu() {
		return LayerF(Relu);
	}
	RegFType getRegNone() {
		return RegFType(None);
	}
	RegFType getRegL1() {
		return RegFType(L1);
	}
	RegFType getRegL2() {
		return RegFType(L2);
	}
};

vector<vector<float&>> numpyToVector2D(pybind11::array_t<float> input_array){
	pybind11::buffer_info buf_info = input_array.request();

	if(buf_info.ndim != 2){
		throw runtime_error("Input array must be 2D");
	}
	size_t rows = buf_info.shape[0];
	size_t cols = buf_info.shape[1];

	vector<vector<float&>> outputs(rows,vector<float&>(cols));
	auto data_ptr = static_cast<float*>(buf_info.ptr);

	for(size_t i = 0;i<rows;i++){
		for(size_t j = 0;j<cols;j++){
			outputs[i][j] = *data_ptr++;
		}
	}

	return outputs;
}

void init_my_module_Enums(pybind11::module& m){
	pybind11::class_<EnumValue>(m,"FTypes")
		.def(pybind11::init<>())
		.def("getBCELoss",&EnumValue::getBCELoss)
		.def("getCELoss",&EnumValue::getCELoss)
		.def("getL1Loss",&EnumValue::getL1Loss)
		.def("getL2Loss",&EnumValue::getL2Loss)
		.def("getTanh",&EnumValue::getTanh)
		.def("getSigmoid",&EnumValue::getSigmoid)
		.def("getRelu",&EnumValue::getRelu)
		.def("getRegNone",&EnumValue::getRegNone)
		.def("getRegL1",&EnumValue::getRegL1)
		.def("getRegL2",&EnumValue::getRegL2);
	pybind11::enum_<RegFType>(m,"RegFType")
		.value("None",RegFType::None)
		.value("L1",RegFType::L1)
		.value("L2",RegFType::L2)
		.export_values();
	pybind11::enum_<LayerF>(m,"LayerF")
		.value("Tanh",LayerF::Tanh)
		.value("Sigmoid",LayerF::Sigmoid)
		.value("Relu",LayerF::Relu)
		.export_values();
	pybind11::enum_<LossFType>(m,"LossFType")
		.value("BCELoss",LossFType::BCELoss)
		.value("CELoss",LossFType::CELoss)
		.value("L1Loss",LossFType::L1Loss)
		.value("L2Loss",LossFType::L2Loss)
		.export_values();
	m.def("numpyToVector2D",&numpyToVector2D,"Convert 2D numpy to 2D vector,variable type must be float");
}