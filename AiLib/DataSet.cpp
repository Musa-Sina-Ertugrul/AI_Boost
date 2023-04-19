
#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include "DataSet.hpp"
#include <python3.10/Python.h>
using namespace std;

DataSet::DataSet(vector<vector<float>> inputs, vector<vector<float>> outputs) {
			this->inputs = inputs;
			this->outputs = outputs;
}

DataSet::~DataSet() {
			for (int i = 0; i < this->inputs.size(); i++) {
				this->inputs[i].clear();
			}
			this->inputs.clear();
			for (int i = 0; i < this->outputs.size(); i++) {
				this->outputs[i].clear();
			}
			this->outputs.clear();
		}

void init_my_module_DataSet(pybind11::module& m){
	pybind11::class_<DataSet>(m,"DataSet")
		.def(pybind11::init<vector<vector<float>>, vector<vector<float>>>());
}