
#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <python3.10/Python.h>
using namespace std;
class DataSet {
	public:
		vector<vector<float>> inputs;
		vector<vector<float>> outputs;
		DataSet(vector<vector<float>> inputs, vector<vector<float>> outputs);
		~DataSet();

};

void init_my_module_DataSet(pybind11::module& m);