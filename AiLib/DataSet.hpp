
#pragma once
#include <vector>
#include <python3.10/Python.h>
#include <sstream>
#include <string>
#include <iostream>

using namespace std;
class DataSet {
	public:
		vector<vector<double>> inputs;
		vector<vector<double>> outputs;
		int row1;
		int row2;
		int col1;
		int col2;
		DataSet(vector<vector<double>>& inputs,vector<vector<double>>& outputs,int row1,int col1,int row2, int col2);
		DataSet& operator=(const DataSet& other);
		~DataSet();

};