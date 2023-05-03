
#pragma once
#include <vector>
#include "DataSet.hpp"
#include <python3.10/Python.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

using namespace std;

DataSet::DataSet(vector<vector<double>>& inputs,vector<vector<double>>& outputs,int row1,int col1,int row2, int col2) {
	this->inputs = inputs;

    this->outputs = outputs;
    this->col1 = col1;
    this->col2 = col2;
    this->row1 = row1;
    this->row2 = row2;
}
DataSet& DataSet::operator=(const DataSet& other) {
        if (this == &other) {
            return *this;
        }

        // Assign other members as needed
        // Note: id_ is not assigned since it's a const member

        return *this;
    }
DataSet::~DataSet() {

		}


