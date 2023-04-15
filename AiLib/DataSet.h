#pragma once
#include <vector>
using namespace std;
class DataSet {
	public:
		vector<vector<float&>> inputs;
		vector<vector<float&>> outputs;
		DataSet(vector<vector<float&>> inputs, vector<vector<float&>> outputs) {
			this->inputs = inputs;
			this->outputs = outputs;
		}
		~DataSet() {
			for (int i = 0; i < this->inputs.size(); i++) {
				this->inputs[i].clear();
			}
			this->inputs.clear();
			for (int i = 0; i < this->outputs.size(); i++) {
				this->outputs[i].clear();
			}
			this->outputs.clear();
		}

};