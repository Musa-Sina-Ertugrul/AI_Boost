#pragma once
#include "Enums.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <pybind11/pybind11.h>
#include <python3.10/Python.h>
using namespace std;
class Layer
{
public:
	vector<float> inputs;
	vector<float> outputs;
	vector<float> grads;
	vector<float> outputsActiveted;
	vector<float> weights;
	vector<float> pastMomentum;
	vector<float> pastVelocity;
	LayerF func;
	float in;
	float out;
	Layer(int in,int out, LayerF f);
	~Layer();
private:
	random_device rd;
	mt19937 gen;
	normal_distribution<float> random_dist; 
};

void init_my_module_Layer(pybind11::module& m);