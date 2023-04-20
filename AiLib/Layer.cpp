#pragma once
#include "Enums.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <pybind11/pybind11.h>
#include "Layer.hpp"
#include <python3.10/Python.h>
using namespace std;

Layer::Layer(int in, int out, LayerF f)
{
	this->inputs.reserve(in+1);
	this->outputs.reserve(out);
	this->grads.reserve(out);
	this->outputsActiveted.reserve(out + 1);
	this->weights.reserve(out * (in+1));
	this->func = f;
	this->outputsActiveted[0] = 1;
	this->in = in+1;
	this->out = out;
	this->pastMomentum = vector<float>(out*this->in,0.0);
	this->pastVelocity = vector<float>(out*this->in,0.0);
	this->gen = mt19937(rd());
	float std = sqrt(2.000000/(this->in+this->out));
	this->random_dist = normal_distribution<float>(0,std);
	for(int i = 0;i<this->weights.size();i++){
		this->weights[i]=this->random_dist(gen);
	}
}

Layer::~Layer()
{
	this->outputs.clear();
	this->weights.clear();
	this->inputs.clear();
}
void init_my_module_Layer(pybind11::module& m){
	pybind11::class_<Layer>(m,"Layer")
		.def(pybind11::init<int, int, LayerF>());
}