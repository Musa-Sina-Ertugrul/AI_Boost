#pragma once
#include "Enums.hpp"
#include "/usr/include/c++/11/bits/stl_vector.h"
#include <random>
#include <cmath>

#include "Layer.hpp"
#include <python3.10/Python.h>


using namespace std;

Layer::Layer(int in, int out, LayerFunction f)
{
	this->inputs = vector<double>(in);
	this->outputs= vector<double>(out);
	this->outputsActiveted= vector<double>(out);
	this->weights= vector<double>(out * (in));
	this->F = f;
	//this->outputsActiveted[0] = 1.0;
	this->in = in;
	this->out = out;
	this->pastMomentum = vector<double>(out*this->in,0.0);
	this->pastVelocity = vector<double>(out*this->in,0.0);
	this->grads= vector<double>(in,0.0);
	this->errorWeights = vector<double>(out*this->in,0.0);
	this->gen = mt19937(random_device{}());
	double std = sqrt(2.000000/(this->in+this->out));
	this->random_dist = normal_distribution<double>(0.0,std);
	this->errorBias = vector<double>(out,0.0);
	this->bias = vector<double>(out,0.0);
	this->pastMomentumBias = vector<double>(out,0.0);
	this->pastVelocityBias = vector<double>(out,0.0);
	for(int i = 0;i<out*this->in;i++){
		this->weights[i]=this->random_dist(gen);
	}
}
Layer& Layer::operator=(const Layer& other) {
        if (this == &other) {
            return *this;
        }

        // Assign other members as needed
        // Note: id_ is not assigned since it's a const member

        return *this;
    }
Layer::Layer(Layer &object){
	this->F = object.F;
	this->in = object.in;
	this->out = object.out;
	this->outputs = object.outputs;
	this->inputs = object.inputs;
	this->weights = object.weights;
	this->grads = object.grads;
	this->pastMomentum = object.pastMomentum;
	this->pastVelocity = object.pastVelocity;
}
Layer::~Layer()
{
	this->outputs.clear();
	this->weights.clear();
	this->inputs.clear();
}
