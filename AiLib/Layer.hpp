#pragma once
#include "Enums.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <python3.10/Python.h>

using namespace std;
class Layer
{
public:
	vector<double> inputs;
	vector<double> outputs;
	vector<double> grads;
	vector<double> outputsActiveted;
	vector<double> weights;
	vector<double> pastMomentum;
	vector<double> pastVelocity;
	vector<double> errorWeights;
	vector<double> errorBias;
	vector<double> bias;
	vector<double> pastMomentumBias;
	vector<double> pastVelocityBias;
	LayerFunction F;
	normal_distribution<double> random_dist;
	mt19937 gen;
	int in;
	int out;
	Layer(int in,int out, LayerFunction f);
	Layer(Layer &object);
	~Layer();
	Layer& operator=(const Layer& other);

};

