#pragma once
#include "Enums.h"
#include <vector>
using namespace std;
class Layer
{
public:
	vector<float&>* inputs;
	vector<float> outputs;
	vector<float> weights;
	LayerF f;
	Layer(int in,int out, LayerF &&f);
	~Layer();

};

Layer::Layer(int in, int out, LayerF&& f)
{
	this->outputs.reserve(out + 1);
	this->weights.reserve((out + 1) * in);
	this->f = f;
	this->outputs[0] = 1;
}

Layer::~Layer()
{
	this->outputs.clear();
	this->weights.clear();
	delete this->inputs;
	this->inputs = nullptr;
}