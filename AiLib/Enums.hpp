#pragma once

#include <vector>

#include <python3.10/Python.h>
using namespace std;
enum LossFType
{
	BCELoss = 1,
	CELoss = 2,
	L1Loss = 3,
	L2Loss = 4
};

enum LayerFunction
{
	SoftMax = 5,
	Tanh = 6,
	Sigmoid = 7,
	Relu = 8
};

enum RegFType
{
	None = 9,
	L1 = 10,
	L2 = 11
};
