#pragma once
#include"Enums.hpp"
#include"DataSet.hpp"
#include"Layer.hpp"
#include <vector>
#include "/usr/include/x86_64-linux-gnu/openblas-pthread/cblas.h"
#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include <python3.10/Python.h>

using namespace std;

class Model
{
public:
	int epochs;
	bool boucingLR;
	LossFType lossType;
	vector<Layer*> layers;
	DataSet* datas;
	float dropOutRate;
	RegFType regType;
	float regLambda;
	float learningRate;
	bool ZeroToOne;
	int bacthSize;
	Model(vector<Layer*>layers,DataSet datas,bool boucingLR,int epochs,float dropOutRate,RegFType regType,float regLambda,
	LossFType lossType,float learningRate,bool ZeroToOne,int bacthSize);
	~Model();
	void trainModel();
private:
	int currentEpoch;
	float currentLR;
	int16_t currentLayer;
	vector<float> resultsLoss;
	vector<float> results;
	void backward();
	void forward();
	void adam();
	inline float&& currentGrad(int i,int j);
	void gradient();
	float&& activation(float& y);
	inline float&& dRelu(float& x);
	float&& dActivation(float& y);
	inline float&& relu(float& x);
	inline void cosBounce();
	inline float&& bCELoss(float& yHat, float& y);
	inline float&& cELoss(float& yHat, float& y);
	inline float&& l1Loss(float& yHat, float& y);
	inline float&& l2Loss(float& yHat, float& y);
	float&& loss(float& yHat, float& input);
	float&& dLoss(float& yHat,float& y);
	inline float&& dBCELoss(float& yHat,float& y);
	inline float&& dCELoss(float& yHat,float& y);
	inline float&& dl1Loss(float& yHat,float& y);
	inline float&& dl2Loss(float& yHat,float& y);
	void bachNorm();
	inline float&& l1Reg();
	inline float&& l2Reg();
	float&& reg();
};

void init_my_module_Modell(pybind11::module& m);