#pragma once
#include "DataSet.hpp"
#include "Enums.hpp"
#include "Layer.hpp"
#include <vector>
#include "/usr/include/x86_64-linux-gnu/openblas-pthread/cblas.h"
#include <cmath>
#include <iostream>
#include <python3.10/Python.h>
#include <mutex>


using namespace std;
class Layer;
class Model
{
public:
	int epochs;
	bool boucingLR;
	LossFType lossType;
	vector<Layer*> layers;
	DataSet* datas;
	double dropOutRate;
	RegFType regType;
	double regLambda;
	double learningRate;
	bool ZeroToOne;
	int bacthSize;
	Model(vector<Layer*> layers,DataSet* datas,bool boucingLR,int epochs,double dropOutRate,RegFType regType,double regLambda,
	LossFType lossType,double learningRate,bool ZeroToOne,int bacthSize,int layerNumber);
	~Model();
	void trainModel();
private:
	int currentBatch;
	int layerNumber;
	normal_distribution<double> normal_rand;
	long currentEpoch;
	double currentLR;
	int16_t currentLayer;
	vector<double> resultsLoss;
	vector<double> results;
	double regulazationNum;
	void backward();
	void forward();
	void adam();
	double currentGrad(int i,int j);
	void gradient();
	double activation(double y);
	double dRelu(double x);
	double dActivation(double y);
	double relu(double x);
	void softmax();
	void cosBounce();
	vector<vector<double>> transpose(const vector<vector<double>>& matrix);
	vector<double> makeFlat(const vector<vector<double>>& matrix);
	double bCELoss(double yHat, double y);
	double cELoss(double yHat, double y);
	double l1Loss(double yHat, double y);
	double l2Loss(double yHat, double y);
	double loss(double yHat, double input);
	double dLoss(double yHat,double y);
	double dBCELoss(double yHat,double y);
	double dCELoss(double yHat,double y);
	double dl1Loss(double yHat,double y);
	double dl2Loss(double yHat,double y);
	void bachNorm();
	double l1Reg();
	double l2Reg();
	double reg();
};
