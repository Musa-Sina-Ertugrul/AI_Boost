#pragma once
#include"Enums.hpp"
#include"DataSet.hpp"
#include"Layer.hpp"
#include <vector>
#include "/usr/include/x86_64-linux-gnu/openblas-pthread/cblas.h"
#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include "Model.hpp"
#include <python3.10/Python.h>

using namespace std;

	Model::Model(vector<Layer*>layers,DataSet datas,bool boucingLR,int epochs,float dropOutRate,RegFType regType,float regLambda,
	LossFType lossType,float learningRate,bool ZeroToOne,int bacthSize){
		this->layers = layers;
		this->datas = &datas;
		this->boucingLR = boucingLR;
		this->epochs = epochs;
		this->dropOutRate = dropOutRate;
		this->regType = regType;
		this->regLambda = regLambda;
		this->lossType = lossType;
		this->learningRate = learningRate;
		this->resultsLoss.reserve(this->layers[this->layers.size()-1]->outputs.size());
		this->results.reserve(this->layers[this->layers.size()-1]->outputs.size());
		this->ZeroToOne = ZeroToOne;
		this->bacthSize = bacthSize;
	}
	Model::~Model(){
	for(int i = 0;i<this->layers.size();i++){
		delete this->layers[i];
	}
	this->layers.clear();
	this->datas = nullptr;
}
	void Model::trainModel(){
		float correct = 0.0;
		float total = 0.0;
		for(int i = 0;i<this->epochs;i++){
			this->currentEpoch = i;
			this->layers[0]->inputs[0]=1.0;
			for(int j = 1;j<sizeof(this->datas->inputs[i])+1;j++){
				this->layers[0]->inputs[j] = this->datas->inputs[i][j-1];
			}
			this->forward();
			
			for(int j = 0;j<this->resultsLoss.size();j++){
				total = total + this->resultsLoss[j];
			}
			if(ZeroToOne){
				for(int j = 0;j<this->results.size();j++){
					if(this->results[j]>0.5 && this->datas->outputs[i][j]>0.5){
						correct = correct + 1.0;
					}else if(this->results[j]<0.5 && this->datas->outputs[i][j]<0.5){
						correct = correct + 1.0;
					}
					correct = correct / (float)this->results.size();
				}
			}else{
				for(int j = 0;j<this->results.size();j++){
					if(this->results[j]==this->datas->outputs[i][j]){
						correct = correct + 1.0;
					}
					correct = correct / (float)this->results.size();
				}
			}
			if(this->bacthSize == (i+1)%this->bacthSize){
				cout<<"error: "<<total/(this->resultsLoss.size()*this->bacthSize)<<"correctness: "<<correct/(float)this->bacthSize<<endl;
				total = 0.0;
				correct = 0.0;
			}
			this->backward();
		}
	}
	void Model::backward(){
		for(int i = this->currentLayer;i>-1;i--){
			this->currentLayer = i;
			if(this->boucingLR){
				this->cosBounce();
			}
			this->gradient();
			this->adam();
		}
	}
	void Model::forward(){
		for(int i = 0;i<sizeof(this->layers);i++){
			this->currentLayer = i;
			this->bachNorm();
			cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,this->layers[i]->out,this->layers[i]->in,1.0f,this->layers[i]->inputs.data(),this->layers[i]->in,
			this->layers[i]->weights.data(),this->layers[i]->out,0.0f,this->layers[i]->outputs.data(),this->layers[i]->out);
			for(int j = 1;j<sizeof(this->layers[i]->outputsActiveted);j++){
				this->layers[i]->outputsActiveted[j]=this->activation(this->layers[i]->outputs[i-1]);
			}
			if(sizeof(this->layers)-1 != i){
				for(int j = 0;j<this->layers[i]->outputsActiveted.size();j++){
					this->layers[i+1]->inputs[j] = this->layers[i]->outputsActiveted[j];
				}
			}else{
				for(int j = 1;j<this->layers[i]->outputsActiveted.size();j++){
					this->resultsLoss[j-1]= this->loss(this->datas->outputs[this->currentEpoch][j-1],this->layers[i]->outputsActiveted[j]);
					this->results[j-1]=this->layers[i]->outputsActiveted[j];
				}
			}
		}
	}
	void Model::adam(){
		int j = 0;
		for(int i = 0;i<this->layers[this->currentLayer]->weights.size();i++){
			if(j == this->layers[this->currentLayer]->out){
				j = 0;
			}
			float tmpGrad = this->currentGrad(i,j);
			this->layers[this->currentLayer]->pastMomentum[i]= 0.9*this->layers[this->currentLayer]->pastMomentum[i] + (0.1*tmpGrad);
			this->layers[this->currentLayer]->pastVelocity[i] = 0.999*this->layers[this->currentLayer]->pastVelocity[i] + (0.001*tmpGrad*tmpGrad);
			float mHat = this->layers[this->currentLayer]->pastMomentum[i]/(1-pow(0.9,this->currentEpoch+1));
			float vHat = this->layers[this->currentLayer]->pastVelocity[i]/(1-pow(0.999,this->currentEpoch+1));
			this->layers[this->currentLayer]->weights[i]=this->layers[this->currentLayer]->weights[i]-this->currentLR*mHat/(sqrt(vHat)+0.000001)
			+this->regLambda*this->reg();
			j++;
		}
	}
	inline float&& Model::currentGrad(int i,int j){
		return this->layers[this->currentLayer]->weights[i] *this->layers[this->currentLayer]->grads[j];
	}
	void Model::gradient(){
		if(sizeof(this->layers)-1 == this->currentLayer){
			for(int j = 1;j<this->layers[this->currentLayer]->outputsActiveted.size();j++){
				this->layers[this->currentLayer]->grads[j-1]=this->dLoss(this->datas->outputs[this->currentEpoch][j-1],this->layers[this->currentLayer]->outputsActiveted[j])
				*this->dActivation(this->layers[this->currentLayer]->outputs[j-1]);
			}
			return;
		}
		float total = 0.0;
		for(int i = 0;i<this->layers[this->currentLayer+1]->grads.size();i++){
			total = total + this->layers[this->currentLayer+1]->grads[i];
		}
		for(int i = 1;i<this->layers[this->currentLayer]->outputsActiveted.size();i++){
			this->layers[this->currentLayer]->grads[i-1]= total* this->layers[this->currentLayer]->outputsActiveted[i]
			*this->dActivation(this->layers[this->currentLayer]->outputs[i-1]);
		}
	}
	float&& Model::activation(float& y) {
		switch (this->layers[this->currentLayer]->func)
		{
		case Tanh:
			return tanh(y);
			break;
		case Sigmoid:
			return 1 / (1 + exp(-y));;
			break;
		case Relu:
			return this->relu(y);
			break;
		default:
			throw "type correct activation func";
			break;
		}
	}
	inline float&& Model::dRelu(float& x) {
		if (x > 0.0) {
			return 1.0;
		}
		return 0.0;
	}
	float&& Model::dActivation(float& y) {
		switch (this->layers[this->currentLayer]->func)
		{
		case Tanh:
			return 1-(tanh(y)*tanh(y));
			break;
		case Sigmoid:
			return (1 / (1 + exp(-y)))*(1-(1 / (1 + exp(-y))));
			break;
		case Relu:
			return this->dRelu(y);
			break;
		default:
			throw "type correct activation func";
			break;
		}
	}
	inline float&& Model::relu(float& x) {
		if (x > 0.0) {
			return float(x);
		}
		return 0.0;
	}
	inline void Model::cosBounce() {
		this->currentLR = cos(this->learningRate * this->currentEpoch);
	}
	inline float&& Model::bCELoss(float& yHat, float& y) {
		return -(yHat * log(y) + (1 - yHat * (1 - log(1 - y))));
	}
	inline float&& Model::cELoss(float& yHat, float& y) {
		return yHat * log(y);
	}
	inline float&& Model::l1Loss(float& yHat, float& y) {
		return abs(yHat - y);
	}
	inline float&& Model::l2Loss(float& yHat, float& y) {
		return (yHat - y)* (yHat - y);
	}
	float&& Model::loss(float& yHat, float& input) {
		switch (this->lossType)
		{
		case BCELoss:
			return this->bCELoss(yHat,input);
			break;
		case CELoss:
			return this->cELoss(yHat,input);
			break;
		case L1Loss:
			return this->l1Loss(yHat,input);
			break;
		case L2Loss:
			return this->l2Loss(yHat,input);
			break;
		default:
			throw "type correct loss func";
			break;
		}
	}
	float&& Model::dLoss(float& yHat,float& y) {
		switch (this->lossType)
		{
		case BCELoss:
			return this->dBCELoss(yHat,y);
			break;
		case CELoss:
			return this->dCELoss(yHat,y);
			break;
		case L1Loss:
			return this->dl1Loss(yHat,y);
			break;
		case L2Loss:
			return this->dl2Loss(yHat,y);
			break;
		default:
			throw "type correct loss func";
			break;
		}
	}
	inline float&& Model::dBCELoss(float& yHat,float& y) {
		return -(yHat / y - (1 - yHat) / (1 - y));
	}
	inline float&& Model::dCELoss(float& yHat,float& y) {
		return -((yHat / y) - ((1 - yHat) / (1 - y)));
	}
	inline float&& Model::dl1Loss(float& yHat,float& y) {
		if (y > yHat) {
			return 1.0;
		}
		if (yHat > y) {
			return -1.0;
		}
		return 0.0;
	}
	inline float&& Model::dl2Loss(float& yHat,float& y) {
		return -(yHat - y);
	}
	void Model::bachNorm() {
		float total = 0.000000;
		int len = sizeof(this->layers[this->currentLayer]->inputs);
		for (int i = 0; i < len;i++ ) {
			total += this->layers[this->currentLayer]->inputs[i];
		}
		total = total/len;
		float meansqr = 0.000000;
		for (int i = 0; i < len; i++) {
			meansqr += (this->layers[this->currentLayer]->inputs[i] - total)* (this->layers[this->currentLayer]->inputs[i] - total);
		}
		float std = sqrt(meansqr / (len - 1));
		for (int i = 0; i < len; i++) {
			this->layers[this->currentLayer]->inputs[i] = (this->layers[this->currentLayer]->inputs[i]-total)/std;
		}

	}
	inline float&& Model::l1Reg() {
		float total(0.000000);
		for (int i = 0; i < this->layers[this->currentLayer]->weights.size(); i++) {
			total += abs(this->layers[this->currentLayer]->weights[i]);
		}
		return this->regLambda*total;
	}
	inline float&& Model::l2Reg() {
		float total(0.000000);
		for (int i = 0; i < this->layers[this->currentLayer]->weights.size(); i++) {
			total += this->layers[this->currentLayer]->weights[i]*this->layers[this->currentLayer]->weights[i];
		}
		return this->regLambda * total;
	}
	float&& Model::reg() {
		switch (this->regType)
		{
		case None:
			return 0.0;
			break;
		case L1:
			return this->l1Reg();
			break;
		case L2:
			return this->l2Reg();
			break;
		default:
			throw "type correct reg type";
			break;
		}
	}

void init_my_module_Modell(pybind11::module& m){
	pybind11::class_<Model>(m,"Model")
		.def(pybind11::init<vector<Layer*>,DataSet,bool,int,float,RegFType ,float,LossFType,float,bool,int>())
		.def("trainModel",&Model::trainModel);
}
