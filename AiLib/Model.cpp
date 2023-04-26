#pragma once
#include <python3.10/Python.h>
#include "DataSet.hpp"
#include "Enums.hpp"
#include "Layer.hpp"
#include "Model.hpp"
#include "DataSet.hpp"
#include <vector>
#include "/usr/include/x86_64-linux-gnu/openblas-pthread/cblas.h"
#include <cmath>
#include <iostream>
#include <iomanip>


using namespace std;

	Model::Model(vector<Layer*> layers,DataSet* datas,bool boucingLR,int epochs,double dropOutRate,RegFType regType,double regLambda,
	LossFType lossType,double learningRate,bool ZeroToOne,int bacthSize,int layerNumber){
		this->layers = layers;
		this->datas = datas;
		this->boucingLR = boucingLR;
		this->epochs = epochs;
		this->dropOutRate = dropOutRate;
		this->regType = regType;
		this->regLambda = regLambda;
		this->lossType = lossType;
		this->learningRate = learningRate;
		this->resultsLoss = vector<double>((int)this->layers[layerNumber-1]->out);
		this->results = vector<double>((int)this->layers[layerNumber-1]->out);
		this->ZeroToOne = ZeroToOne;
		this->bacthSize = bacthSize;
		this->regulazationNum = 0.0;
		this->layerNumber=layerNumber;
	}
	Model::~Model(){
	this->layers.clear();
	this->datas = nullptr;
}
	void Model::trainModel(){

		double correct = 0.0;
		double total = 0.0;
		for(int i = 0;i<this->epochs;i++){
			this->currentEpoch = i%(this->datas->row1);
			this->layers.at(0)->inputs.at(0)=0.001;
			for(int j = 0;j<this->datas->col1;j++){
				this->layers[0]->inputs[j+1] = this->datas->inputs.at(this->currentEpoch).at(j);
			}
			this->forward();
			if(ZeroToOne){
				for(int j = 0;j<this->datas->col2;j++){
					
					cout<<"result size "<<this->results.at(j)<<" ouputs size "<<this->datas->outputs.at(this->currentEpoch).at(j)<<endl;
					if(this->results.at(j)>0.5 && this->datas->outputs.at(this->currentEpoch).at(j)>0.5){
						correct = correct + 1.0;
					}else if (this->results.at(j)<0.5 && this->datas->outputs.at(this->currentEpoch).at(j)<0.5)
					{
						correct = correct + 1.0;
					}
				}
				correct /= (double)this->datas->col2+1.0;
			}else{
				for(int j = 0;j<this->layers[this->layerNumber-1]->out;j++){
					if(this->results[j]==this->datas->outputs[this->currentEpoch][j]){
						correct = correct + 1.0;
					}
				}
				correct /= (double)this->datas->col2+1.0;
			}
			if(0 == (this->currentEpoch+1)%this->bacthSize){
				cout<<fixed<<setprecision(6)<<"correctness: "<<correct*100.0<<endl;
				total = 0.0;
				correct = 0.0;
			}
			this->backward();

		}
		return;
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
		for(int i = 0;i<this->layerNumber;i++){
			this->currentLayer = i;
			this->bachNorm();
			cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,this->layers[i]->out,this->layers[i]->in,1.0,this->layers[i]->inputs.data(),this->layers[i]->in,
			this->layers[i]->weights.data(),this->layers[i]->out,0.0f,this->layers[i]->outputs.data(),this->layers[i]->out);
			if(this->layers[this->currentLayer]->F!=SoftMax){
				for(int j = 1;j<(int)this->layers[i]->out+1;j++){
					this->layers.at(i)->outputsActiveted.at(j)=this->activation(this->layers.at(i)->outputs.at(j-1));
				}
			}else{
				this->softmax();
			}
			this->layers.at(i)->outputsActiveted.at(0) = 1.0;
			if(this->layerNumber-1 != i){
				for(int j = 0;j<this->layers.at(i)->out+1;j++){
					this->layers.at(i+1)->inputs.at(j) = this->layers.at(i)->outputsActiveted.at(j);
				}
			}else{
				for(int j = 0;j<this->datas->col2;j++){
					this->results[j] = this->layers[this->layerNumber-1]->outputsActiveted[j+1];
				}
			}
		}
	}
	void Model::adam(){
		int j = 0;
		this->regulazationNum = this->reg();
		for(int i = 0;i<this->layers[this->currentLayer]->out*this->layers[this->currentLayer]->in;i++){
			if(j == this->layers[this->currentLayer]->out){
				j = 0;
			}
			double tmpGrad = this->currentGrad(i,j);
			if((tmpGrad<0.001&&tmpGrad>-0.001)||isnan(tmpGrad)){
				if(tmpGrad>0){
					tmpGrad = 0.001;
				}else{
					tmpGrad = -0.001;
				}
				
			}
			this->layers[this->currentLayer]->pastMomentum[i]= 0.9*this->layers[this->currentLayer]->pastMomentum[i] + (0.1*tmpGrad);
			this->layers[this->currentLayer]->pastVelocity[i] = 0.999*this->layers[this->currentLayer]->pastVelocity[i] + (0.001*tmpGrad*tmpGrad);
			double mHat = this->layers[this->currentLayer]->pastMomentum[i]/(1-pow(0.9,this->currentEpoch+1));
			double vHat = this->layers[this->currentLayer]->pastVelocity[i]/(1-pow(0.999,this->currentEpoch+1));
			double pastWeight = this->layers[this->currentLayer]->weights[i];
			this->layers[this->currentLayer]->weights[i]=this->layers[this->currentLayer]->weights[i]-this->currentLR*mHat/(sqrt(abs(vHat))+0.000001)
			+this->regLambda*this->regulazationNum;
			j++;
			if((this->layers[this->currentLayer]->weights[i]<0.001&&this->layers[this->currentLayer]->weights[i]>-0.001)
			||isnan(this->layers[this->currentLayer]->weights[i])){
				this->layers[this->currentLayer]->weights[i] = this->layers[this->currentLayer]->random_dist(this->layers[this->currentLayer]->gen);
				
			}
		}
	}
	inline double Model::currentGrad(int i,int j){
		return this->layers[this->currentLayer]->weights[i] *this->layers[this->currentLayer]->grads[j];
	}
	void Model::gradient(){
		if(this->layerNumber-1 == this->currentLayer){
			if(this->layers[this->currentLayer]->F != SoftMax){
				for(int j = 1;j<this->layers[this->currentLayer]->out+1;j++){
					this->layers[this->currentLayer]->grads[j-1]=this->dLoss(this->datas->outputs[this->currentEpoch][j-1],this->layers[this->currentLayer]->outputsActiveted[j])
					*this->dActivation(this->layers[this->currentLayer]->outputs[j-1]);
				}
				return;
			}
			vector<vector<double>> jacobian(this->datas->col2,vector<double>(this->datas->col2,0.0));
			for(int i = 0;i<this->datas->col2;i++){
				for(int j = 0;j<this->datas->col2;j++){
					int ij = i==j ? 1.0: 0.0;
					jacobian[i][j] = this->layers[this->layerNumber-1]->outputsActiveted[i] *(ij - this->layers[this->layerNumber-1]->outputsActiveted[j]);
				}
			}
			vector<vector<double>> jacobianTrans = this->transpose(jacobian);
			vector<double> jacobianTransFlat = makeFlat(jacobianTrans);
			cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,this->datas->col2,1,this->datas->col2,1.0,jacobianTransFlat.data(),this->datas->col2,this->layers[this->layerNumber-1]->outputsActiveted.data(),
			1,0.0,this->layers[this->layerNumber-1]->grads.data(),this->datas->col2);
			return;
		}
		double total = 0.0;
		for(int i = 0;i<(int)this->layers[this->currentLayer+1]->out;i++){
			total = total + this->layers[this->currentLayer+1]->grads[i];
		}
		for(int i = 1;i<(int)this->layers[this->currentLayer]->out+1;i++){
			this->layers[this->currentLayer]->grads[i-1]= total* this->layers[this->currentLayer]->outputsActiveted[i]
			*this->dActivation(this->layers[this->currentLayer]->outputs[i-1]);
		}
	}
	vector<double> Model::makeFlat(const vector<vector<double>>& matrix){

		vector<double> tmp(this->datas->col2*this->datas->col2);
		for(int i = 0;i<this->datas->col2*this->datas->col2;i++){
			tmp[i] = matrix[i/this->datas->col2][i%this->datas->col2];
		}
		return tmp;
	}
	vector<vector<double>> Model::transpose(const vector<vector<double>>& matrix) {
		int rows = matrix.size();
		int cols = matrix[0].size();
		vector<vector<double>> transposed_matrix(cols, vector<double>(rows));

		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				transposed_matrix[j][i] = matrix[i][j];
			}
		}

    return transposed_matrix;
}
	double Model::activation(double y) {
		switch (this->layers[this->currentLayer]->F)
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
	void Model::softmax(){
		double total = 0.0000000;
		for(int i = 0;i<this->layers[this->currentLayer]->out;i++){
			total += exp(this->layers[this->currentLayer]->outputs[i]);
		}
		for(int i = 0;i<this->layers[this->currentLayer]->out;i++){
			this->layers[this->currentLayer]->outputsActiveted[i+1]=exp(this->layers[this->currentLayer]->outputs[i])/total;
		}
	}
	inline double Model::dRelu(double x) {
		if (x > 0.0) {
			return 1.0;
		}
		return 0.0;
	}
	double Model::dActivation(double y) {
		switch (this->layers[this->currentLayer]->F)
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
	double Model::relu(double x) {
		if (x > 0.0) {
			return double(x);
		}
		return 0.0;
	}
	void Model::cosBounce() {
		this->currentLR = cos(this->learningRate * this->currentEpoch);
	}
	double Model::bCELoss(double yHat, double y) {
		return -(yHat * log(y) + (1 - yHat * (1 - log(1 - y))));
	}
	double Model::cELoss(double yHat, double y) {
		return -yHat * log(y);
	}
	double Model::l1Loss(double yHat, double y) {
		return abs(yHat - y);
	}
	double Model::l2Loss(double yHat, double y) {
		return (yHat - y)* (yHat - y);
	}
	double Model::loss(double yHat, double input) {
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
	double Model::dLoss(double yHat,double y) {
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
	inline double Model::dBCELoss(double yHat,double y) {
		return -((yHat / (y)) - (1 - yHat) / (1 - y));
	}
	inline double Model::dCELoss(double yHat,double y) {
		return yHat/y;
	}
	inline double Model::dl1Loss(double yHat,double y) {
		if (y > yHat) {
			return 1.0;
		}
		if (yHat > y) {
			return -1.0;
		}
		return 0.0;
	}
	inline double Model::dl2Loss(double yHat,double y) {
		return -(yHat - y);
	}
	void Model::bachNorm() {
		double total = 0.000000;
		int len = (int)this->layers[this->currentLayer]->in;
		for (int i = 0; i < len;i++ ) {
			total += this->layers[this->currentLayer]->inputs[i];
		}
		total = total/len;
		double meansqr = 0.000000;
		for (int i = 0; i < len; i++) {
			meansqr += abs((this->layers[this->currentLayer]->inputs[i] - total)* (this->layers[this->currentLayer]->inputs[i] - total));
		}
		double std = sqrt(abs(meansqr) / (len - 1));
		for (int i = 0; i < len; i++) {
			this->layers[this->currentLayer]->inputs[i] = (this->layers[this->currentLayer]->inputs[i]-total)/(std+ 0.000001);
		}

	}
	inline double Model::l1Reg() {
		double total(0.000000);
		for (int i = 0; i < this->layers[this->currentLayer]->in*this->layers[this->currentLayer]->out; i++) {
			total += abs(this->layers[this->currentLayer]->weights[i]);
		}
		return this->regLambda*total+0.00000001;
	}
	inline double Model::l2Reg() {
		double total(0.000000);
		for (int i = 0; i <this->layers[this->currentLayer]->in*this->layers[this->currentLayer]->out; i++) {
			total += this->layers[this->currentLayer]->weights[i]*this->layers[this->currentLayer]->weights[i];
		}
		return this->regLambda * total+0.00000001;
	}
	double Model::reg() {
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

