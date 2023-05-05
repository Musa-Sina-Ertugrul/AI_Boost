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
		this->resultsLoss = vector<double>((int)this->layers[layerNumber-1]->out,0.0);
		this->results = vector<double>((int)this->layers[layerNumber-1]->out,0.0);
		this->ZeroToOne = ZeroToOne;
		this->bacthSize = bacthSize;
		this->regulazationNum = 0.0;
		this->layerNumber=layerNumber;
		this->currentBatch = 1;
		this->normal_rand = normal_distribution<double>(0.0,0.001);
		this->currentLayer=0;
		this->currentEpoch=0;
		this->currentBatch=0;
		this->currentLR=learningRate;
	}
	Model::~Model(){
	this->layers.clear();
	this->datas = nullptr;
}
	void Model::trainModel(){

		double correct = 0.0;
		double total = 0.0;
		for(int i = 0;i<this->epochs;i++){
			this->currentEpoch = i%this->datas->row1;
			//this->layers.at(0)->inputs.at(0)=0.001;
			for(int j = 0;j<this->datas->col1;j++){
				this->layers[0]->inputs[j] = this->datas->inputs.at(this->currentEpoch).at(j);
			}
			this->bachNorm();
			this->forward();
			if(ZeroToOne){
				for(int j = 0;j<this->datas->col2;j++){
					
					//cout<<"result size "<<this->results.at(j)<<endl;
					//cout<<" ouputs size "<<this->datas->outputs.at(this->currentEpoch).at(j)<<endl;
					if(this->results.at(j)>0.5 && this->datas->outputs.at(this->currentEpoch).at(j)>0.5){
						correct = correct + 1.0;
					}else if (this->results.at(j)<0.5 && this->datas->outputs.at(this->currentEpoch).at(j)<0.5)
					{
						correct = correct + 1.0;
					}
				}
				//cout<<"---------"<<endl;
				correct /= (double)this->datas->col2+1.0;
			}else{
				for(int j = 0;j<this->layers[this->layerNumber-1]->out;j++){
					if(this->results[j]==this->datas->outputs[this->currentEpoch][j]){
						correct = correct + 1.0;
					}
				}
				correct /= (double)this->datas->col2+1.0;
			}
			this->backward();
			if(0 == (this->currentEpoch+1)%this->bacthSize){
				cout<<fixed<<setprecision(6)<<"correctness: "<<correct*100.0<<endl;
				total = 0.0;
				correct = 0.0;
				this->currentBatch++;
				for(int i = 0;i<this->layerNumber;i++){
					this->currentLayer = i;
					this->adam();
				}
			}
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
		}
	}
	void Model::forward(){
		for(int i = 0;i<this->layerNumber;i++){
			this->currentLayer = i;
			
			cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,this->layers[i]->out,this->layers[i]->in,1.0,this->layers[i]->inputs.data(),this->layers[i]->in,
			this->layers[i]->weights.data(),this->layers[i]->out,0.0,this->layers[i]->outputs.data(),this->layers[i]->out);
			for(int j = 0;j<this->layers[i]->out;j++){
				this->layers.at(i)->outputs.at(j) = this->layers.at(i)->outputs.at(j) + this->layers.at(i)->bias.at(j);
			}
			if(this->layers[this->currentLayer]->F!=SoftMax){
				for(int j = 0;j<(int)this->layers[i]->out;j++){
					this->layers.at(i)->outputsActiveted.at(j)=this->activation(this->layers.at(i)->outputs.at(j));
					/*if(isnanf(this->layers.at(i)->outputsActiveted.at(j))){
						this->layers.at(i)->outputsActiveted.at(j)=this->normal_rand(this->layers[this->currentLayer]->gen);
					}*/
				}
			}else{
				this->softmax();
			}
			if(this->layerNumber-1 != i){
				for(int j = 0;j<this->layers.at(i)->out;j++){
					/*if(isnanf(layers.at(i)->outputsActiveted.at(j))){
						layers.at(i)->outputsActiveted.at(j) = this->normal_rand(this->layers[this->currentLayer]->gen);
					}*/
					this->layers.at(i+1)->inputs.at(j) = this->layers.at(i)->outputsActiveted.at(j);
				}
			}else{
				/*if(ZeroToOne && this->layers[this->currentLayer]->F!=SoftMax){
					double total = 0.0;
					for(int j = 0;j<this->datas->col2;j++){
						total = total + this->layers[this->layerNumber-1]->outputsActiveted[j];
					}
					for(int j = 0;j<this->layers[this->layerNumber-1]->out;j++){
						this->layers[this->layerNumber-1]->outputsActiveted.at(j) = this->layers[this->layerNumber-1]->outputsActiveted.at(j)/total;
					}
				}*/
				for(int j = 0;j<this->datas->col2;j++){
					/*if(isnanf(this->layers[this->layerNumber-1]->outputsActiveted[j+1])){
						this->layers[this->layerNumber-1]->outputsActiveted[j+1] = this->normal_rand(this->layers[this->currentLayer]->gen);
					}*/
					this->results[j] = this->layers[this->layerNumber-1]->outputsActiveted[j];
				}
			}
		}
	}
	void Model::adam(){
		int j = 0;
		this->regulazationNum = this->reg();
		for(int i = 0;i<this->layers[this->currentLayer]->out;i++){
			double tmpGrad=this->layers[this->currentLayer]->errorBias[i]/this->bacthSize;
			this->layers[this->currentLayer]->errorBias[i] = 0.0;
			this->layers[this->currentLayer]->pastMomentumBias[i]= 0.9*this->layers[this->currentLayer]->pastMomentumBias[i] + (0.1*tmpGrad);
			this->layers[this->currentLayer]->pastVelocityBias[i] = 0.999*this->layers[this->currentLayer]->pastVelocityBias[i] + (0.001*tmpGrad*tmpGrad);
			double mHat = this->layers[this->currentLayer]->pastMomentumBias[i]/(1-pow(0.9,this->currentEpoch+1));
			double vHat = this->layers[this->currentLayer]->pastVelocityBias[i]/(1-pow(0.999,this->currentEpoch+1));
			this->layers[this->currentLayer]->bias[i]=this->layers[this->currentLayer]->bias[i]-(this->currentLR*mHat/(sqrt(abs(vHat))+0.000000000000001));
		}

		for(int i = 0;i<this->layers[this->currentLayer]->out*this->layers[this->currentLayer]->in;i++){
			double tmpGrad = this->layers[this->currentLayer]->errorWeights[i]/this->bacthSize;
			/*if(isnanf(tmpGrad)){
				tmpGrad = this->normal_rand(this->layers[this->currentLayer]->gen);
			}*/
			this->layers[this->currentLayer]->errorWeights[i] = 0.0;
			this->layers[this->currentLayer]->pastMomentum[i]= 0.9*this->layers[this->currentLayer]->pastMomentum[i] + (0.1*tmpGrad);
			/*if(isnanf(this->layers[this->currentLayer]->pastMomentum[i])){
				this->layers[this->currentLayer]->pastMomentum[i]=this->normal_rand(this->layers[this->currentLayer]->gen);
			}*/
			this->layers[this->currentLayer]->pastVelocity[i] = 0.999*this->layers[this->currentLayer]->pastVelocity[i] + (0.001*tmpGrad*tmpGrad);
			/*if(isnanf(this->layers[this->currentLayer]->pastVelocity[i])){
				this->layers[this->currentLayer]->pastVelocity[i]=this->normal_rand(this->layers[this->currentLayer]->gen);
			}*/
			/*double mHat = this->layers[this->currentLayer]->pastMomentum[i]/(1-(isnanf(pow(0.9,this->currentEpoch+1)?this->normal_rand(this->layers[this->currentLayer]->gen):pow(0.9,this->currentEpoch+1))));
			double vHat = this->layers[this->currentLayer]->pastVelocity[i]/(1-(isnanf(pow(0.999,this->currentEpoch+1)?this->normal_rand(this->layers[this->currentLayer]->gen):pow(0.999,this->currentEpoch+1))));
			*/
			double mHat = this->layers[this->currentLayer]->pastMomentum[i]/(1-pow(0.9,this->currentEpoch+1));
			double vHat = this->layers[this->currentLayer]->pastVelocity[i]/(1-pow(0.999,this->currentEpoch+1));
			/*if(isnanf(mHat)){
				mHat = this->normal_rand(this->layers[this->currentLayer]->gen);
			}*/
			/*if(isnanf(vHat)){
				vHat = this->normal_rand(this->layers[this->currentLayer]->gen);
			}*/
			this->layers[this->currentLayer]->weights[i]=this->layers[this->currentLayer]->weights[i]-(mHat*(this->currentLR/(sqrt(abs(vHat))+0.000000000000001)))
			+this->regLambda*this->regulazationNum;
			/*if(isnanf(this->layers[this->currentLayer]->weights[i])){
				this->layers[this->currentLayer]->weights[i]=this->normal_rand(this->layers[this->currentLayer]->gen);
			}*/
		}
	}
	void Model::gradient(){

		if(this->layerNumber-1 == this->currentLayer){
			vector<double> tmpSigma = vector<double>(this->layers[this->currentLayer]->out,0.0);
			if(this->layers[this->currentLayer]->F != SoftMax){
				for(int i = 0;i<this->layers[this->currentLayer]->out;i++){
					tmpSigma[i]=this->dLoss(this->datas->outputs[this->currentEpoch][i],this->results[i])
					*this->dActivation(this->layers[this->currentLayer]->outputs[i]);
				}
			}else{
				for(int i = 0;i<this->layers[this->currentLayer]->out;i++){
					tmpSigma[i]=this->dLoss(this->datas->outputs[this->currentEpoch][i],this->results[i]);
				}
			}
			cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,this->layers[this->currentLayer]->in,this->layers[this->currentLayer]->out,1,1.0,
			this->layers[this->currentLayer]->inputs.data(),1,tmpSigma.data(),
			this->layers[this->currentLayer]->out,1.0,this->layers[this->currentLayer]->errorWeights.data(),
			this->layers[this->currentLayer]->out);
			vector<double> tmpWeights = vector<double>(this->layers[this->currentLayer]->out*this->layers[this->currentLayer]->in,0.0);
			for(int i = 0;i<(this->layers[this->currentLayer]->in)*this->layers[this->currentLayer]->out;i++){
				tmpWeights[(i%(this->layers[this->currentLayer]->in))*this->layers[this->currentLayer]->out+i/(this->layers[this->currentLayer]->in)] 
				= this->layers[this->currentLayer]->weights.at(i);
			}
			for(int i = 0;i<this->layers[this->currentLayer]->out;i++){
				this->layers[this->currentLayer]->errorBias[i] = this->layers[this->currentLayer]->errorBias[i]+tmpSigma[i];
			}
			cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,this->layers[this->currentLayer]->in,this->layers[this->currentLayer]->out,1.0,
			tmpSigma.data(),this->layers[this->currentLayer]->out,tmpWeights.data(),this->layers[this->currentLayer]->in,0.0,this->layers[this->currentLayer]->grads.data()
			,this->layers[this->currentLayer]->in);
		}else{
			vector<double> tmpDActivedted = vector<double>(this->layers[this->currentLayer]->out,0.0);
			for(int i = 0;i<this->layers[this->currentLayer]->out;i++){
				tmpDActivedted[i]=this->dActivation(this->layers[this->currentLayer]->outputs[i])*this->layers[this->currentLayer+1]->grads[i];
			}
			cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,this->layers[this->currentLayer]->in,this->layers[this->currentLayer]->out,1,1.0,
			this->layers[this->currentLayer]->inputs.data(),1,tmpDActivedted.data(),
			this->layers[this->currentLayer]->out,1.0,this->layers[this->currentLayer]->errorWeights.data(),
			this->layers[this->currentLayer]->out);
			if(this->currentLayer != 0){
				vector<double> tmpWeights = vector<double>(this->layers[this->currentLayer]->out*this->layers[this->currentLayer]->in,0.0);
				for(int i = 0;i<(this->layers[this->currentLayer]->in)*this->layers[this->currentLayer]->out;i++){
					tmpWeights[(i%(this->layers[this->currentLayer]->in))*this->layers[this->currentLayer]->out+i/(this->layers[this->currentLayer]->in)] 
					= this->layers[this->currentLayer]->weights.at(i);
				}
				cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,this->layers[this->currentLayer]->in,this->layers[this->currentLayer]->out,1.0,
				tmpDActivedted.data(),this->layers[this->currentLayer]->out,tmpWeights.data(),this->layers[this->currentLayer]->in,0.0,this->layers[this->currentLayer]->grads.data()
				,this->layers[this->currentLayer]->in);
			}
			for(int i = 0;i<this->layers[this->currentLayer]->out;i++){
					this->layers[this->currentLayer]->errorBias[i] = this->layers[this->currentLayer]->errorBias[i]+tmpDActivedted[i];
				}
			
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
			/*return isnanf(tanh(y))?this->normal_rand(this->layers[this->currentLayer]->gen):tanh(y);*/
			return tanh(y);
			break;
		case Sigmoid:
			/*return (isnanf(exp(-y)))?0.5:1 / (1 + exp(-y));*/
			return 1 / (1 + exp(-y));
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
			/*if(isnanf(exp(this->layers[this->currentLayer]->outputs[i]))){
				continue;
			}*/
			total += exp(this->layers[this->currentLayer]->outputs[i]);
		}
		for(int i = 0;i<this->layers[this->currentLayer]->out;i++){
			this->layers[this->currentLayer]->outputsActiveted[i]=(exp(this->layers[this->currentLayer]->outputs[i]))/(total);
			/*if(isnanf(this->layers[this->currentLayer]->outputsActiveted[i+1])){
				this->layers[this->currentLayer]->outputsActiveted[i+1]=this->normal_rand(this->layers[this->currentLayer]->gen);
			}*/
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
			/*return isnanf(tanh(y)*tanh(y)) ? 1.0: 1-tanh(y)*tanh(y);*/
			return 1-tanh(y)*tanh(y);
			break;
		case Sigmoid:
			/*return (isnanf(exp(-y)))?0.25:(1 / (1 + exp(-y)))*(1-(1 / (1 + exp(-y))));*/
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
		this->currentLR = this->learningRate *abs(cos( this->currentEpoch));
		/*if(isnanf(this->currentLR)){
			this->currentLR=this->normal_rand(this->layers[this->currentLayer]->gen);
		}*/
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
		return -((yHat / (y+0.00000001)) - (1 - yHat) / ((1 - y)));
	}
	inline double Model::dCELoss(double yHat,double y) {
		return y-yHat ;
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
			/*if(isnanf(this->layers[this->currentLayer]->inputs[i])){
				this->layers[this->currentLayer]->inputs[i]=this->normal_rand(this->layers[this->currentLayer]->gen);
			}*/
			total += this->layers[this->currentLayer]->inputs[i];
		}
		total = total/len;
		double meansqr = 0.000000;
		for (int i = 0; i < len; i++) {
			meansqr += abs((this->layers[this->currentLayer]->inputs[i] - total)* (this->layers[this->currentLayer]->inputs[i] - total));
		}
		/*if(isnanf(meansqr)){
			meansqr = 0.0;
		}*/
		double std = sqrt(abs(meansqr) / (len - 1));
		for (int i = 0; i < len; i++) {
			this->layers[this->currentLayer]->inputs[i] = (this->layers[this->currentLayer]->inputs[i]-total)/(std+ 1e-10);
			/*if(isnanf(this->layers[this->currentLayer]->inputs[i])){
				this->layers[this->currentLayer]->inputs[i]=this->normal_rand(this->layers[this->currentLayer]->gen);
			}*/
		}

	}
	inline double Model::l1Reg() {
		double total(0.000000);
		for (int i = 0; i < this->layers[this->currentLayer]->in*this->layers[this->currentLayer]->out; i++) {
			/*if(isnanf(this->layers[this->currentLayer]->weights[i])){
				this->layers[this->currentLayer]->weights[i] = this->normal_rand(this->layers[this->currentLayer]->gen);
			}*/
			total += abs(this->layers[this->currentLayer]->weights[i]);
		}
		return total;
	}
	inline double Model::l2Reg() {
		double total(0.000000);
		for (int i = 0; i <this->layers[this->currentLayer]->in*this->layers[this->currentLayer]->out; i++) {
			/*if(isnanf(this->layers[this->currentLayer]->weights[i])){
				this->layers[this->currentLayer]->weights[i] = this->normal_rand(this->layers[this->currentLayer]->gen);
			}*/
			total += this->layers[this->currentLayer]->weights[i]*this->layers[this->currentLayer]->weights[i];
		}
		return total;
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

