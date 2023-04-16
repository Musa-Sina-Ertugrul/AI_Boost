#pragma once
#include"Enums.h"
#include"DataSet.h"
#include"Layer.h"
#include <vector>
#include <cblas.h>
#include <cmath>
using namespace std;

class Model
{
public:
	int16_t epochs;
	bool boucingLR;
	LossFType lossType;
	Layer** layers;
	DataSet* datas;
	float dropOutRate;
	RegFType regType;
	float regLambda;
	float learningRate;
	Model(Layer layers[],DataSet &datas,bool &&boucingLR,int16_t epochs,float &&dropOutRate,RegFType &&regType,float &&regLambda,LossFType &&lossType,float &&learningRate);
	~Model();
	void trainModel(){
		for(int i = 0;i<this->epochs;i++){
			this->currentEpoch = i;
			this->layers[0]->inputs[0]=1.0;
			for(int j = 1;j<sizeof(this->datas->inputs[i])+1;j++){
				this->layers[0]->inputs[j] = *this->datas->inputs[i][j-1];
			}
			this->forward();
			this->backward();
		}
	}
private:
	int16_t currentEpoch;
	float currentLR;
	int16_t currentLayer;
	void backward(){
		for(int i = this->currentLayer;i>-1;i--){
			this->currentLayer = i;
			if(this->boucingLR){
				this->cosBounce();
			}
			this->gradient();
			adam();
		}
	}
	void forward(){
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
				
			}
		}
	}
	void adam(){
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
	inline float&& currentGrad(int i,int j){
		return this->layers[this->currentLayer]->weights[i] *this->layers[this->currentLayer]->grads[j];
	}
	void gradient(){
		if(sizeof(this->layers)-1 == this->currentLayer){
			for(int j = 1;j<this->layers[this->currentLayer]->outputsActiveted.size();j++){
				this->layers[this->currentLayer]->grads[j-1]=this->dLoss(*this->datas->outputs[this->currentEpoch][j-1],this->layers[this->currentLayer]->outputsActiveted[j])
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
	float&& activation(float& y) {
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
	inline float&& dRelu(float& x) {
		if (x > 0.0) {
			return 1.0;
		}
		return 0.0;
	}
	float&& dActivation(float& y) {
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
	inline float&& relu(float& x) {
		if (x > 0.0) {
			return move(x);
		}
		return 0.0;
	}
	inline void cosBounce() {
		this->currentLR = cos(this->learningRate * this->currentEpoch);
	}
	inline float&& bCELoss(float& yHat, float& y) {
		return -(yHat * log(y) + (1 - yHat * (1 - log(1 - y))));
	}
	inline float&& cELoss(float& yHat, float& y) {
		return yHat * log(y);
	}
	inline float&& l1Loss(float& yHat, float& y) {
		return abs(yHat - y);
	}
	inline float&& l2Loss(float& yHat, float& y) {
		return (yHat - y)* (yHat - y);
	}
	float&& loss(float& yHat, float& input) {
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
	float&& dLoss(float& yHat,float& y) {
		switch (this->lossType)
		{
		case BCELoss:
			return this->dBCELoss(y,yHat);
			break;
		case CELoss:
			return this->dCELoss(y, yHat);
			break;
		case L1Loss:
			return this->dl1Loss(y, yHat);
			break;
		case L2Loss:
			return this->dl2Loss(y, yHat);
			break;
		default:
			throw "type correct loss func";
			break;
		}
	}
	inline float&& dBCELoss(float& yHat,float& y) {
		return -(yHat / y - (1 - yHat) / (1 - y));
	}
	inline float&& dCELoss(float& yHat,float& y) {
		return -((yHat / y) - ((1 - yHat) / (1 - y)));
	}
	inline float&& dl1Loss(float& yHat,float& y) {
		if (y > yHat) {
			return 1.0;
		}
		if (yHat > y) {
			return -1.0;
		}
		return 0.0;
	}
	inline float&& dl2Loss(float& yHat,float& y) {
		return -(yHat - y);
	}
	void bachNorm() {
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
	inline float&& l1Reg() {
		float total(0.000000);
		for (int i = 0; i < this->layers[this->currentLayer]->weights.size(); i++) {
			total += abs(this->layers[this->currentLayer]->weights[i]);
		}
		return this->regLambda*total;
	}
	inline float&& l2Reg() {
		float total(0.000000);
		for (int i = 0; i < this->layers[this->currentLayer]->weights.size(); i++) {
			total += this->layers[this->currentLayer]->weights[i]*this->layers[this->currentLayer]->weights[i];
		}
		return this->regLambda * total;
	}
	float&& reg() {
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
};

Model::Model(Layer layers[],DataSet &datas,bool &&boucingLR,int16_t epochs,float &&dropOutRate,RegFType &&regType,float &&regLambda,LossFType &&lossType,float &&learningRate){
		this->layers = &layers;
		this->datas = &datas;
		this->boucingLR = boucingLR;
		this->epochs = epochs;
		this->dropOutRate = dropOutRate;
		this->regType = regType;
		this->regLambda = regLambda;
		this->lossType = lossType;
		this->learningRate = learningRate;
	}

Model::~Model()
{
	this->layers = nullptr;
	this->datas = nullptr;
}
