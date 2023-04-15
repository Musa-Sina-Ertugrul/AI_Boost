#pragma once
enum LossFType
{
	BCELoss = 1,
	CELoss = 2,
	L1Loss = 3,
	L2Loss = 4
};

enum LayerF
{
	Tanh = 5,
	Sigmoid = 6,
	Relu = 7
};

enum RegFType
{
	None = 8,
	L1 = 9,
	L2 = 10
};

struct EnumValue
{
	LossFType getBCELoss() {
		return LossFType(BCELoss);
	}
	LossFType getCELoss() {
		return LossFType(CELoss);
	}
	LossFType getL1Loss() {
		return LossFType(L1Loss);
	}
	LossFType getL2Loss() {
		return LossFType(L2Loss);
	}
	LayerF getTanh() {
		return LayerF(Tanh);
	}
	LayerF getSigmoid() {
		return LayerF(Sigmoid);
	}
	LayerF getRelu() {
		return LayerF(Relu);
	}
	RegFType getRegNone() {
		return RegFType(None);
	}
	RegFType getRegL1() {
		return RegFType(L1);
	}
	RegFType getRegL2() {
		return RegFType(L2);
	}
};
