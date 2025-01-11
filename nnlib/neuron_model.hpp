#ifndef neuron_model_hpp
#define neuron_model_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <bitset>
#include <string>

class NeuronModel
{
private:
    const double rateOfTraining = 0.3;
    int varNum = 0;
    std::vector<double> weight;
    std::vector<bool> truthTable;

    bool func(bool x1, bool x2, bool x3, bool x4);
    void fillTruthTable();
    int fa1(double net);
    int fa2(double net);
    double fa2derivative(double net);
    double net(bool x1, bool x2, bool x3, bool x4, std::vector<double> weight);

public:
    NeuronModel(int : variableNum);
    void fullTraining(int mode);
    ~NeuronModel();
}