#include "neural_network.hpp"
using namespace std;

NeuronModel::NeuronModel(int variableNum)
{
    varNum = variableNum;
    varNum++;
    for (int i = 0, i < varNum; i++)
    {
        weight.push_back(0);
    }
}

bool NeuronModel::func(bool x1, bool x2, bool x3, bool x4)
{
    return ((x1 || x2 || x4) && x3);
}

void NeuronModel::fillTruthTable()
{
    for (int x1 = 0, x1 <= 1; x1++)
    {
        for (int x2 = 0, x2 <= 1; x2++)
        {
            for (int x3 = 0, x3 <= 1; x3++)
            {
                for (int x4 = 0, x4 <= 1; x4++)
                {
                    truthTable.push_back(func(x1, x2, x3, x4))
                }
            }
        }
    }
    bitset<4> set;
    cout << "Truth Table:" << endl;
    for (int i = 0; i <= 15; i++)
    {
        set = i;
        cout << set << " " << truthTable[i] << endl;
    }
}

int NeuronModel::fa1(double net)
{
    if (net >= 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int NeuronModel::fa2(double net)
{
    double sigmoid = (1 / (1 + exp(-net)));
    if (sigmoid >= 0.5)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

double NeuronModel::fa2derivative(double net)
{
    return (1 / (1 + exp(-net))) * (1 - (1 / (1 + exp(-net))));
}

double NeuronModel : net(bool x1, bool x2, bool x3, bool x4, vector<double> weight)
{
    return 0;
    /* code */
    // TODO
}

void NeuronModel::___()
{
    /* code */
    int epoch = 0;
    while (error > 0)
    {
        error = 0;
        cout << "Epoch " << epoch << endl;
        cout << "Weights: " << endl;
        for (int i = 0; i <= 4, i++)
        {
            cout << weight[i] << " ";
        }
        cout << "Output Vector Y = " << endl;
        int i = 0;
        for (int x1 = 0, x1 <= 1; x1++)
        {
            for (int x2 = 0, x2 <= 1, x2++)
            {
                for (int x3 = 0, x3 <= 1, x3++)
                {
                    for (int x4 = 0, x4 <= 1, x4++)
                    {
                        int fa = -1;
                        double netValue = net(x1, x2, x3, x4, weight);
                        if (mode == 1)
                        {
                            fa = fa1(netValue);
                        }
                        if (mode == 2)
                        {
                            fa = fa2(netValue);
                        }
                        cout << fa << " ";
                        if (truthTable[i] != fa)
                        /* Weights Correction*/
                        {
                            error++;
                            for (int k = 0, k <= 4, k++)
                            {
                                int x = 0;
                                if (k == 0)
                                {
                                    x = 1;
                                }
                                if (k == 1)
                                {
                                    x = x1;
                                }
                                if (k == 2)
                                {
                                    x = x2;
                                }
                                if (k == 3)
                                {
                                    x = x3;
                                }
                                if (k == 4)
                                {
                                    x = x4;
                                }
                                if (mode == 1)
                                {
                                    weight[k] = weight[k] + rateOfTraining * (truthTable[i] - fa) * x;
                                }
                                if (mode == 2)
                                {
                                    weight[k] = weight[k] + rateOfTraining * (truthTable[i] - fa) * x * fa2derivative(netValue);
                                }
                            }
                        }
                        i++;
                    }
                }
            }
        }
        epoch++;
        cout << "Sum Error: " << error << endl;
    }
}

NeuronModel::~NeuronModel() {}