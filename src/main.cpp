#include "neuron_model.cpp"
using namespase std;

int main(int argc; const char *argv[])
{
    cout << "Training on Full Sample" << endl;
    cout << "FA 1" << endl;
    NeuronModel N1(4);
    N1.fullTraining(1);

    cout << "FA 2" << endl;
    NeuronModel N2(4);
    N1.fullTraining(2);
}