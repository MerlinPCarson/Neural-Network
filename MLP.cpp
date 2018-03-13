/*
	Multi-Layer-Perceptron network (MNIST)
	Created by: Merlin Carson
	Authored: 2-5-2018

	Notes: This is a windows application that loads the mnist training and test sets in csv format from the subdirectory data
	\data\minst_train
	\data\mnist_test
	
	By default, the program displays the confusion matrix for the training and test sets after each epoch.
	By uncommenting the DEMO pre-processor directive, instead of the confusion matricies, a random sample
	from the test set is displayed along with the prediction made by the network after each epoch.
*/

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <random>
#include <Windows.h>

using namespace std;

//#define DEBUG
//#define SHOW_WEIGHTS
//#define SHOW_DATA
//#define TESTING
//#define DEMO
#define RANDOMIZE

// hyper parameters
const int INPUT_SIZE = 784;			// number of inputs -> flatten(28x28)
//const int HIDDEN_SIZE = 10;			// number of neurons in hidden layer
const int OUTPUT_SIZE = 10;			// number of neurons in output layer

// data files 
const int MAX_VALUE = 255;	// for Normalization
#ifdef TESTING
const string TRAIN_DATA = "data\\mnist_train_100.csv";
const int TRAIN_SIZE = 100;
const string TEST_DATA = "data\\mnist_test_10.csv";
const int TEST_SIZE = 10;
#else
const string TRAIN_DATA = "data\\mnist_train.csv";
const int TRAIN_SIZE = 60000;
const string TEST_DATA = "data\\mnist_test.csv";
const int TEST_SIZE = 10000;
#endif

// output files
const std::string TRAIN_ACC = "data\\train_acc";
const std::string TEST_ACC = "data\\test_acc";

// data structures
struct Data{
	int label;
	double * value;
};

struct Neuron{
	double * weight;
	double * prev_delta_weight;
	double bias;
	double bias_prev_delta_weight;
	double activation;
	double loss;
	double connection_error;
};

struct Layer{
	Neuron * neuron;
	int size;
};

struct NeuralNet{
	Layer hidden;
	Layer output;
};

// Neural Net functions
void build_model(NeuralNet &neural_net, int hidden_size);
void init_layer(Layer &layer, int input_size);

void feed_forward_hidden(Layer &layer, const Data &input);
void feed_forward_output(Layer &layer, const Layer &input);

void back_prop_output(Layer &layer, const Layer &input, int label, double learning_rate, double momentum_rate, double decay_rate);
void back_prop_hidden(Layer &layer, const Data &input, const Layer &output, double learning_rate, double momentum_rate, double decay_rate);
double sigmoid(double input);
int softmax(const Layer &layer);


// helper functions
void load_csv(Data * &data, string data_file, int size);
void init_train_order(int * train_order);
double evaluate(int * perdiction, Data * data, int size);
int save_accuracy(double * train_acc, double * test_acc, int num_epochs, double learning_rate);
void print_number(Data * test_data, int * test_preds);
void print_prediction(Data data, int prediction);


int main(void){
	
	char end_prompt = ' ';
	double learning_rate = 0.001;
	double momentum_rate = 0.9;
	double decay_rate = 0.1;
	int num_epochs = 1;
	int hidden_size = 20;
	
	// create the network variable
	NeuralNet neural_net;
	
	// init example data structs
	Data * train_data = new Data[TRAIN_SIZE];
	int * train_preds = new int[TRAIN_SIZE];
	double * train_acc;
	Data * test_data = new Data[TEST_SIZE];
	int * test_preds = new int[TEST_SIZE];
	double * test_acc;

	int  * train_order = new int[TRAIN_SIZE];

	// load training and test data from csv file
	load_csv(train_data, TRAIN_DATA, TRAIN_SIZE);
	load_csv(test_data, TEST_DATA, TEST_SIZE);
	

	// main loop
	do{
		
		std::cout << "Enter the number of neurons in hidden layer: ";
		std::cin >> hidden_size;
		
		std::cout << "Enter the learning rate: ";
		std::cin >> learning_rate;

		std::cout << "Enter the momentum for the learning: ";
		std::cin >> momentum_rate;

		std::cout << "Enter the weight decay: ";
		std::cin >> decay_rate;

		std::cout << "Enter the number of epochs: ";
		std::cin >> num_epochs;
		
		// initilize layers and neurons
		build_model(neural_net, hidden_size);
		
		// initilize accuracy arrays
		train_acc = new double[num_epochs];
		test_acc  = new double[num_epochs];

		// training set
		for (int epoch = 0; epoch < num_epochs; ++epoch){

			std::cout << "Epoch: " << epoch << endl << endl;
			
			// initialize order of training examples
			init_train_order(train_order);
			
			for (int i = 0; i < TRAIN_SIZE; ++i){				// for each training example
				
				// forward feed
				feed_forward_hidden(neural_net.hidden, train_data[train_order[i]]);			// feed input -> hidden
				feed_forward_output(neural_net.output, neural_net.hidden);					// feed hidden -> output
				
				// get prediction from output layer
				train_preds[train_order[i]] = softmax(neural_net.output);

				// update weights if perdiction was wrong and it's not the first epoch				
				if (train_preds[train_order[i]] != train_data[train_order[i]].label && epoch != 0){

					// SGD: update output layer weights
					back_prop_output(neural_net.output, neural_net.hidden, train_data[train_order[i]].label, learning_rate, momentum_rate, decay_rate);
					// SGD: update hidden layer weights
					back_prop_hidden(neural_net.hidden, train_data[train_order[i]], neural_net.output, learning_rate, momentum_rate, decay_rate);

				}

			}

			// test set
			for (int i = 0; i < TEST_SIZE; ++i){
				// forward feed
				feed_forward_hidden(neural_net.hidden, test_data[i]);			// feed input -> hidden
				feed_forward_output(neural_net.output, neural_net.hidden);		// feed hidden -> output
				// get prdictions from output layer
				test_preds[i] = softmax(neural_net.output);

			}

			
#ifdef DEMO
			print_number(test_data, test_preds);
#endif

			// calc accuracy
#ifndef DEMO
			std::cout << "Train -- ";
#endif
			train_acc[epoch] = evaluate(train_preds, train_data, TRAIN_SIZE);
#ifndef DEMO
			std::cout << "Test -- ";
#endif
			test_acc[epoch] = evaluate(test_preds, test_data, TEST_SIZE);

			std::cout << "train accuracy: " << train_acc[epoch] << " test accuracy: " << test_acc[epoch] << endl << endl;
		}



		// save accuracies of epochs to files
		save_accuracy(train_acc, test_acc, num_epochs, learning_rate);

		std::cout << "Would you like to start the training over: ";
		std::cin >> end_prompt;

	} while (toupper(end_prompt) == 'Y');



	return 1;
}

// Neural Network functions
void build_model(NeuralNet &neural_net, int hidden_size){
	
	//size of other layers is number of neurons
	neural_net.hidden.size = hidden_size;
	neural_net.output.size = OUTPUT_SIZE;

	// 1st hidden layer
	std::cout << "First Hidden Layer -- ";
	init_layer(neural_net.hidden, INPUT_SIZE);

	// output layer
	std::cout << "Output Layer --";
//	init_layer(neural_net.output, neural_net.hidden.size);
	init_layer(neural_net.output, INPUT_SIZE);

}

void init_layer(Layer &layer, int input_size){

	layer.neuron = new Neuron[layer.size];

	// seed random number generator
	srand(time(NULL));

	std::cout << "initializing weights... " ;
	for (int i = 0; i < layer.size; ++i){		// for each neuron
		layer.neuron[i].weight = new double[input_size];
		layer.neuron[i].prev_delta_weight = new double[input_size];
		for (int j = 0; j < input_size; ++j){	// for each weight per neuron
			layer.neuron[i].weight[j] = (double)(rand() % 100) / 100 - 0.5;
			layer.neuron[i].prev_delta_weight[j] = 0.0;
		}
		layer.neuron[i].bias = (double)(rand() % 100) / 100 - 0.5;
		layer.neuron[i].bias_prev_delta_weight = 0.0;
		
	}

#ifdef SHOW_WEIGHTS
	for (int i = 0; i < layer.size; ++i){
		std::cout << "neuron " << i << endl;


		for (int j = 0; j < input_size; ++j){
			std::cout << layer.neuron[i].weight[j] << ' ';
		}
		std::cout << endl;

	}
#endif
	std::cout << "completed.\n\n";
}

void feed_forward_output(Layer &layer, const Layer &input){

	for (int i = 0; i < layer.size; ++i){		// for each neuron
		layer.neuron[i].activation = layer.neuron[i].bias;
		for (int j = 0; j < input.size; ++j){	// for each input, multiply by weight and sum
			layer.neuron[i].activation += input.neuron[j].activation * layer.neuron[i].weight[j];
		}

		layer.neuron[i].activation = sigmoid(layer.neuron[i].activation);
	}

}

void feed_forward_hidden(Layer &layer, const Data &input){

	for (int i = 0; i < layer.size; ++i){		// for each neuron
		layer.neuron[i].activation = layer.neuron[i].bias;
		for (int j = 0; j < INPUT_SIZE; ++j){	// for each input, multiply by weight and sum
			layer.neuron[i].activation += input.value[j] * layer.neuron[i].weight[j];
		}

		layer.neuron[i].activation = sigmoid(layer.neuron[i].activation);
	}

}

void back_prop_output(Layer &layer, const Layer &input, int label, double learning_rate, double momentum_rate, double decay_rate){
	int i = 0;
	double delta_weight = 0.0;

	for (i = 0; i < layer.size; ++i){		// for each neuron
		double target = 0.1;
		if (label == i){
			target = 0.9;
		}

		// this neuron's error
		layer.neuron[i].loss = layer.neuron[i].activation*(1-layer.neuron[i].activation)*(target - layer.neuron[i].activation);

	}

	// error between this layer and previous layer
	for (i = 0; i < input.size; ++i){
		input.neuron[i].connection_error = 0.0;
		for (int j = 0; j < layer.size; ++j){
			input.neuron[i].connection_error += layer.neuron[j].loss*layer.neuron[j].weight[i];
		}
	}

	for (i = 0; i < layer.size; ++i){
		layer.neuron[i].loss *= learning_rate;
		// update weights
		for (int j = 0; j < input.size; ++j){
			delta_weight = layer.neuron[i].loss*input.neuron[j].activation;
			layer.neuron[i].weight[j] += delta_weight + momentum_rate * layer.neuron[i].prev_delta_weight[j] - decay_rate*layer.neuron[i].prev_delta_weight[j];
			layer.neuron[i].prev_delta_weight[j] = delta_weight;
		}

		// update bias
		delta_weight = layer.neuron[i].loss;
		layer.neuron[i].bias += delta_weight + momentum_rate * layer.neuron[i].bias_prev_delta_weight - decay_rate * layer.neuron[i].bias_prev_delta_weight;
		layer.neuron[i].bias_prev_delta_weight = delta_weight;
		
	}
}

void back_prop_hidden(Layer &layer, const Data &input, const Layer &output, double learning_rate, double momentum_rate, double decay_rate){
	double delta_weight = 0.0;

	for (int i = 0; i < layer.size; ++i){		// for each neuron

		// this neuron's error
		layer.neuron[i].loss = learning_rate*layer.neuron[i].activation * (1 - layer.neuron[i].activation) * layer.neuron[i].connection_error;
		
		// update weights
		for (int j = 0; j < INPUT_SIZE; ++j){	
			delta_weight = layer.neuron[i].loss * input.value[j];
			layer.neuron[i].weight[j] += delta_weight + momentum_rate *layer.neuron[i].prev_delta_weight[j] - decay_rate * layer.neuron[i].prev_delta_weight[j];
			layer.neuron[i].prev_delta_weight[j] = delta_weight;
		}

		// update bias
		delta_weight = layer.neuron[i].loss;
		layer.neuron[i].bias += delta_weight + momentum_rate * layer.neuron[i].bias_prev_delta_weight - decay_rate * layer.neuron[i].bias_prev_delta_weight;
		layer.neuron[i].bias_prev_delta_weight = delta_weight;

	}
}

double sigmoid(double input){
	return 1 / (1 + exp(-input));
}

int softmax(const Layer &layer){
	int output = 0;
	double max_activation = layer.neuron[0].activation;
	for (int i = 1; i < OUTPUT_SIZE; ++i){		// find neuron with highest activation
		if (layer.neuron[i].activation > max_activation){
			output = i;
			max_activation = layer.neuron[i].activation;
		}
	}

	return output;
}

// helper functions
void load_csv(Data * &data, string data_file, int size){
	
	// open the data file
	ifstream csv_file(data_file);
	
	if (!csv_file.is_open()){
		std::cout << "Error opening " << data_file;

		exit(1);
	}

	std::cout << "Loading data from " << data_file << endl;
	for (int i = 0; i < size; ++i){
		// first element of row is the label
		csv_file >> data[i].label;
		csv_file.ignore(1);

		// allocate memory for the data item's values
		data[i].value = new double[INPUT_SIZE];

		// load in the values for the data item
		for (int j = 0; j < INPUT_SIZE; ++j){
			csv_file >> data[i].value[j];
			data[i].value[j] /= MAX_VALUE;
			csv_file.ignore(1);	// ignore comma or end of line char
		}
	}

#ifdef SHOW_DATA
	for (int i = 0; i < size; ++i){
		std::cout << data[i].label << ": ";
		for (int j = 0; j < INPUT_SIZE; ++j){
			std::cout << data[i].value[j] << ' ';
		}
		std::cout << endl;
	}
#endif
	
	csv_file.close();
	std::cout << data_file << " loaded." << endl;
}

void init_train_order(int * train_order){

	int temp = 0;
	int swap = 0;

	// seed random number generator
	srand(time(NULL));

	// init training data order
	for (int i = 0; i < TRAIN_SIZE; ++i){
		train_order[i] = i;
	}

#ifdef RANDOMIZE
	for (int i = 0; i < TRAIN_SIZE; ++i){

		temp = train_order[i];
		swap = rand() % TRAIN_SIZE;
		train_order[i] = train_order[swap];
		train_order[swap] = temp;
	}
#endif

#ifdef DEBUG
	std::cout << "Training Order: ";
	for (int i = 0; i < TRAIN_SIZE; ++i){
		std::cout << train_order[i] << ' ';
	}
	std::cout << endl;
#endif

}

double evaluate(int * preds, Data * data, int size){
	
	// accuracy variables
	int correct = 0; // numerator
	int total = 0;	// denominator
	
	// instantiate confusion matrix
	int ** confusion_matrix = new int*[OUTPUT_SIZE];
	for (int i = 0; i < OUTPUT_SIZE; ++i){
		confusion_matrix[i] = new int[OUTPUT_SIZE];
	}

	// initialize confusion matrix
	for (int i = 0; i < OUTPUT_SIZE; ++i){
		for (int j = 0; j < OUTPUT_SIZE; ++j){
			confusion_matrix[i][j] = 0;
		}
	}

#ifndef DEMO
	std::cout << "Confusion Matrix" << endl;
#endif

	for (int i = 0; i < size; ++i){
		++confusion_matrix[data[i].label][preds[i]];
	}

	// display confusion matrix
	for (int i = 0; i < OUTPUT_SIZE; ++i){
		for (int j = 0; j < OUTPUT_SIZE; ++j){
			total += confusion_matrix[i][j];
			if (i == j){	// diagonals are correct values
				correct += confusion_matrix[i][j];
			}
#ifndef DEMO
			std::cout << right << setw(6) << confusion_matrix[i][j];
#endif
		}
#ifndef DEMO
		std::cout << endl << endl;
#endif
	}

	std::cout << endl;

	return (float)correct / total;
}

int save_accuracy(double * train_acc, double * test_acc, int num_epochs, double learning_rate){
	
	string fileName = TRAIN_ACC + to_string(learning_rate) + ".csv";
	ofstream outFile(fileName);

	for (int i = 0; i < num_epochs; ++i){
		outFile << train_acc[i] << '\n';
	}

	outFile.close();
	std::cout << "Training accuracies saved to "<< fileName << endl;

	fileName = TEST_ACC + to_string(learning_rate) + ".csv";
	outFile.open(fileName);

	for (int i = 0; i < num_epochs; ++i){
		outFile << test_acc[i] << '\n';
	}

	outFile.close();
	std::cout << "Testing accuracies saved to " << fileName << endl << endl;

	return 1;
}

void print_number(Data * test_data, int * test_preds){

	int number = rand() % TEST_SIZE;
	int count = 0;
	for (int i = 0; i < 28; ++i){
		for (int j = 0; j < 28; ++j){
			

			if (test_data[number].value[count++] == 0.0){

				std::cout << " ";
			}
			else if (test_data[number].value[count] < 0.5){
				std::cout << "/";
			}
			else{
				std::cout << "#";
			}

		}
		std::cout << endl;
	}

	std::cout << endl << "Prediction is: " << test_preds[number] << endl << endl;

	Sleep(1000);
	return;
}