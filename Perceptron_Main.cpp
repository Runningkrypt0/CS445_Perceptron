//	Andrew Blackledge
//	requires mnist datasets in the same directory
//	outputs to output.csv

#include "stdafx.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

double randomFloat() {
	return ((double)std::rand()) / ((double)RAND_MAX);
}

struct Vector {
	float* values;
	int dimensions;
	Vector(int dim) {
		values = (float*)std::malloc(sizeof(float)*dim);
		dimensions = dim;
	}
	float DotProduct(Vector* partner){
		int result = 0;
		int minDimensions = dimensions;

		if (dimensions > partner->dimensions) {
			minDimensions = partner->dimensions;
		}
		
		for (int d = 0; d < minDimensions; d++) {
			result += values[d] * partner->values[d];
		}
		return result;
	}
};

struct Image {
	//associates a class with an input vector
	Vector* values;
	int classification;

	Image(int size, int c) {
		classification = c;
		values = new Vector(size+1);
	}
};

struct ImageSet {
	//stores many images and provides a way of loading them
	std::vector<Image*> members;
	ImageSet(std::string fileName, int inputCount) {

		std::string heldLine;
		std::string::size_type tracker;
		std::ifstream file;

		file.open(fileName);
		if (!file.is_open()) {
			std::cout << "\nCould not open image set\n";
		}
		/*while(std::getline(file, heldLine)){
			Image* joiningMember = new Image(inputCount, std::stoi(heldLine, &tracker));

			for (int x = 0; x < inputCount; x++) {
				heldLine = heldLine.substr(++tracker);
				float value = std::stof(heldLine, &tracker) / 255.0;
				joiningMember->values->values[x] = value;
			}

			joiningMember->values->values[inputCount] = 1;

			members.push_back(joiningMember);
		}*/

		while (std::getline(file, heldLine, ',')) {
			Image* joiningMember = new Image(inputCount, std::stoi(heldLine));

			for (int x = 0; x < inputCount-1; x++) {
				std::getline(file, heldLine, ',');
				float value = std::stof(heldLine) / 255;
				joiningMember->values->values[x] = value;
			}

			std::getline(file, heldLine);
			float value = std::stof(heldLine) / 255;
			joiningMember->values->values[inputCount-1] = value;

			joiningMember->values->values[inputCount] = 1;

			members.push_back(joiningMember);

			
		}


	}
};

struct Perceptron {
	//stores a weight vector and provides perceptron methods for that vector
private:
	Vector* weights;

public:
	Perceptron(int size) {
		weights = new Vector(size + 1);
	}
	void seed(float range) {
		for (int v = 0; v < weights->dimensions; v++) {
			weights->values[v] = (randomFloat() * 2 - 1)*range;
		}
	}
	void train(Vector* target, bool positive, float rate) {
		//stochasticly trains from an input vector
		float result = weights->DotProduct(target);

		float error = 0;
		if (result > 0) {
			error--;
		}
		if (positive) {
			error++;
		}

		if (error == 0) {
			return;
		}

		float coef = error*rate;

		for (int v = 0; v < weights->dimensions; v++) {
			weights->values[v] += coef*target->values[v];
		}
	}
	float test(Vector* target){
		//returns the output of this Perceptron
		return weights->DotProduct(target);
	}
};

struct PerceptronSet {
	//stores many Perceptrons as disjoint classes, and provides multi-class classification methods
	Perceptron** members;
	int classCount;
	PerceptronSet(int c, int inputCount) {
		classCount = c;
		members = (Perceptron**)malloc(sizeof(Perceptron*)*classCount);
		for (int v = 0; v < classCount; v++) {
			members[v] = new Perceptron(inputCount);
		}
	}
	void seed(float range) {
		for (int v = 0; v < classCount; v++) {
			members[v]->seed(range);
		}
	}
	void train(ImageSet* target, float rate) {
		//stochasticly trains all Perceptrons from an ImageSet
		int size = target->members.size();

		//create a shuffled index array
		int* indexSet = (int*)malloc(sizeof(int)*size);
		for (int v = 0; v < size; v++) {
			indexSet[v] = v;
		}

		std::random_shuffle(indexSet, indexSet + size);

		for (int v = 0; v < size; v++) {
			Image* next = target->members[indexSet[v]];
			for (int x = 0; x < classCount; x++) {
				members[x]->train(next->values, next->classification == x, rate);
			}
		}

	}
	int classify(Image* target) {
		//returns the most likely class of a single Image
		float highestScore = 0;
		int scorer = -1;
		for (int v = 0; v < classCount; v++) {
			float score = members[v]->test(target->values);
			if (score > highestScore) {
				highestScore = score;
				scorer = v;
			}
		}
		return scorer;
	}
	float test(ImageSet* target) {
		//evaluates the accuraccy of the classifier for an ImageSet
		int correct = 0;

		for (int v = 0; v < target->members.size(); v++) {
			if (classify(target->members[v]) == target->members[v]->classification) {
				correct++;
			}
		}

		return (float)correct / (float)target->members.size();
	}
	int* matrix(ImageSet* target) {
		//constructs a confusion matrix for an ImageSet
		int* result = (int*)malloc(sizeof(int) * 10 * 10);
		for (int v = 0; v < 100; v++) {
			result[v] = 0;
		}
		for (int v = 0; v < target->members.size(); v++) {
			int predicted = classify(target->members[v]);
			int actual = target->members[v]->classification;
			result[actual * 10 + predicted]++;
		}
		return result;
	}
};

float* TrainingCycle(PerceptronSet* classifiers, ImageSet* trainer, ImageSet* tester, int epochCount, float rate, std::ofstream& output) {
	//stochasticly trains a classifier and stores the accuracy after every epoch
	classifiers->seed(.5);

	float* accuracy = (float*)malloc(sizeof(float)*(epochCount+1)*2);
	for (int v = 0; v < epochCount; v++) {
		accuracy[v*2] = classifiers->test(trainer);
		accuracy[v*2+1] = classifiers->test(tester);

		std::cout << v << " - " << accuracy[v * 2] << " : " << accuracy[v * 2 + 1] << "\n";
		output << accuracy[v * 2] << "," << accuracy[v * 2 + 1] << "\n";

		classifiers->train(trainer, rate);
	}

	accuracy[epochCount * 2] = classifiers->test(trainer);
	accuracy[epochCount * 2 + 1] = classifiers->test(tester);
	std::cout << "final - " << accuracy[epochCount * 2] << " : " << accuracy[epochCount * 2 + 1] << "\n";
	output << accuracy[epochCount * 2] << "," << accuracy[epochCount * 2 + 1] << "\n";

	return accuracy;
}

void DisplayMatrix(int* matrix, std::ofstream& output) {
	std::cout << "\n";
	for (int x = 0; x < 10; x++) {
		for (int y = 0; y < 10; y++) {
			std::cout << matrix[x * 10 + y] << " ";
			output << matrix[x * 10 + y] << ",";
		}
		std::cout << "\n";
		output << "\n";
	}
	std::cout << "\n";
	output << "\n";
}

int main()
{
	//###########################
	//########## SETUP ##########
	//###########################

	int epochCount = 50;

	int classCount = 10;
	int inputCount = 764;

	std::cout << "loading data...\n";
	std::ofstream output;
	output.open("output.csv");
	ImageSet trainSet("mnist_train.csv", inputCount);
	ImageSet testSet("mnist_test.csv", inputCount);
	PerceptronSet classifiers(classCount, inputCount);
	
	std::cout << "-----Starting LR=.001 cycle-----\n";
	float* thousandthAccuracy = TrainingCycle(&classifiers, &trainSet, &testSet, epochCount, .001, output);
	std::cout << "-----Constructing confusion matrix-----\n";
	int* thousandthMatrix = classifiers.matrix(&testSet);
	DisplayMatrix(thousandthMatrix, output);
	std::cout << "-----DONE-----\n\n";

	std::cout << "-----Starting LR=.01 cycle-----\n";
	float* hundredthAccuracy = TrainingCycle(&classifiers, &trainSet, &testSet, epochCount, .01, output);
	std::cout << "-----Constructing confusion matrix-----\n";
	int* hundredthMatrix = classifiers.matrix(&testSet);
	DisplayMatrix(hundredthMatrix, output);
	std::cout << "-----DONE-----\n\n";

	std::cout << "-----Starting LR=.1 cycle-----\n";
	float* tenthAccuracy = TrainingCycle(&classifiers, &trainSet, &testSet, epochCount, .1, output);
	std::cout << "-----Constructing confusion matrix-----\n";
	int* tenthMatrix = classifiers.matrix(&testSet);
	DisplayMatrix(tenthMatrix, output);
	std::cout << "-----DONE-----\n\n";

	output.close();

	//stall out so I can actually read the info
	std::cin >> epochCount;

    return 0;
}

