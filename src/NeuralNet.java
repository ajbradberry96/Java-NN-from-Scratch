/**
 * Author: Andrew Bradberry
 * Date Written: 10/7/2018
 * Description: A basic multi-layer neural network using the sigmoid activation function,
 * that has variable size.
**/

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Collections;
import java.util.Random;

public class NeuralNet implements Serializable{
	private static final long serialVersionUID = 1L;
	
	private int numInputs;
	private int numClasses;
	// DOES NOT INCLUDE INPUT LAYER.
	private int numLayers;
	private int nodesInHL;

	private double learningRate = 3;
	private int batchSize = 10;
	private int numEpochs = 30;
	
	private Random random;
	// For control over randomness
	private static final long RANDOM_SEED = 1111;
	
	// In form Layer, to, from
	private Matrix[] weights;
	// Vector in form Layer, node
	private Matrix[] biases;
	
	/**
	 * Constructs a random, untrained neural network.
	 * @param numInputs Number of input neurons. This is the number of traits of the data.
	 * @param numClasses Number of output neurons. The number of classifications in the data.
	 * @param numLayers The number of layers of the network. DOES NOT INCLUDE INPUT LAYER.
	 * @param nodesInHL The size of each layer in the hidden layers.
	 */
	public NeuralNet(int numInputs, int numClasses, int numLayers, int nodesInHL)
	{
		this.numInputs = numInputs;
		this.numClasses = numClasses;
		this.numLayers = numLayers;
		this.nodesInHL = nodesInHL;
		
		random = new Random();
		random.setSeed(RANDOM_SEED);
		
		constructRandomNet();
	}
	
	/**
	 * Train the neural network for numEpochs epochs with mini-batch size of batchSize.
	 * Uses basic stochastic gradient descent and backpropegation.
	 * @param trainingData The training data, where each matrix is {X, Y}
	 * @param testingData The testing data, where each matrix is {X, Y}
	 */
	public void train(ArrayList<Matrix[]> trainingData, ArrayList<Matrix[]> testingData)
	{
		Collections.shuffle(trainingData);
		
		ArrayList<ArrayList<Matrix[]>> batches = new ArrayList<>();
		
		// Split data into mini-batches of batchSize, with the last one being potentially smaller if 
		// the size of the training data is not a multiple of batchSize.
		for(int batch = 0; trainingData.size() - batch * batchSize > 0; batch++)
		{
			ArrayList<Matrix[]> currentBatch = new ArrayList<>();
			
			for(int i = batch * batchSize; i < (batch + 1) * batchSize && i < trainingData.size(); i++)
			{
				currentBatch.add(trainingData.get(i));
			}

			batches.add(currentBatch);
		}
		
		// Actual training time
		for(int epoch = 0; epoch < numEpochs; epoch++)
		{
			for(int batch = 0; batch < batches.size(); batch++)
			{
				// Grab the first input and output
				Matrix input = batches.get(batch).get(0)[0];
				Matrix y = batches.get(batch).get(0)[1];
				
				// gradsToAdd is a placeholder that holds this input's weightGradient and biasGradient.
				Matrix[][] gradsToAdd = backpropegate(feedForward(input), y);
				
				// weightGradient and biasGradient will hold the sums of the weight and bias gradients for
				// every input in the batch.
				Matrix[] weightGradient = gradsToAdd[0];
				Matrix[] biasGradient = gradsToAdd[1];
				
				// Add the gradients from the rest of the inputs in the batch.
				for(int x = 1; x < batches.get(batch).size(); x++)
				{
					input = batches.get(batch).get(x)[0];
					y = batches.get(batch).get(x)[1];
					
					gradsToAdd = backpropegate(feedForward(input), y);
					
					for(int i = 0; i < gradsToAdd[0].length; i++)
					{
						weightGradient[i] = MatrixUtil.matAdd(weightGradient[i], gradsToAdd[0][i]);
					}
					for(int i = 0; i < gradsToAdd[1].length; i++)
					{
						biasGradient[i] = MatrixUtil.matAdd(biasGradient[i], gradsToAdd[1][i]);
					}
				}
				
				// Modify the weights and biases using matrix formulas, where i represents the iteration:
				// w_i = w_(i-1) + (-learningRate / batchSize) * weightGradient
				// b_i = b_(i-1) + (-learningRate / batchSize) * biasGradient
				for(int i = 0; i < weightGradient.length; i++)
				{
					weights[i] = MatrixUtil.matAdd(weights[i], MatrixUtil.matMul(-learningRate / batches.get(batch).size(),  weightGradient[i]));
				}
				for(int i = 0; i < biasGradient.length; i++)
				{
					biases[i] = MatrixUtil.matAdd(biases[i], MatrixUtil.matMul(-learningRate / batches.get(batch).size(),  biasGradient[i]));
				}
			}
			
			// Display current training status
			System.out.println("\nEpoch over\n");
			
			printAccuracy(trainingData, "Training");
		}
		
		printAccuracy(testingData, "Testing");
	}
	
	/**
	 * Feeds an input through the network and obtains the activations of each neuron.
	 * @param x The input vector
	 * @return The activations of each layer of the network (including the input layer)
	 */
	private Matrix[] feedForward(Matrix x)
	{
		Matrix[] a = new Matrix[numLayers + 1];
		
		// The first activation is simply the input vector
		a[0] = x;

		// Between hidden layers and to output
		for(int layer = 1; layer <= numLayers; layer++)
		{
			// Uses matrix formula where i represents the layer:
			// a_(i+1) = sigmoid(w_i * a_i + b_i)
			a[layer] = applyActivationFunction(MatrixUtil.matAdd(biases[layer - 1], MatrixUtil.matMul(weights[layer - 1], a[layer - 1])));
		}

		return a;
	}
	
	/**
	 * Obtains the weight and bias gradients of a certain input based upon the activations from 
	 * a feedForward pass and the intended output.
	 * @param a The activations of each neuron based upon a feedForward pass
	 * @param y The one-hot encoded correct classification based upon the input.
	 * @return A matrix tuple that represents {weightGradient, biasGradient} of the input
	 */
	private Matrix[][] backpropegate(Matrix[] a, Matrix y)
	{
		Matrix[] weightGradients = new Matrix[numLayers];
		Matrix[] biasGradients = new Matrix[numLayers];
		
		// Final layer
		Matrix aL = a[numLayers];
		Matrix difference = MatrixUtil.matSub(aL, y);
		Matrix sigPrime = MatrixUtil.pairwiseMul(aL, MatrixUtil.matSub(MatrixUtil.ones(aL.getHeight(), aL.getWidth()), aL));
		
		// Uses equation for error in the output layer
		biasGradients[numLayers - 1] = MatrixUtil.pairwiseMul(difference, sigPrime);
		// Uses equation for rate of change of cost with respect to any weight
		weightGradients[numLayers - 1] = MatrixUtil.matMul(biasGradients[numLayers - 1], MatrixUtil.transpose(a[numLayers - 1]));
		
		// Every other layer
		for(int layer = numLayers - 2; layer >= 0; layer--)
		{
			aL = a[layer + 1];
			Matrix product = MatrixUtil.matMul(MatrixUtil.transpose(weights[layer + 1]), biasGradients[layer + 1]);
			sigPrime = MatrixUtil.pairwiseMul(aL, MatrixUtil.matSub(MatrixUtil.ones(aL.getHeight(), aL.getWidth()), aL));
			
			// Uses equation for error in non-output layer
			biasGradients[layer] = MatrixUtil.pairwiseMul(sigPrime, product);
			// Uses equation for rate of change of cost with respect to any weight
			weightGradients[layer] = MatrixUtil.matMul(biasGradients[layer], MatrixUtil.transpose(a[layer]));
		}

		return new Matrix[][]{weightGradients, biasGradients};
	}
	
	/**
	 * Return the activations of just the output layer from input x.
	 * @param x The input vector
	 * @return The activation vector.
	 */
	public Matrix getPrediction(Matrix x)
	{
		Matrix[] result = feedForward(x);
		return result[result.length - 1];
	}
	
	/**
	 * Apply the activation function to every member of matrix x.
	 * Here, the activation function is the sigmoid function.
	 * @param x 
	 * @return x with every element set to sigmoid(x(element))
	 */
	public Matrix applyActivationFunction(Matrix x)
	{
		for(int i = 0; i < x.getHeight(); i++)
		{
			for(int j = 0; j < x.getWidth(); j++)
			{
				x.setElement(i, j, activationFunction(x.getElement(i, j)));
			}
		}
		
		return x;
	}
	
	/**
	 * The sigmoid activation function
	 * @param x input
	 * @return sigmoid(x)
	 */
	private double activationFunction(double x)
	{
		return 1 / (1 + Math.exp(-x));
	}
	
	/**
	 * Prints the accuracy of the classification from the neural network,
	 * both in terms of each class index, and overall.
	 * @param dataset The dataset you want to classify
	 * @param title The name of the dataset
	 */
	public void printAccuracy(ArrayList<Matrix[]> dataset, String title)
	{
		System.out.println("Class Accuracy:");
		
		HashMap<Integer, Integer> correct = new HashMap<>();
		HashMap<Integer, Integer> total = new HashMap<>();
		
		for(int i = 0; i < numClasses; i++)
		{
			correct.put(i, 0);
			total.put(i, 0);
		}
		
		for(Matrix[] datum : dataset)
		{
			if(MatrixUtil.maxElement(datum[1]) == MatrixUtil.maxElement(getPrediction(datum[0])))
			{
				correct.put(MatrixUtil.maxElement(datum[1]), correct.get(MatrixUtil.maxElement(datum[1])) + 1);
			}
			total.put(MatrixUtil.maxElement(datum[1]), total.get(MatrixUtil.maxElement(datum[1])) + 1);
		}
		
		int totalCorrect = 0;
		for(int i = 0; i < numClasses; i++)
		{
			totalCorrect += correct.get(i);
			System.out.println(i + ": " + correct.get(i) + " / " + total.get(i) + " = " + (double)correct.get(i) / (double)total.get(i));
		}
		
		System.out.println(title + " Accuracy: " + totalCorrect + " / " + dataset.size() + " = " + (double)totalCorrect / (double)dataset.size());
	}
	
	/**
	 * Construct neural network based upon current specs.
	 * Set all weights and biases to random values.
	 */
	private void constructRandomNet()
	{
		weights = new Matrix[numLayers];
		biases = new Matrix[numLayers];
		
		// Add weights and biases to first layer from input layer
		Matrix w_ph = new Matrix(nodesInHL, numInputs);
		w_ph.populateRandom(random);
		
		weights[0] = w_ph;
		
		Matrix b_ph = new Matrix(nodesInHL, 1);
		b_ph.populateRandom(random);
		
		biases[0] = b_ph;
		
		// Add weights and biases for any other hidden layers
		for(int layer = 1; layer < numLayers - 1; layer++)
		{
			w_ph = new Matrix(nodesInHL, nodesInHL);
			w_ph.populateRandom(random);
			
			weights[layer] = w_ph;
			
			b_ph = new Matrix(nodesInHL, 1);
			b_ph.populateRandom(random);
			
			biases[layer] = b_ph;
		}
		
		// Add weights and biases for final layer
		w_ph = new Matrix(numClasses, nodesInHL);
		w_ph.populateRandom(random);
		
		weights[numLayers - 1] = w_ph;
		
		b_ph = new Matrix(numClasses, 1);
		b_ph.populateRandom(random);
		
		biases[numLayers - 1] = b_ph;
	}

	/**
	 * Obtain facts about the network,
	 * currently just the number of layers and the size of the hidden layers
	 * @return A string (to be printed) with all the fun facts you want,
	 * I guess.
	 */
	public String getStuff() {
		return "Number of layers (including input layer): " + (numLayers + 1) + "\nSize of hidden layers: " + nodesInHL;
	}
}
