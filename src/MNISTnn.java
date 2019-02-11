/**
 * Author: Andrew Bradberry
 * Date Written: 10/7/2018
 * Description: Implements a basic Multi-Layer neural network in order to classify handwritten digits from the ubiquitous MNIST dataset. 
**/

import java.io.IOException;
import java.util.ArrayList;

public class MNISTnn {
	// Handles all IO, you never know, someday maybe we want this to be a GUI
	private static IOManager io;
	
	// Each member in train and test is {X, Y}
	private static ArrayList<Matrix[]> train;		
	private static ArrayList<Matrix[]> test;
	
	private static NeuralNet net;
	
	// Has the network been trained yet/is this a pre-trained network
	private static boolean trained;
	
	/**
	 * Sets up our environment and starts our "game loop".
	 */
	public static void main(String[] args) {
		io = new IOManager();
		
		train = new ArrayList<>();
		test = new ArrayList<>();
		
		// Grab the training and test sets
		try {
			io.parseCSV("/Users/Moosicguy/Desktop/mnist_train.csv", train);
			io.parseCSV("/Users/Moosicguy/Desktop/mnist_test.csv", test);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// By default, our network doesn't even exist, so of course it isn't trained
		trained = false;
		
		// Effectively our "game loop"
		loop();
	}
	
	/**
	 * This is the body of our program where we get input, and make choices
	 */
	private static void loop()
	{
		// We do this until we quit
		while(true)
		{
			// Display our options
			io.display("\nTo train a new network, press 1.");
			io.display("To load an existing network, press 2.");
			// Only display these if we have a functional network
			if(trained)
			{
				io.display("To display network accuracy on training data, press 3.");
				io.display("To display network accuracy on testing data, press 4.");
				io.display("To save the current network state, press 5.");
				io.display("To learn about this network, press 6.");
				io.display("To walk through samples, press 7.");
			}
			io.display("To quit, press 0\n");
			
			switch (io.getInput())
			{
				case "1":
					createNewNet();
					break;
				case "2":
					loadNet();
					break;
				case "3":
					displayTrainingAcc();
					break;
				case "4":
					displayTestingAcc();
					break;
				case "5":
					saveNet();
					break;
				case "6":
					displayNetStuff();
					break;
				case "7":
					walkThroughSamples();
					break;
				case "0":
					close();
					break;
				default:
					io.display("Invalid Option");
					break;
			}
		}
	}

	/**
	 * Sequentially display digits of the MNIST dataset along with the correct label, and the predicted label
	 */
	private static void walkThroughSamples() {
		io.display("Only walk through missed samples(y/n)?: ");
		
		boolean all = io.getInput().equals("n");
		
		io.display("To walk through training set, press 1. To walk through test set, press 2");
		
		boolean trainOrTest = io.getInput().equals("1");
		
		ArrayList<Matrix[]> dataset;
		if(trainOrTest)
		{
			dataset = train;
		}
		else
		{	
			dataset = test;
		}
		
		for(Matrix[] digit : dataset)
		{
			if(MatrixUtil.maxElement(net.getPrediction(digit[0])) != MatrixUtil.maxElement(digit[1]) || all)
			{
				io.display("Correct Class: " + MatrixUtil.maxElement(digit[1]) + "\tNet Prediction: " + MatrixUtil.maxElement(net.getPrediction(digit[0])));
				io.displayDigit(digit, .5);
				io.display("Press 1 to continue, 2 to return to main menu");
				if(io.getInput().equals("2"))
				{
					break;
				}
			}
		}
	}
	
	/**
	 * Display information about the network, for now just number of layers and size of hidden layers.
	 */
	private static void displayNetStuff() {
		io.display(net.getStuff());
	}

	/**
	 * Quit the program.
	 */
	private static void close() {
		System.exit(0);
	}
	
	/**
	 * Save the current network to a file
	 */
	private static void saveNet() {
		io.display("Filename: ");
		
		try {
			io.save(io.getInput(), net);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Display the class accuracy and overall accuracy using the testing dataset.
	 */
	private static void displayTestingAcc() {
		net.printAccuracy(test, "Testing");
	}

	/**
	 * Display the class accuracy and overall accuracy using the training dataset.
	 */
	private static void displayTrainingAcc() {
		net.printAccuracy(train, "Training");
	}

	/**
	 * Load a network from a file. 
	 */
	private static void loadNet() {
		io.display("Filename: ");
		
		try {
			net = io.load(io.getInput());
			trained = true;
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Create a new network with user-specified number of layers and size of hidden layers. 
	 * Trains the network automatically.
	 */
	private static void createNewNet() {
		io.display("Input the number of layers (Not inluding input layer): ");
		int numLayers = Integer.parseInt(io.getInput());
		io.display("Input the size of the hidden layers: ");
		int sizeOfHL = Integer.parseInt(io.getInput());
		
		net = new NeuralNet(28 * 28, 10, numLayers, sizeOfHL);
		
		io.display("Training...");
		
		net.train(train, test);
		trained = true;
	}
}
