/**
 * Author: Andrew Bradberry
 * Date Written: 10/7/2018
 * Description: Handles all IO for MNIST assignment, including gathering user-input, displaying information,
 * and saving to files. Currently, uses a command line system.
**/

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Scanner;

public class IOManager {
	Scanner sc = new Scanner(System.in);
	
	/**
	 * Parses the CSVs provided that contain the MNIST dataset.
	 * Stores each digit in those files into dataset, with each
	 * Matrix[] containing {X, Y}
	 * @param csvFile Filename of dataset.
	 * @param dataset Dataset to have digits stored in.
	 * @throws IOException You probably messed up the file extension, lol
	 */
	public void parseCSV(String csvFile, ArrayList<Matrix[]> dataset) throws IOException
	{
        BufferedReader br = new BufferedReader(new FileReader(csvFile));
        String line = "";
        
        // Each line in the csv represents a single digit, with the first value being the classification, and the next 28 * 28 
        // being the intensity values of the pixel, ranging from 0-255.
		while ((line = br.readLine()) != null)
		{
		    String[] digit = line.split(",");
		    
		    Matrix X = new Matrix(28 * 28, 1);
		    Matrix Y = new Matrix(10, 1);
			    
		    // One-hot encoding of the classification vector
		    for(int i = 0; i < 10; i++)
		    {
		    	if(i == Integer.parseInt(digit[0]))
		    	{
		    		Y.setElement(i, 0, 1);
		    	}
		    	else
		    	{
		    		Y.setElement(i, 0, 0);
		    	}
		    }
		    
		    for(int i = 0; i < 28 * 28; i++)
		    {
		    	// Value is divided by 255 to put into the range 0-1
		    	X.setElement(i, 0, Double.parseDouble(digit[i + 1]) / 255);
		    }
		    
		    dataset.add(new Matrix[] {X, Y});

		}
		br.close();
	}

	/**
	 * Get input from the user
	 * @return The requested input.
	 */
	public String getInput() {
		return sc.next();
	}

	/**
	 * Store the network passed to a file using object serialization.
	 * @param filename File to save to.
	 * @param net Network to be saved.
	 * @throws IOException You probably messed up the filename, lol
	 */
	public void save(String filename, NeuralNet net) throws IOException
	{
        // Saving of object in a file 
        FileOutputStream file = new FileOutputStream(filename); 
        ObjectOutputStream out = new ObjectOutputStream(file); 

        // Method for serialization of object 
        out.writeObject(net); 

        out.close(); 
        file.close(); 
	}
	
	/**
	 * Reads a stored network from a file using object serialization.
	 * @param filename File to save to.
	 * @return The loaded neural net.
	 * @throws IOException You probably messed up the filename, lol
	 */
	public NeuralNet load(String filename) throws IOException, ClassNotFoundException
	{
        FileInputStream file = new FileInputStream(filename); 
        ObjectInputStream in = new ObjectInputStream(file); 
        
        NeuralNet net = (NeuralNet)in.readObject(); 

        in.close(); 
        file.close(); 
        
        return net;
	}

	/**
	 * Print out the digit using "Ascii art"
	 * @param digit The digit to be displayed
	 * @param threshold The value for which an element of digit needs to exceed to be displayed.
	 */
	public void displayDigit(Matrix[] digit, double threshold) {
		System.out.println("\n");
		String toDisplay = "";
		
		// For each pixel
		for(int i = 0; i < 28 * 28; i++)
		{
			// Display as a '$' if the pixel intensity is over threshold, ' ' otherwise
			if(digit[0].getElement(i, 0) > threshold)
			{
				toDisplay += "$";
			}
			else
			{
				toDisplay += " ";
			}
			
			// If this is the last pixel in the row, add a newline
			if (((i + 1) % 28) == 0)
			{
				toDisplay += "\n";
			}
		}
		System.out.println(toDisplay);
	}
	
	/**
	 * Prints a string to System.out
	 * @param toDisplay The string to be displayed. 
	 */
	public void display(String toDisplay)
	{
		System.out.println(toDisplay);
	}
}
