/**
 * Author: Andrew Bradberry
 * Student ID: 102-43-710
 * Date Written: 10/7/2018
 * Assignment #2: MNIST with Neural Network
 * Description: A basic container for two-dimensional matrices of doubles.
**/

import java.io.Serializable;
import java.util.Random;

public class Matrix implements Serializable{
	private static final long serialVersionUID = 1L;
	
	private int height;
	private int width;
	
	private double[][] matrix;
	
	/**
	 * Create empty matrix of given height and width.
	 * @param height
	 * @param width
	 */
	public Matrix(int height, int width)
	{
		this.height = height;
		this.width = width;
		
		matrix = new double[height][width];
	}
	
	/**
	 * Create matrix that contains the double[][].
	 * @param matrix
	 */
	public Matrix(double[][] matrix)
	{
		this.height = matrix.length;
		this.width = matrix[0].length;
		
		this.matrix = matrix;
	}
	
	/**
	 * Populate the matrix with random values uniformly distributed from -1 to 1.
	 * @param random
	 */
	public void populateRandom(Random random)
	{
		for(int i = 0; i < getHeight(); i++)
		{
			for(int j = 0; j < getWidth(); j++)
			{
				// By default, random.nextDouble will be uniformly distributed form 0 to 1.
				setElement(i, j,  random.nextDouble() * 2 - 1);
			}
		}
	}
	
	/**
	 * Get matrix element at height, width.
	 * @param height
	 * @param width
	 * @return
	 */
	public double getElement(int height, int width)
	{
		return matrix[height][width];
	}
	
	/**
	 * Set matrix element at height, width.
	 * @param height
	 * @param width
	 * @param value
	 * @return
	 */
	public void setElement(int height, int width, double value)
	{
		matrix[height][width] = value;
	}
	
	public int getHeight()
	{
		return height;
	}
	
	public int getWidth()
	{
		return width;
	}
	
	/**
	 * Returns a string representation of the matrix
	 */
	public String toString()
	{
		String pString = "[";
		
		for(int i = 0; i < height; i++)
		{
			pString += "[";
			for(int j = 0; j < width; j++)
			{
				if(j < width - 1)
					pString += matrix[i][j] + ", ";
				else if(i < height - 1)
					pString += matrix[i][j] + "]\n";
				else
					pString += matrix[i][j] + "]]";
			}
		}
		
		return pString;
	}
}
