/**
 * Author: Andrew Bradberry
 * Date Written: 10/7/2018
 * Description: Static class which handles all mathematical Matrix operations. 
**/

public class MatrixUtil {
	/**
	 * Pairwise adds the two matrices together, if they are of the same dimension in the following form:
	 * [[a, b]  + [[e, f]  = [[a + e, b + f]
	 *  [c, d]]    [g, h]]    [c + g, d + h]]
	 * @param a
	 * @param b
	 * @return A matrix with the same dimensions as the input matrices, a + b.
	 * @throws ArithmeticException
	 */
	public static Matrix matAdd(Matrix a, Matrix b) throws ArithmeticException
	{
		// Check to ensure matrix sizes match.
		if (a.getWidth() != b.getWidth() || a.getHeight() != b.getHeight())
		{
			throw new ArithmeticException("Invalid matrix sizes: [" + a.getHeight() + ", " + a.getWidth() + "], [" + b.getHeight() + ", " + b.getWidth() + "]");
		}
		
		Matrix c = new Matrix(a.getHeight(), a.getWidth());
		
		for(int i = 0; i < a.getHeight(); i++)
		{
			for(int j = 0; j < a.getWidth(); j++)
			{
				c.setElement(i, j, a.getElement(i, j) + b.getElement(i, j));
			}
		}
		
		return c;
	}
	
	/**
	 * Pairwise subtracts the two matrices, if they are of the same dimension in the following form:
	 * [[a, b]  - [[e, f]  = [[a - e, b - f]
	 *  [c, d]]    [g, h]]    [c - g, d - h]]
	 * @param a
	 * @param b
	 * @return A matrix with the same dimensions as the input matrices, a - b.
	 * @throws ArithmeticException
	 */
	public static Matrix matSub(Matrix a, Matrix b) throws ArithmeticException
	{
		// Multiply b by negative one, and add together. 
		return matAdd(a, matMul(-1, b));
	}
	
	/**
	 * Returns the regular matrix product, assuming legal, of the two matrices provided in the following form:
	 * [[a, b]  * [[e, f]  = [[a * e + b * g, a * f + b * h]
	 *  [c, d]]    [g, h]]    [c * e + d * g, c * f + d * h]]
	 * @param a An m x n matrix
	 * @param b A n x l matrix
	 * @return The product of a * b, an m x l matrix.
	 * @throws ArithmeticException
	 */
	public static Matrix matMul(Matrix a, Matrix b) throws ArithmeticException
	{
		// Check that the multiplication is legal
		if (a.getWidth() != b.getHeight())
		{
			throw new ArithmeticException("Invalid matrix sizes: [" + a.getHeight() + ", " + a.getWidth() + "], [" + b.getHeight() + ", " + b.getWidth() + "]");
		}
		
		Matrix c = new Matrix(a.getHeight(), b.getWidth());
		
		// Vertical position in output matrix
		for(int i = 0; i < a.getHeight(); i++)
		{
			// Horizontal position in output matrix
			for(int j = 0; j < b.getWidth(); j++)
			{
				double sum = 0;
				for(int k = 0; k < a.getWidth(); k++)
				{
					sum += a.getElement(i, k) * b.getElement(k, j);
				}
				
				c.setElement(i, j, sum);
			}
		}
		
		return c;
	}
	
	/**
	 * Multiplies an input matrix by a scalar in the following form:
	 * a * [[m, n]   =  [[a * m, a * n]
	 * 		[o, p]]		 [a * o, a * p]]
	 * @param a The scalar.
	 * @param b The matrix.
	 * @return A matrix with the same dimensions as b.
	 */
	public static Matrix matMul(double a, Matrix b)
	{
		
		Matrix c = new Matrix(b.getHeight(), b.getWidth());
		
		for(int i = 0; i < b.getHeight(); i++)
		{
			for(int j = 0; j < b.getWidth(); j++)
			{
				c.setElement(i, j, b.getElement(i, j) * a);
			}
		}
		
		return c;
	}
	
	/**
	 * Pairwise multiplies the two matrices together, if they are of the same dimension in the following form:
	 * [[a, b]  .* [[e, f]  = [[a * e, b * f]
	 *  [c, d]]     [g, h]]    [c * g, d * h]]
	 * @param a
	 * @param b
	 * @return A matrix with the same dimensions as the input matrices, a .* b.
	 * @throws ArithmeticException
	 */
	public static Matrix pairwiseMul(Matrix a, Matrix b) throws ArithmeticException
	{
		// Check to ensure legal opoeration
		if (a.getWidth() != b.getWidth() || a.getHeight() != b.getHeight())
		{
			throw new ArithmeticException("Invalid matrix sizes: [" + a.getHeight() + ", " + a.getWidth() + "], [" + b.getHeight() + ", " + b.getWidth() + "]");
		}
		
		Matrix c = new Matrix(a.getHeight(), a.getWidth());
		
		for(int i = 0; i < a.getHeight(); i++)
		{
			for(int j = 0; j < a.getWidth(); j++)
			{
				c.setElement(i, j, a.getElement(i, j) * b.getElement(i,j));
			}
		}
		
		return c;
	}
	
	/**
	 * Create a matrix of dimensions height x width such that every element is 1.
	 * @param height
	 * @param width
	 * @return
	 */
	public static Matrix ones(int height, int width)
	{
		Matrix m = new Matrix(height, width);
		
		for(int i = 0; i < m.getHeight(); i++)
		{
			for(int j = 0; j < m.getWidth(); j++)
			{
				m.setElement(i, j, 1);
			}
		}
		
		return m;
	}
	
	/**
	 * Returns the matrix transpose (Rows become columns) of the provided matrix in the following form:
	 * [[a, b, c] T   =  [[a, d]
	 *  [d, e, f]]		  [b, e]
	 *  				  [c, f]]
	 * @param a A n x m matrix
	 * @return The transpose of a, m x n matrix
	 */
	public static Matrix transpose(Matrix a)
	{
		Matrix t = new Matrix(a.getWidth(), a.getHeight());
		
		for(int i = 0; i < a.getHeight(); i++)
		{
			for(int j = 0; j < a.getWidth(); j++)
			{
				t.setElement(j, i, a.getElement(i, j));
			}
		}
		
		return t;
	}
	
	/**
	 * Returns the index of the maximum element of a as if it were a flattened matrix. 
	 * Ex:
	 * a = [[1, 2]		
	 * 		[4, 3]]
	 * flatten(a) = [[1, 2, 4, 3]]
	 * maxElement(a) = 2
	 * 
	 * a = [[1, 2]
	 * 		[5, 3]
	 * 		[1, 4]]
	 * flatten(a) = [[1, 2, 4, 3, 5, 4]]
	 * maxElement(a) = 4
	 * 
	 * @param a
	 * @return The index of the maximum element of a.
	 */
	public static int maxElement(Matrix a)
	{
		int max = 0;
		
		for(int i = 0; i < a.getHeight() * a.getWidth(); i++)
		{
			if(a.getElement(max / a.getWidth(), max % a.getWidth()) < a.getElement(i / a.getWidth(), i % a.getWidth()))
			{
				max = i;
			}
		}
		
		return max;
	}
	
}
