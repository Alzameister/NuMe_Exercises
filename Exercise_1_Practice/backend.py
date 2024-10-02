import numpy as np

# Task a)
# Implement the gaussian elimination method, to solve the given system of linear equations;
# Add partial pivoting to increase accuracy and stability of the solution;
# Return the solution for x
# Assume a square matrix
def gaussianElimination(A, b):
  n = len(b)
  
  #Iterate over pivot positions
  for i in range(n):
    #Partial pivoting
    #Iterate over row indeces below pivot
    for j in range(i+1, n):
      if abs(A[j,i]) > abs(A[i,i]):
        #Iterate over column indeces
        for k in range(i, n):
          #Swap values so highest absolute value is in pivot position
          A[i,k], A[j,k] = A[j,k], A[i,k]
        b[i], b[j] = b[j], b[i]
    
    pivot = A[i,i]
    #Iterate over columns
    if pivot != 0:
      #Normalize pivot row by dividing the entire row by pivot value
      for j in range(i, n):
        A[i,j] /= pivot
      b[i] /= pivot
  
    #Elimination Step
    #Iterate over rows
    for j in range(n):
      #Check for zero-column
      if j == i or A[j, i] == 0:
        continue
      
      factor = A[j, i]
      #Iterate over columns
      for k in range(i, n):
        #Create zeros above / below pivot
        A[j,k] -= factor * A[i,k]
      b[j] -= factor * b[i]
  
  return A, b
  

def solveLinearSystem(A, b):
  #Convert arrays from int to float to allow for more precision
  A = np.array(A, float)
  b = np.array(b, float)
  
  reduced_matrix, solution_vector = gaussianElimination(A, b)

  return solution_vector

# Task b)
# Implement a method, checking whether the system is consistent or not;
# Obviously, you're not allowed to use any method solving that problem for you.
# Return either true or false
def isConsistent(A,b):
  reduced_matrix, solution_vector = gaussianElimination(A, b)
  n = len(b)
  
  #Iterate over pivot
  for i in range(n):
    #Check if pivot = 0 and solution != 0
    if reduced_matrix[i,i] == 0 and solution_vector[i] != 0:
      return False  
  return True

# Task c)
# Implement a method to compute the daily amounts of chicken breast, brown rice, black beans and avocado to eat to achieve the daily nutritional intake described in the exercise;
# Return a vector x with the grams of chicken breast, brown rice, black beans and avocado to eat each day.
def solveNutrients(A, b):
  #Multiply the solution by 10 because the matrix is measured per 10g, then round it to full grams
  solution_vector = np.round(solveLinearSystem(A, b) * 10)
  return solution_vector


A = np.array([[2,4,-1,5],
              [-4,-5,3,-8],
              [2,-5,-4,1],
              [-6,0,7,-3]])

b = np.array([-2,1,8,1])

print(gaussianElimination(A, b))