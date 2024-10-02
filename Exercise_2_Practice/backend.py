import numpy as np

# Task a)
# Implement a method, calculating the LU factorization of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: Matrices P, L and U - same shape as A each.
def lu(A):
  P = np.identity(A.shape[0])
  U = np.copy(A).astype('float64')
  L = np.zeros(shape=(A.shape[0], A.shape[1]))
  n = A.shape[0]                                  #Nr. of rows
  
  #Iterate over pivots
  for i in range(n):
    #Set diagonal of L to 1
    L[i,i] = 1
    
    ###########Partial Pivoting###########
    pivot = U[i,i]
    
    #Check for largest absolute value in column
    for j in range(i+1, n):
        if abs(U[j,i]) > abs(pivot):
            pivot = U[j,i]
            
            #Swap rows in U and P matrix
            U[[i,j],:] = U[[j,i],:]
            P[[i,j],:] = P[[j,i],:]
    
    ###########Elimination Step###########
    #Iterate over rows below pivot
    for j in range(i+1, n):
        #Check if pivot is zero --> Skip
        if pivot == 0:
          break
        
        #Eliminate zeros below pivot
        factor = U[j,i] / pivot
        U[j, :] -= factor * U[i, :]
        
        #Add factor to L matrix
        L[j,i] = factor
    
  return P, L, U

# Task b)
# Implement a method, calculating the determinant of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: The determinant - a floating number
def determinant(A):
  P, L, U = lu(A)
  detU = 1
  r = 0
  
  #Calculate Determinant of U and row interchanges made to A
  for i in range(U.shape[0]):
    detU *= U[i,i]
    
    if P[i,i] != 1:
      r += 1
    
  det = (-1)**r * detU
  
  return det