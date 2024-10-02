import numpy as np
import scipy as sp

# Task a)
# Implement a method, calculating the LU factorization of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: Matrices P, L and U - same shape as A each.
def lu(A):
  P = np.identity(A.shape[0])
  U = np.copy(A)
  U = U.astype('float64')
  L = np.zeros(shape=(A.shape[0], A.shape[1]))
  n = A.shape[0]      #Nr. of rows
  cols = A.shape[1]
  
  #Iterate over rows / pivots
  for row in range(n):
    col = row
    pivot = U[row,col]
    
    #Check if pivot is zero
    while pivot == 0 and col < cols:
      #Move one index to the right
      col += 1
      if col >= cols:
        break
      pivot = U[row,col]
    
    if col >= cols:
      continue
    
    ###########Partial Pivoting###########
    #Iterate over rows below pivot
    for j in range(row+1, n):
      if abs(U[j,col]) > abs(U[row,col]):
        #Swap rows in both U and P matrices
        U[[row,j],:] = U[[j,row],:]
        P[[row,j],:] = P[[j,row],:]
        pivot = U[row,col]
        
    #Place pivot column in L and normalize
    L[row:, col] = U[row:, col] / pivot
  
    ###########Elimination Step###########
    #Iterate over rows below pivot
    for j in range(row+1, n):
      factor = L[j,col]
      U[j,:] -= factor * U[row,:] 
  
  return P, L, U

# Task b)
# Implement a method, calculating the determinant of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: The determinant - a floating number
def determinant(A):
  P, L, U = sp.linalg.lu(A)
  detU = 1
  r = 0
  
  #Calculate Determinant of U and row interchanges made to A
  for i in range(U.shape[0]):
    detU *= U[i,i]
    if P[i,i] != 1:
      r += 1
    
  det = (-1)**r * detU
  
  return det


############################################

A = np.array([[0,5,22/3],
              [4,2,1],
              [2,7,9]])

# A = np.triu(np.ones((10, 10)))
# for i in range(10):
#   A[i] *= (i + 1) / 10.
#   A[:, i] *= (10 - i + 1) / 10.
# A = A.transpose().dot(A).dot(A) * np.pi / np.e * 50.


print(lu(A))