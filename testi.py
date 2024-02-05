import numpy as np
from scipy.sparse import csr_matrix
import cupyx as cpx


A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
v = np.array([1, 0, -1])
A.dot(v)

print( type(A))

A_cupy_coo = cpx.scipy.sparse.rand( int(3), int(3) ).dot( v )

print( dir( A_cupy_coo) )