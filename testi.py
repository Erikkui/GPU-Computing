import numpy as np
import time
# import pycuda stuff
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

modd = SourceModule("""
    __global__ void matmul(int n, const float *A, const float *B, float *C){

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by*blockDim.y + ty;
    int col = bx*blockDim.x + tx;

    if(row < n && col < n){
        float val = 0.0;
        for(int i=0; i<n; ++i){
        val += A[row*n + i]*B[n*i + col];
        }
        C[row*n + col] = val;
    }
    }
""")

BLOCK_SIZE = 16

n = 4
ni = np.int32(n)

# matrix A 
a = np.random.randn(n, n)*100
a = a.astype(np.float32)


# matrix B
b = np.random.randn(n, n)*100
b = b.astype(np.float32)

# matrix B
c = np.empty([n, n])
c = c.astype(np.float32)

# allocate memory on device
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# copy matrix to memory
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# get function
matmul = modd.get_function("matmul");

    

    
# call gpu function

matmul(ni, a_gpu, b_gpu, c_gpu, block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=(1,1,1));

# copy back the result
cuda.memcpy_dtoh(c, c_gpu)

print (np.linalg.norm(c - np.dot(a,b)) )
print (c)
print (np.dot(a,b))
print (c - np.dot(a,b))