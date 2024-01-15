kernel_code_template = """
  __global__ void matmul_sharedmem( const float* A, const float* B, float* C, int M, int N, int K, int n_tiles) {
    
    float C_elem = 0;
    
    // Block indices, each block computes submatrix of C, C_sub
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    // Thread indices. Each thread computes an element of C_sub
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    
    // Determining number of tiles for loop
    if ( round( N / %(TILE_WIDTH)s ) * %(TILE_WIDTH)s < N ) {    
        int n_tiles = round( N / %(TILE_WIDTH)s ) + 1;
    } else {
        int n_tiles = round( N / %(TILE_WIDTH)s );
    }

    // Looping through relevant submatrices to compute C_sub
    for (int ii = 0; ii < n_tiles; ++ii ) { 
    
      // Loading submatrices from global memory
      int linear_ind_A = N * thread_row + thread_col + ii * %(TILE_WIDTH)s + block_row * %(TILE_WIDTH)s * N;
      int linear_ind_B = K * thread_row + thread_col + ii * %(TILE_WIDTH)s * K + block_col * %(TILE_WIDTH)s;
      
      // Shared memory for the submatrices of A and B
      __shared__ float A_sub[ %(TILE_WIDTH)s ][ %(TILE_WIDTH)s ];
      __shared__ float B_sub[ %(TILE_WIDTH)s ][ %(TILE_WIDTH)s ];
      
      // TODO check for threads outside matrix bounds
      
      A_sub[ thread_row ][ thread_col ] = A[ linear_ind_A ];
      B_sub[ thread_row ][ thread_col ] = B[ linear_ind_B ];
      
      __syncthreads();
      
      // Doin the actual multiplication of the submatrices
      for (int kk = 0; kk < %(TILE_WIDTH)s; ++kk) {
        
        float A_sub_elem = A_sub[ thread_row ][ kk ];
        float B_sub_elem = B_sub[ kk ][ thread_col ];
        
        C_elem += A_sub_elem * B_sub_elem;
        
      }
      
      __syncthreads();
    
    }
      
    // Saving the C_elem to matrix C
    int linear_ind_C = block_row * %(TILE_WIDTH)s * K + block_col * %(TILE_WIDTH)s + thread_row * K + thread_col;
    C[ linear_ind_C ] = C_elem;
    
  } 
"""