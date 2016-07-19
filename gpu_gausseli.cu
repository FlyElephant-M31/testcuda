#include <cutil.h>
#include "mycuda.h"
#include "gpu_gausseli.h"
#include "timing.h"

/**********************************************/
// (tuning) constants
/**********************************************/

// the matrix is subdivided into blocks of this size
static const size_t block_size = 16;
// in the main block, as many blocks per side are combined
// to maximize shared memory usage
static const size_t merge_blocks = 16384/(2*sizeof(float)*block_size*block_size) - 1;

/**********************************************/
// implementation
/**********************************************/

__device__ uint block_address(uint row, uint col) {
  return row*block_size + col;
}

__device__ uint global_address(uint row, uint col, uint stride) {
  return row*stride + col;
}

/**********************************************/
// kernels elimination
/**********************************************/

__global__ void elimination_head_and_b(float *A_base, uint stride, float *b_base)
{
  const uint col = threadIdx.x;
  const uint row = threadIdx.y;
  const uint bottom = block_size - 1;

  __shared__ float A_block[block_size*block_size];
  __shared__ float b_block[block_size];

  // every thread is responsible for one matrix element
  float *A_elem = A_block + block_address(row, col);

  // copy data to shared mem
  *A_elem =  A_base[global_address(row, col, stride)];
  // first row also copies b (then this hits only first warp)
  if (row == 0) b_block[col] = b_base[col];

  __syncthreads();

  // do elimination in shared memory
  for (int r = 0; r < bottom; ++r) {
    const float pref = A_block[block_address(row, r)]/A_block[block_address(r, r)];

    __syncthreads(); // to prevent overwriting!

    if (row > r) {
      if (col  > r) {
	// update my matrix element
	*A_elem -= pref*A_block[block_address(r, col)];
      }
      if (col == r) {
	// remember the prefactor
	*A_elem = pref;
	// and update b-vector
	b_block[row] -= pref*b_block[r];
      }
    }

    __syncthreads();
  }

  // write back results to main memory
  A_base[global_address(row, col, stride)] = *A_elem;
  if (row == 0) b_base[col] = b_block[col];
}

__global__ void elimination_top_blocks(float *A_base, uint stride)
{
  const uint col     = threadIdx.x;
  const uint row     = threadIdx.y;
  const uint col_off = blockDim.x*(blockIdx.x + 1);
  const uint bottom = block_size - 1;

  __shared__ float head_A_block[block_size*block_size];
  __shared__ float cur_A_block[block_size*block_size];

  // every thread is responsible for one matrix element
  uint my_elem = block_address(row, col);

  // read head block
  if (row > col)
    head_A_block[my_elem] = A_base[global_address(row, col, stride)];
  else
    head_A_block[my_elem] = 0;

  // read current block
  cur_A_block[my_elem] = A_base[global_address(row, col + col_off, stride)];

  for (int r = 0; r < bottom; ++r) {
    __syncthreads();

    if (row > r)
      cur_A_block[my_elem] -= cur_A_block[block_address(r, col)]*head_A_block[block_address(row, r)];
  }

  // write back
  A_base[global_address(row, col + col_off, stride)] = cur_A_block[my_elem];
}

__global__ void elimination_row_heads_and_b(float *A_base, uint stride, float *b_base)
{
  const uint col     = threadIdx.x;
  const uint row     = threadIdx.y;
  const uint row_off = block_size*(blockIdx.x + 1);

  __shared__ float head_A_block[block_size*block_size];
  __shared__ float cur_A_block[block_size*block_size];
  __shared__ float head_b_block[block_size];
  __shared__ float cur_b_block[block_size];

  // every thread is responsible for one matrix element
  uint my_elem = block_address(row, col);

  // load data
  head_A_block[my_elem] = A_base[global_address(row, col, stride)];
  cur_A_block[my_elem]  = A_base[global_address(row + row_off, col, stride)];
  // first row also copies b (then this hits only the first warp)
  if (row == 0) {
    head_b_block[col] = b_base[col];
    cur_b_block[col]  = b_base[col + row_off];
  }
  __syncthreads();

  for (int r = 0; r < block_size; ++r) {
    const float pref = cur_A_block[block_address(row, r)]/head_A_block[block_address(r, r)];

    __syncthreads(); // to prevent overwriting!

    if (col  > r) {
      // update my matrix element
      cur_A_block[my_elem] -= pref*head_A_block[block_address(r, col)];
    }
    if (col == r) {
      // remember the prefactor
      cur_A_block[my_elem] = pref;
      // and update b-vector
      cur_b_block[row] -= pref*head_b_block[r];
    }

    __syncthreads();
  }

  // write back results to main memory
  A_base[global_address(row + row_off, col, stride)] = cur_A_block[my_elem];
  if (row == 0) b_base[col + row_off] = cur_b_block[col];
}

__global__ void elimination_rest(float *A_base, uint stride, uint n_blocks)
{
  const uint col     = threadIdx.x;
  const uint row     = threadIdx.y;

  __shared__ float head_A_block[block_size*block_size*merge_blocks];
  __shared__ float side_A_block[block_size*block_size*merge_blocks];

  // load the multiply used update data
  // load head data
  for (uint mblock_x = 0; mblock_x < merge_blocks; ++mblock_x) {
    const uint block_x = blockIdx.x*merge_blocks + mblock_x + 1;
    if (block_x < n_blocks)
      head_A_block[block_address(mblock_x*block_size + row, col)] =
	A_base[global_address(row, col + block_x*block_size, stride)];
  }
  
  // load prefactors
  for (uint mblock_y = 0; mblock_y < merge_blocks; ++mblock_y) {
    const uint block_y = blockIdx.y*merge_blocks + mblock_y + 1;
    if (block_y < n_blocks)
      side_A_block[block_address(mblock_y*block_size + row, col)] =
	A_base[global_address(row + block_y*block_size, col, stride)];
  }
  __syncthreads();

  for (uint mblock_y = 0; mblock_y < merge_blocks; ++mblock_y) {
    for (uint mblock_x = 0; mblock_x < merge_blocks; ++mblock_x) {
      const uint block_y = blockIdx.y*merge_blocks + mblock_y + 1;
      const uint block_x = blockIdx.x*merge_blocks + mblock_x + 1;

      if (block_y < n_blocks & block_x < n_blocks) {
	// every thread is responsible for one matrix element
	float my_elem = A_base[global_address(row + block_y*block_size, col + block_x*block_size, stride)];

	float *side_ptr = side_A_block + block_address(mblock_y*block_size + row, 0);
	float *head_ptr = head_A_block + block_address(mblock_x*block_size, col);

	for (uint r = 0; r < block_size; ++r)
	  my_elem -= side_ptr[r]*head_ptr[block_size*r];

	// write back results to main memory
	A_base[global_address(row + block_y*block_size, col + block_x*block_size, stride)] = my_elem;
      }
    }
  }
}


/**********************************************/
// kernels substitution
/**********************************************/

__global__ void substitution_full_blocks(float *A_base, uint stride, float *x_base, float *buffer)
{
  const uint row = threadIdx.x;
  const uint col_off = block_size*(blockIdx.x + 1);

  __shared__ float x_block[block_size];

  x_block[row] = x_base[row + col_off];
  __syncthreads();

  float my_row = 0;
  for (uint c = 0; c < block_size; c++)
    my_row += A_base[global_address(row, c + col_off, stride)]*x_block[c];

  buffer[row + col_off - block_size] = my_row;
}

__global__ void substitution_head_b_and_write_x(float *A_base, uint stride, float *b_base,
						float *buffer, uint full_blocks,
						uint last_block_w, uint tri_block_w, float *x_base)
{
  const uint row = threadIdx.x;

  __shared__ float x_block[block_size];

  // b-vector
  float my_row = b_base[row];

  // (partial) end-block
  if (last_block_w > 0) {
    const uint lb_col_off = (full_blocks + 1)*block_size;

    if (row < last_block_w) x_block[row] = x_base[row + lb_col_off];
    __syncthreads();
    
    for (int c = 0; c < last_block_w; ++c)
      my_row -= A_base[global_address(row, c + lb_col_off, stride)]*x_block[c];
  }

  // buffered inner blocks
  for (int b = 0; b < full_blocks; ++b)
    my_row -= buffer[b*block_size + row];

  // triangular part, we abuse x_block[0] to store the
  // last calculated x which all others use in the next step
  for (int c = tri_block_w - 1; c >= 0; --c) {
    const float matrix_elem = A_base[row*stride + c];

    // first the new x-element 
    if (row == c) {
      my_row /= matrix_elem;
      x_block[0]  = my_row;
      x_base[row] = my_row;
    }

    __syncthreads();

    if (c > row)  my_row -= matrix_elem*x_block[0];
  }
}

/**********************************************/
// CPU wrapper code
/**********************************************/

inline void print_state(float *gpuA, float *gpub, size_t size, size_t stride)
{
  Matrix A(size, size);
  Vector b(size);
  cudaSafeMemcpy2D(&(A[0]), size, gpuA, stride, size, size, cudaMemcpyDeviceToHost);
  cudaSafeMemcpy(&(b[0]), gpub, size, cudaMemcpyDeviceToHost);

  printf("----------------------------\n");    
  for (size_t r = 0; r < size; ++r) {
    printf("row %ld\n", r);    
    for (size_t s = 0; s < size; ++s)
      printf("%f ", A.get(r, s));
    printf("= %f\n", b[r]);    
  }
  printf("----------------------------\n");    
}

inline void check_state(const char *ident, float *gpuA, float *gpub, size_t size, size_t stride)
{
  Matrix A(size, size);
  Vector b(size);
  cudaSafeMemcpy2D(&(A[0]), size, gpuA, stride, size, size, cudaMemcpyDeviceToHost);
  cudaSafeMemcpy(&(b[0]), gpub, size, cudaMemcpyDeviceToHost);

  for (size_t r = 0; r < size; ++r)
    for (size_t s = 0; s < size; ++s)
      if (isinf(A.get(r, s))) {
	printf("%s: found inf at %ld.%ld\n", ident, r, s);
      }
}

void gpu_init(int argc, char **argv) { CUT_DEVICE_INIT(argc, argv); }

void gpu_solve(const Matrix &A, Vector &x, const Vector &b, Matrix &finalA, Vector &finalb)
{
  size_t size = x.size();
  size_t aligned_size = align(size, block_size);
  size_t pad = aligned_size - size;

  dim3 dimBlock;
  dim3 dimGrid;

  float *gpuA, *gpux, *gpub, *gpu_buffer;

  // allocate and copy memory
  /*******************************************************/

  fprintf(stderr, "GPU solver: block size %ld, merging %ld blocks\n", block_size, merge_blocks);

  cudaSafeMalloc(gpuA, aligned_size*aligned_size);
  cudaSafeMalloc(gpux, aligned_size);
  cudaSafeMalloc(gpub, aligned_size);
  cudaSafeMalloc(gpu_buffer, aligned_size);

  cudaSafeMemcpy2D(gpuA, aligned_size, &(A[0]), size,
		   size, size, cudaMemcpyHostToDevice);
  cudaSafeMemcpy(gpub, &(b[0]), size, cudaMemcpyHostToDevice);

  // elimination
  /*******************************************************/

  dimBlock = dim3(block_size, block_size);
  for (uint n_blocks = aligned_size/block_size; n_blocks >= 1; --n_blocks) {
    // number of merged blocks
    uint big_blocks = (n_blocks - 1 + merge_blocks - 1)/merge_blocks;
    // this is the starting coordinate of the block
    uint block_start     = aligned_size - n_blocks*block_size;
    // the matrix we are left with
    float *sub_A = gpuA + block_start + block_start*aligned_size;
    // the b-part we are left with
    float *sub_b = gpub + block_start;

    // head block with pivot elements + b-vector head
    /****************************************************/
    // we just use a single MP due to data dependencies
    dimGrid  = dim3(1);

    elimination_head_and_b<<<dimGrid, dimBlock, 0>>>(sub_A, aligned_size, sub_b);

    if (n_blocks > 1) {
      // top blocks
      /****************************************************/
      dimGrid  = dim3(n_blocks - 1);
      elimination_top_blocks<<<dimGrid, dimBlock, 0>>>(sub_A, aligned_size);

      // row head blocks, calculate prefactors + b-vector rest
      /****************************************************/
      dimGrid  = dim3(n_blocks - 1);
      elimination_row_heads_and_b<<<dimGrid, dimBlock, 0>>>(sub_A, aligned_size, sub_b);

#define TIME_INNER_LOOP
#ifdef TIME_INNER_LOOP
      cudaEvent_t ev_start, ev_end;
      if (block_start == 0) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_end);
        cudaEventRecord(ev_start, 0); cudaEventSynchronize(ev_start);
      }
#endif
      // rest, big blocks
      /****************************************************/
      dimGrid  = dim3(big_blocks, big_blocks);
      elimination_rest<<<dimGrid, dimBlock, 0>>>(sub_A, aligned_size, n_blocks);

#ifdef TIME_INNER_LOOP
      if (block_start == 0) {
        cudaEventRecord(ev_end, 0); cudaEventSynchronize(ev_end);
        float time; cudaEventElapsedTime(&time, ev_start, ev_end);
        fprintf(stderr, "Shared memory %f GB/s\n", 1./(1024*1024*1024)*sizeof(float)*big_blocks*big_blocks*
	        (2*merge_blocks*block_size*block_size +
	         2*merge_blocks*merge_blocks*block_size*block_size*block_size)/(1e-3*time));
        fprintf(stderr, "Main memory %f GB/s\n", 1./(1024*1024*1024)*sizeof(float)*big_blocks*big_blocks*
	        (2*merge_blocks*block_size*block_size +
	         2*merge_blocks*merge_blocks*block_size*block_size)/(1e-3*time));
        fprintf(stderr, "MADD %f GFlops\n", 1e-9*big_blocks*big_blocks*
	        2*merge_blocks*merge_blocks*block_size*block_size*block_size/(1e-3*time));
      }
#endif
    }
  }

  // substitution
  /*******************************************************/
  
  dimBlock = dim3(block_size);
  for (uint n_blocks = 1; n_blocks <= aligned_size/block_size; n_blocks++) {
    // this is the starting coordinate of the block
    uint block_start = aligned_size - n_blocks*block_size;
    // the matrix we deal with
    float *sub_A = gpuA + block_start + block_start*aligned_size;
    // the b-part we deal with
    float *sub_b = gpub + block_start;
    // the x-part we deal with
    float *sub_x = gpux + block_start;

    uint full_blocks;
    uint last_block_width;
    uint tri_block_width;

    // full inner blocks
    /****************************************************/
    // blocks without the triangular ones

    if (pad == 0) {
      // no pad, triangular block is always full, no special treatment of last block
      full_blocks      = n_blocks - 1;
      last_block_width = 0;
      tri_block_width  = block_size;
    }
    else {
      // padding, last block needs special treatment, namely shortening
      if (n_blocks > 1) {
	// last block is not the triangular block, both have separate treatment
	full_blocks      = n_blocks - 2;
	last_block_width = block_size - pad;
	tri_block_width  = block_size;
      }
      else {
	// last block is the triangular block
	full_blocks      = 0;
	last_block_width = 0;
	tri_block_width  = block_size - pad;
      }
    }

    if (full_blocks > 0) {
      dimGrid  = dim3(full_blocks);
      substitution_full_blocks<<<dimGrid, dimBlock, 0>>>(sub_A, aligned_size, sub_x, gpu_buffer);
    }

    // triangular head block, b-vector head, and (partial) end-block
    // also takes adds the results of the inner blocks
    /****************************************************/
    // we just use a single MP due to data dependencies
    dimGrid  = dim3(1);
    substitution_head_b_and_write_x<<<dimGrid, dimBlock, 0>>>(sub_A,
							      aligned_size, sub_b,
							      gpu_buffer, full_blocks,
							      last_block_width,
							      tri_block_width, sub_x);
  }

  // copy back and deallocate
  /*******************************************************/

  cudaSafeMemcpy2D(&(finalA[0]), size, gpuA, aligned_size,
		   size, size, cudaMemcpyDeviceToHost);
  cudaSafeMemcpy(&(finalb[0]), gpub, size, cudaMemcpyDeviceToHost);
  cudaSafeMemcpy(&(x[0]), gpux, size, cudaMemcpyDeviceToHost);

  cudaSafeFree(gpuA);
  cudaSafeFree(gpux);
  cudaSafeFree(gpub);
  cudaSafeFree(gpu_buffer);
}
