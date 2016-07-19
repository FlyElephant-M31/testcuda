#ifndef MYCUDA_H
#define MYCUDA_H

/**********************************************/
// CUDA helper functions
/**********************************************/

#include <cutil.h>

// timing
/**********************************************/

#define START_TIMING(name)			\
  cudaEvent_t ev_start_##name, ev_end_##name;	\
  cudaEventCreate(&ev_start_##name);		\
  cudaEventCreate(&ev_end_##name);		\
  cudaEventRecord(ev_start_##name, 0);		\
  cudaEventSynchronize(ev_start_##name);

#define GET_TIMING(result, name)		\
  float result;					\
  cudaEventRecord(ev_end_##name, 0);		\
  cudaEventSynchronize(ev_end_##name);		\
  cudaEventElapsedTime(&result, ev_start_##name, ev_end_##name);

// obtain physical thread id
/**********************************************/

#define THREADID1D (threadIdx.x)
#define THREADID2D (threadIdx.x + blockDim.x*threadIdx.y)
#define THREADID3D (threadIdx.x + blockDim.x*(threadIdx.y + blockDim.y*threadIdx.z))

// alignment
/**********************************************/

// smallest multiple of <alignment> larger or equal to <size>
// - how many blocks of size <alignment> are need to cover
//   an array of size <size>
inline size_t mincover(size_t size, size_t alignment) {
  return (size + alignment - 1)/alignment;
}

// largest multiple of alignment smaller or equal to size
// - how many blocks of size <alignment> fit at most into
//   an array of size <size>
inline size_t maxfit(size_t size, size_t alignment) {
  return size/alignment;
}

// return the smallest multiple of alignment greater or equal to size
// i.e. the size of the area calculated by maxfit
inline size_t align(size_t size, size_t alignment) {
  return alignment*((size + alignment - 1)/alignment);
}

// templated memory access with security check
/**********************************************/

template<class t> void cudaSafeMalloc(t *&dst, size_t size) {
  CUDA_SAFE_CALL(cudaMalloc((void **)&dst, size*sizeof(t)));
}

template<class t>
void cudaSafeMemcpySymbol(const t &dst, const t &src) {
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dst, &src, sizeof(src), 0));
}


template<class t>
void cudaSafeMemcpy(t *dst, const t *src, size_t size, cudaMemcpyKind kind) {
  CUDA_SAFE_CALL(cudaMemcpy(dst, src, size*sizeof(t), kind));
}

template<class t>
void cudaSafeMemcpyToDevice(t *dst, const t *src, size_t size) {
  cudaSafeMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

template<class t>
void cudaSafeMemcpyFromDevice(t *dst, const t *src, size_t size) {
  cudaSafeMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

template<class t>
void cudaSafeMemcpy2D(t *dst, size_t dpitch, const t *src, size_t spitch,
		      size_t width, size_t height, cudaMemcpyKind kind) {
  CUDA_SAFE_CALL(cudaMemcpy2D(dst, dpitch*sizeof(t),
			      src, spitch*sizeof(t),
			      width*sizeof(t), height,
			      kind));
}

template<class t>
void cudaSafeMemcpy2DToDevice(t *dst, size_t dpitch, const t *src, size_t spitch,
			      size_t width, size_t height) {
  cudaSafeMemcpy2D(dst, dpitch, src, spitch,
		   width, height, cudaMemcpyHostToDevice);
}

template<class t>
void cudaSafeMemcpy2DFromDevice(t *dst, size_t dpitch, const t *src, size_t spitch,
				size_t width, size_t height) {
  cudaSafeMemcpy2D(dst, dpitch, src, spitch,
		   width, height, cudaMemcpyDeviceToHost);
}

void cudaSafeFree(void *dst) { CUDA_SAFE_CALL(cudaFree(dst)); }



#endif
