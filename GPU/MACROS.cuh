

// general

#define CUDA_DEBUG		// provides some tests and output specific to the used CUDA code
#define STUFF_DEBUG		// some stuff
//#define MATRIX_DEBUG		// enables the matrix multiplication on the CPU (in the inner CG solver)
//#define CG_DEBUG		// enables the CG on the CPU

//#define OPERATOR_BENCHMARK
//#define CPU_BENCHMARK
#define GPU_BENCHMARK






// CUDA parameters

#define BLOCKSIZE1 128
#define BLOCKSIZE2 128
#define BLOCKSIZE3 128
#define BLOCKSIZE4 128
#define BLOCKSIZE5 128






// debugging macros for CUDA, CUBLAS and kernel functions



// debug	// CUDA

#define CUDA_CHECK(errorMessage, successMessage) {					\
          if ( (cudaerr = cudaGetLastError()) != cudaSuccess ) {			\
            printf("%s: %s\n", errorMessage, cudaGetErrorString(cudaerr));		\
            exit(-1);									\
          }										\
          else printf("%s%s", successMessage, "\n");					\
        }

#define CUDA_CHECK_NO_SUCCESS_MSG(errorMessage) {					\
          if ( (cudaerr = cudaGetLastError()) != cudaSuccess ) {			\
            printf("%s: %s\n", errorMessage, cudaGetErrorString(cudaerr));		\
            exit(-1);									\
          }										\
        }




// debug	// CUBLAS core function

#define CUBLAS_CORE_CHECK(errorMessage, successMessage) {				\
          if ( (cublasstatus = cublasGetError()) != CUBLAS_STATUS_SUCCESS ) {		\
            printf("%s%s", errorMessage, "\n");						\
            exit(-1);									\
          }										\
          else printf("%s%s", successMessage, "\n");					\
        }

#define CUBLAS_CORE_CHECK_NO_SUCCESS_MSG(errorMessage) {				\
          if ( (cublasstatus = cublasGetError()) != CUBLAS_STATUS_SUCCESS ) {		\
            printf("%s%s", errorMessage, "\n");						\
            exit(-1);									\
          }										\
        }




// debug	// CUBLAS helper function

#define CUBLAS_HELPER_CHECK(function, errorMessage, successMessage) {			\
          if ( (cublasstatus = function) != CUBLAS_STATUS_SUCCESS ) {			\
            printf("%s%s", errorMessage, "\n");						\
            exit(-1);									\
          }										\
          else printf("%s%s", successMessage, "\n");					\
        }

#define CUBLAS_HELPER_CHECK_NO_SUCCESS_MSG(function, errorMessage) {			\
          if ( (cublasstatus = function) != CUBLAS_STATUS_SUCCESS ) {			\
            printf("%s%s", errorMessage, "\n");						\
            exit(-1);									\
          }										\
        }




// debug	// kernel function

#define CUDA_KERNEL_CHECK(errorMessage, successMessage) {				\
          if ( (cudaerr = cudaThreadSynchronize()) != cudaSuccess ) {			\
            printf("%s: %s\n", errorMessage, cudaGetErrorString(cudaGetLastError()));	\
            exit(-1);									\
          }										\
          else printf("%s%s", successMessage, "\n");					\
        }

#define CUDA_KERNEL_CHECK_NO_SUCCESS_MSG(errorMessage) {				\
          if ( (cudaerr = cudaThreadSynchronize()) != cudaSuccess ) {			\
            printf("%s: %s\n", errorMessage, cudaGetErrorString(cudaGetLastError()));	\
            exit(-1);									\
          }										\
        }




////////////////////////////// EXAMPLES ////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
//    // debug	// CUDA
//    #ifdef CUDA_DEBUG
//      CUDA_CHECK("CUDA error in mixedsolve_eo_nd(). Host to device interaction failed.", "Fields initializedhallo on device.");
//    #endif
//
//   
//    // debug	// CUBLAS helper function
//    #ifdef CUDA_DEBUG
//      CUBLAS_HELPER_CHECK(cublasInit(), "Error in cublasInit(). Couldn't initialize CUBLAS.", "CUBLAS is initialized.");
//    #endif
//
//
//    // debug	// kernel
//    #ifdef CUDA_DEBUG
//      CUDA_KERNEL_CHECK("Error in cg_eo_nd(): Initializing spinor fields on device failed.", "Spinor fields initialized on device.");
//    #endif
//
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


