#include <iostream>
#include <sstream>

#include "cuda_runtime.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "mikpoly.cuh"
#include <cublas_v2.h>

#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error)      \
                << " at: " << __LINE__ << std::endl;                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUDA_CHECK(status)                                                     \
  {                                                                            \
    cudaError_t error = status;                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)        \
                << " at line: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

int run(cublasHandle_t handle, int length_m, int length_n, int length_k,
        double &cb_time, double &ct_time) {
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk()); // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn()); // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn()); // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn()); // <- Create matrix D with dimensions M x N used to
                          // store output from CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(), 1, ElementInputA(4), ElementInputA(-4),
      0); // <- Fill matrix A on host with uniform-distribution random data

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(), 1, ElementInputB(4), ElementInputB(-4),
      0); // <- Fill matrix B on host with uniform-distribution random data

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(), 1, ElementOutput(4), ElementOutput(-4),
      0); // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view()); // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view()); // <- fill matrix D on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  float falpha = 1, fbeta = 0;

  // Launch mikpoly gemm kernel
  mikpoly::run_mikpoly_gemm(length_m, length_n, length_k,
                            tensor_a.device_data(), tensor_b.device_data(),
                            tensor_d.device_data(), falpha, fbeta);
  cudaDeviceSynchronize();
  tensor_d.sync_host();

  // Launch cublas gemm kernel
  half *A = reinterpret_cast<half *>(tensor_a.device_data());
  half *B = reinterpret_cast<half *>(tensor_b.device_data());
  half *C = reinterpret_cast<half *>(tensor_ref_d.device_data());
  cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
  cublasSetMathMode(handle, cublas_flags);
  int lda = length_k, ldb = length_k, ldc = length_m;
  cublasStatus_t cu_status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, length_m, length_n, length_k, &falpha,
      A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &fbeta, C, CUDA_R_16F, ldc,
      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  if (cu_status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Got cublas error: RunTime Error"
              << " at: " << __LINE__ << std::endl;
    exit(EXIT_FAILURE);
  }

  // check result
  tensor_ref_d.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(
      tensor_d.host_view(), tensor_ref_d.host_view());
  if (!passed) {
    return -1;
  }

  // test cublas time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double total_time = 0.0;
  float milliseconds = 0.0;
  int num_iter = 20;
  for (int i = 0; i < num_iter; ++i) {
    cudaEventRecord(start);
    cu_status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, length_m,
                             length_n, length_k, &falpha, A, CUDA_R_16F, lda, B,
                             CUDA_R_16F, ldb, &fbeta, C, CUDA_R_16F, ldc,
                             CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cu_status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "Got cublas error: RunTime Error"
                << " at: " << __LINE__ << std::endl;
      exit(EXIT_FAILURE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_time += milliseconds;
  }
  cudaDeviceSynchronize();
  cb_time = total_time / (num_iter * 1.0);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaDeviceSynchronize();

  // test mikpoly time
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  total_time = 0.0;
  milliseconds = 0.0;
  for (int i = 0; i < num_iter; ++i) {
    cudaEventRecord(start);
    mikpoly::run_mikpoly_gemm(length_m, length_n, length_k,
                              tensor_a.device_data(), tensor_b.device_data(),
                              tensor_d.device_data(), falpha, fbeta);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_time += milliseconds;
  }
  cudaDeviceSynchronize();
  ct_time = total_time / (num_iter * 1.0);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}

int main() {
  int num_case = 8;
  int m_arr[] = {35, 35, 35, 8457, 5120, 36458, 31999, 31999};
  int n_arr[] = {8457, 8457, 8457, 2560, 400, 1024, 84, 1024};
  int k_arr[] = {4096, 2048, 2560, 35, 5120, 1632, 1024, 84};

  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
      std::cerr << "Got cublas error: CUBLAS_STATUS_NOT_INITIALIZED"
                << " at: " << __LINE__ << std::endl;
    }
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < num_case; i++) {
    double cb_time, ct_time;
    int passed = run(handle, m_arr[i], n_arr[i], k_arr[i], cb_time, ct_time);

    if (passed == -1) {
      std::cout << "Failed," << std::flush;
      break;
    } else {
      std::cout << "(m, n, k) = (" << m_arr[i] << ", " << n_arr[i] << ", "
                << k_arr[i] << ")" << std::endl;
      std::cout << "cublas time: " << cb_time << ", mikpoly time: " << ct_time
                << std::endl;
    }
  }

  std::cout << std::endl << std::flush;
  return 0;
}
