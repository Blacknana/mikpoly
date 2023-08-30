#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

namespace mikpoly {

void run_mikpoly_gemm(int m, int n, int k, const cutlass::half_t *A,
                      const cutlass::half_t *B, cutlass::half_t *C,
                      float falpha, float fbeta);

}
