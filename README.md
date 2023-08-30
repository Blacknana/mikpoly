## Mikpoly

##### How to use

- Modify `test_mikpoly.cu` and customize the test gemm cases as described below

  ```c++
  int num_case = 8;
  int m_arr[] = {35, 35, 35, 8457, 5120, 36458, 31999, 31999};
  int n_arr[] = {8457, 8457, 8457, 2560, 400, 1024, 84, 1024};
  int k_arr[] = {4096, 2048, 2560, 35, 5120, 1632, 1024, 84};
  ```

- Build and run the project

  ```bash
  mkdir -p build & cd build
  cmake .. && make
  ./test_mikpoly
  ```

  Performance versus cublas will be on the screen, unless the results are inconsistent with cublas.

