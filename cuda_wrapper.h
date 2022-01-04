#if !defined(__WRAPPER__)
#define __WRAPPER__

#include <iostream>  //input output stream
#include <new>
#include <cstddef>  

using namespace std;

// Matrices are stored in row-major order:
// // M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
     int width;
     int height;
     float* elements;
} cu_Matrix;


// Matrices are stored in row-major order:
// // M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
     int width;
     int height;
     int* elements;
} cu_Matrix_int;

int get_grid_Efield(
	float*  E,
	double* phi_grid,
	double*  bvec,
	double** Gmtx,
	double*  weight_j,
	double*  weight_jp1,
	int*  	 pidx,
	int*  	 pidx1);

#endif 
