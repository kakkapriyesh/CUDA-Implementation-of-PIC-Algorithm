#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include  "input_file.h"
#include "cuda_wrapper.h"

// // Thread block size
// #define BLOCK_SIZE  1  // number of threads in a direction of the block
// #define M_WIDTH     100 // number of columns
// #define M_HEIGHT    1 // number of rows

/*
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} cu_Matrix;
*/
__device__ int stop_flag;
//__device__ int d_iter_cache;
// Forward declaration of the matrix multiplication kernel
__global__ void Jacobi_Egrid_Kernel(const cu_Matrix, const cu_Matrix,const cu_Matrix, cu_Matrix, int* );
__global__ void Eparticle_Kernel(const cu_Matrix, const cu_Matrix, const cu_Matrix,const cu_Matrix_int, const cu_Matrix_int, cu_Matrix );

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE (M_P, M_B, M_G, M_W1, M_W2, M_p1, M_p2, M_Eg, M_Ep)
void Egrid_solver(const cu_Matrix P, const cu_Matrix B, const cu_Matrix G, 
                 const cu_Matrix W1, const cu_Matrix W2, const cu_Matrix_int p1, 
                 const cu_Matrix_int p2, cu_Matrix Eg, cu_Matrix Ep)
{

    int iter_cache=0;
    int* d_iter_cache;  //creates pointer to pass iter value

    // Load P and B to device 
    cu_Matrix d_P;
    d_P.width = P.width; d_P.height = P.height;
    size_t size = P.width * P.height * sizeof(float);
    cudaMalloc(&d_P.elements, size);
    cudaMemcpy(d_P.elements, P.elements, size,
               cudaMemcpyHostToDevice);
    cu_Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate G in device memory
    cu_Matrix d_G;
    d_G.width = G.width; d_G.height = G.height;
    size = G.width * G.height * sizeof(float);
    cudaMalloc(&d_G.elements, size);
    cudaMemcpy(d_G.elements, G.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate W1 in device memory
    cu_Matrix d_W1;
    d_W1.width = W1.width; d_W1.height = W1.height;
    size = W1.width * W1.height * sizeof(float);
    cudaMalloc(&d_W1.elements, size);
    cudaMemcpy(d_W1.elements, W1.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate W2 in device memory
    cu_Matrix d_W2;
    d_W2.width = W2.width; d_W2.height = W2.height;
    size = W2.width * W2.height * sizeof(float);
    cudaMalloc(&d_W2.elements, size);
    cudaMemcpy(d_W2.elements, W2.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate p1 in device memory
    cu_Matrix_int d_p1;
    d_p1.width = p1.width; d_p1.height = p1.height;
    size = p1.width * p1.height * sizeof(int);
    cudaMalloc(&d_p1.elements, size);
    cudaMemcpy(d_p1.elements, p1.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate p2 in device memory
    cu_Matrix_int d_p2;
    d_p2.width = p2.width; d_p2.height = p2.height;
    size = p2.width * p2.height * sizeof(int);
    cudaMalloc(&d_p2.elements, size);
    cudaMemcpy(d_p2.elements, p2.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate Eg in device memory
    cu_Matrix d_Eg;
    d_Eg.width = Eg.width; d_Eg.height = Eg.height;
    size = Eg.width * Eg.height * sizeof(float);
    cudaMalloc(&d_Eg.elements, size);

    // Allocate Ep in device memory
    cu_Matrix d_Ep;
    d_Ep.width = Ep.width; d_Ep.height = Ep.height;
    size = Ep.width * Ep.height * sizeof(float);
    cudaMalloc(&d_Ep.elements, size);

    cudaMalloc(&d_iter_cache, sizeof(int));// creates memory in gpu

    // Invoke kernel
    // dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 dimGrid(A.width / dimBlock.x, A.height / dimBlock.y);
    
    // Invoke kernel
    // dim3 dimGrid(32, 32);
    // dim3 dimBlock(38,26);
    
    ////////////////// Benchmarking  ///////////////
    
    // cudaEvent_t start, stop; 
	// cudaEventCreate(&start); 
	// cudaEventCreate(&stop); 
	// cudaEventRecord(start, 0); 
	 
	// /// your kernel call here 
    Jacobi_Egrid_Kernel<<< P.width/Thread_grid ,Thread_grid >>>(d_P, d_B, d_G, d_Eg, d_iter_cache);
    Eparticle_Kernel<<<Ep.width/Thread_particle,Thread_particle>>>(d_Eg, d_W1, d_W2, d_p1, d_p2, d_Ep);
	 //<<no of block in the grid,no of Thread in the block>>
	
    // cudaEventRecord(stop, 0); 

	// cudaEventSynchronize(stop); 
	 
	// float elapseTime; 
	// cudaEventElapsedTime(&elapseTime, start, stop); 
	// cout << "Time to run the kernel: "<< elapseTime << " ms "<< endl;
	
    ////////////////// Benchmarking  ///////////////

    
    // Read Eg from device memory
    size = Eg.width * Eg.height * sizeof(float);
    cudaMemcpy(Eg.elements, d_Eg.elements, size, //d stands for device
               cudaMemcpyDeviceToHost);
    size = Ep.width * Ep.height * sizeof(float);
    cudaMemcpy(Ep.elements, d_Ep.elements, size, //d stands for device
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&iter_cache, d_iter_cache, sizeof(int),  // (cpu variable,cuda variable,size)
               cudaMemcpyDeviceToHost);
    
    cout<<"Jacobi_iter_stopped_at "<<iter_cache<<std::endl;
    

    // Free device memory
    cudaFree(d_P.elements);
    cudaFree(d_B.elements);
    cudaFree(d_G.elements);
    cudaFree(d_Eg.elements);
    cudaFree(d_W1.elements);
    cudaFree(d_W2.elements);
    cudaFree(d_p1.elements);
    cudaFree(d_p2.elements);
    cudaFree(d_Ep.elements);
    cudaFree(d_iter_cache);
    
}

// Matrix multiplication kernel called by MatMul()
__global__ void Jacobi_Egrid_Kernel(cu_Matrix P, cu_Matrix B, cu_Matrix G, cu_Matrix Eg, int* d_iter_cache)
{
    // Each thread computes one element of Eg(Egrid)
    // P(Phi0), B(b), G(Gmtx)
    // by accumulating results into Egvalue
    stop_flag=iterations_Jacobi;
    
    float Egvalue = 0;
    int iter = 11;

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Gauss-Siedel solver to solve poission equation for potential
    for (iter = 0 ; iter < stop_flag ; iter++)
    {
        if( (col>0) && (col<(P.width-1)) )
            Egvalue = 0.5*(P.elements[col+1]+P.elements[col-1]-B.elements[col]);

        if( col == 0 )
            Egvalue = 0.5*(P.elements[1]+P.elements[P.width-1]-B.elements[0]);
        
        if( col == (P.width-1) )
            Egvalue = 0.5*(P.elements[0]+P.elements[P.width-2]-B.elements[P.width-1]);

        __syncthreads(); // This syncs all the threads after the iteration 

        G.elements[col] = abs(P.elements[col] - Egvalue);

        if((iter % 100 == 0) && (col == 0))
        {   
            float error = 0.0;
            for( int i = 0 ; i < P.width ; i++ )
                error += G.elements[i];

            if(error <= 1e-6)
                stop_flag = 0; //global variable is used here
        }   

        P.elements[col] = 0.7*Egvalue + (1.0-0.7)*P.elements[col];
        // weighted average, (this helps in convergence)
    }

    // Calculate Efield on grid from potential
    Egvalue = 0.0;
    for (int icolG = 0 ; icolG < G.width ; icolG++)
        Egvalue += (G.elements[col*G.width + icolG]*P.elements[icolG]);
    
    
    *d_iter_cache = iter; //to pass variable in iter

    Eg.elements[col] = Egvalue;
    
}

__global__ void Eparticle_Kernel(cu_Matrix Eg, cu_Matrix W1, cu_Matrix W2, cu_Matrix_int p1, cu_Matrix_int p2, cu_Matrix Ep)
{
    // Each thread computes one element of Eg(Efield)
    //Ep=Electric filed at particle posn, Eg= Electric field calc at grid from wrapper above
    //W1,W2= left and right node weight, p1,p2=left and right node index of the particle 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // E[i] = weight_j[i] * E_grid[pidx[i]] + weight_jp1[i] * E_grid[pidx1[i]]
    Ep.elements[col] = W1.elements[col] * Eg.elements[p1.elements[col]] + W2.elements[col] * Eg.elements[p2.elements[col]];
}






int get_grid_Efield(
	float*  E,
	double*  phi_grid,
	double*  bvec,
    double** Gmtx,
	double*  weight_j,
	double*  weight_jp1,
	int*     pidx,
	int*     pidx1){
	
    float *P, *B, *G, *Eg, *Ep, *W1, *W2;
    int i, j, *p1, *p2;
    cu_Matrix M_P, M_B, M_G, M_Eg, M_Ep, M_W1, M_W2;
    cu_Matrix_int M_p1, M_p2; 

    P = (float*)malloc(No_Of_Mesh_Cell*sizeof(float));
    B = (float*)malloc(No_Of_Mesh_Cell*sizeof(float));
    Eg = (float*)malloc(No_Of_Mesh_Cell*sizeof(float));
    G = (float*)malloc(No_Of_Mesh_Cell*No_Of_Mesh_Cell*sizeof(float)); 

    
    Ep = (float*)malloc(No_Of_Particles*sizeof(float));
    W1 = (float*)malloc(No_Of_Particles*sizeof(float));
    W2 = (float*)malloc(No_Of_Particles*sizeof(float));
    p1 = (int*)malloc(No_Of_Particles*sizeof(int));
    p2 = (int*)malloc(No_Of_Particles*sizeof(int)); 

    srand((unsigned)time( NULL ));

    // initialize A[] and B[]
    for(i = 0; i < No_Of_Mesh_Cell; i++)
    {
        P[i] = (float)phi_grid[i];
        B[i] = (float)bvec[i];
        Eg[i] = 0.0;
 
        for(j = 0; j < No_Of_Mesh_Cell; j++)
            G[i*No_Of_Mesh_Cell + j] = (float)Gmtx[i][j];
    }

    for(i = 0; i < No_Of_Particles; i++)
    {
        Ep[i] = 0.0;

        W1[i] = (float)weight_j[i];
        W2[i] = (float)weight_jp1[i];
        p1[i] = (int)pidx[i];
        p2[i] = (int)pidx1[i];
    }
    
    M_P.width = No_Of_Mesh_Cell; M_P.height = M_HEIGHT;
    M_P.elements = P; 
    M_B.width = No_Of_Mesh_Cell; M_B.height = M_HEIGHT;
    M_B.elements = B; 
    M_G.width = No_Of_Mesh_Cell; M_G.height = No_Of_Mesh_Cell;
    M_G.elements = G; 
    M_Eg.width = No_Of_Mesh_Cell; M_Eg.height = M_HEIGHT;
    M_Eg.elements = Eg; 
    M_Ep.width = No_Of_Particles; M_Ep.height = M_HEIGHT;
    M_Ep.elements = Ep; 
    M_W1.width = No_Of_Particles; M_W1.height = M_HEIGHT;
    M_W1.elements = W1; 
    M_W2.width = No_Of_Particles; M_W2.height = M_HEIGHT;
    M_W2.elements = W2; 
    M_p1.width = No_Of_Particles; M_p1.height = M_HEIGHT;
    M_p1.elements = p1; 
    M_p2.width = No_Of_Particles; M_p2.height = M_HEIGHT;
    M_p2.elements = p2; 

        
    Egrid_solver(M_P, M_B, M_G, M_W1, M_W2, M_p1, M_p2, M_Eg, M_Ep);
        
    
    for(i = 0; i < No_Of_Mesh_Cell; i++)
    {
        phi_grid[i] = (double)Eg[i];
        // cout << "phi "<<phi_grid[i] << endl;
    }
    // cin.get();


    for(i = 0; i < No_Of_Particles; i++)
    {
        E[i] = (float)Ep[i];
         //cout << "E "<<E[i] << endl;
    }
    // cin.get();

    free(P); free(B); free(G); free(Eg); free(Ep); free(W1); free(W2); free(p1); free(p2);
    return 0;
}

