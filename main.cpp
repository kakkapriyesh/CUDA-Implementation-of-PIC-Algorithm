#include <iostream>
#include <fstream>
#include <math.h>
#include <random>
#include <fstream>
#include<chrono>
#include "cuda_wrapper.h"
#include  "input_file.h"  //Get inputs from here

using namespace std::chrono;
// The electrostatic PIC code has been implemented for two stream instability case in C++ 
//The code is accelerated using CUDA programming.
//Jacobi iterative method is used to solve Potential poission equation. 

// All the codes equation and methodology is obtained from the following blog
//https://medium.com/swlh/create-your-own-plasma-pic-simulation-with-python-39145c66578b

// And the python code can be found here
//https://github.com/pmocz/pic-python


#define PI 3.14159265




double getAcc( float*, double*, double*, int, int, double, int, double**);

int main()
{   

#ifdef device_GPU
cout<<"Running this code at a god speed on GPU " <<endl;
#endif

#ifdef device_CPU
cout<<"Running this code with a snail pace on CPU " <<endl;
#endif


    // Generate Initial Conditions
	srand(42);  

    default_random_engine generator;
    uniform_real_distribution<double> uniform_distribution(0.0,1.0); //Intializing uniform distribution
    normal_distribution<double> norm_distribution(0.0,1.0);          //Intializing normal distribution   

	// Simulation parameters
    int N         = No_Of_Particles;        // Number of particles
	int Nx        = No_Of_Mesh_Cell;        // Number of mesh cells
	int tEnd      = sim_end_time;           // time at which simulation ends
	int dt        = timestep;               // timestep
	int n0        = electron_no_density;    // electron number density //q charge is 1
	int vb        = beam_vel;               // beam velocity
	int vth       = beam_width;             // beam width
	double A      = pertub;			        // perturbation
	double boxsize= box_size;               // periodic domain [0,boxsize]
	double t          = 0;                  // current time of the simulation
	bool plotRealTime = 1;                  // switch on for plotting as the simulation goes along
	

    double dx = boxsize/Nx;

    int i, j, Nh, Nt;
    double random01, randn, *pos, *vel, *phi_grid, **Gmtx, *bvec;
    float *E;

    pos         = new double[N];
    vel         = new double[N];
    E           = new float[N];
    bvec        = new double[Nx];
    phi_grid    = new double[Nx];   //allocating dynamic memory after creating the pointer
    Gmtx        = new double*[Nx];  // couldve use array itself as it acts as pointer and pass it to cuda  


    for(i = 0; i < Nx; ++i)         //2d matrix so we create row pointer 
        Gmtx[i] = new double[Nx+1]; //and each row has pointer for corresponding column vector
                                    

    // construct 2 opposite-moving Guassian beams
    Nh = (int) N/2;
    for (i=0 ; i < N ; i++)
    {   
        pos[i] = uniform_distribution(generator) * boxsize; //Particles are positioned in box with uniform distirbution
        vel[i] = vth * norm_distribution(generator) + vb;   //Velocities of particles are given normal distirbution
    }   

    for (i=Nh ; i < N ; i++)
        vel[i] *= -1;                                      //half of the partivles are given -ve vel

    // add perturbation
    for (i=0 ; i < N ; i++)
        vel[i] *= (1 + A*sin(2*PI*pos[i]/boxsize));     //particles are perturbated with different velocity

    
	// Construct matrix G to compute Gradient which is E (1st derivative)
    for (i=0 ; i < Nx ; i++)
    {   
        for (j=0 ; j < Nx ; j++)
        {   
            Gmtx[i][j] = 0;
            if (i == j+1) Gmtx[i][j] =  1/(2*dx);      
            if (i+1 == j) Gmtx[i][j] = -1/(2*dx);
        }
    } 
    Gmtx[0][Nx-1] =  1/(2*dx);
	Gmtx[Nx-1][0] = -1/(2*dx);

    // Just intializing fields
    getAcc( E, phi_grid, pos, N, Nx, boxsize, n0, Gmtx );

	// number of timesteps
	Nt = (int)(ceil(tEnd/dt));


    // Simulation main loop
    auto start = high_resolution_clock::now(); //starting wall clock
    for (int niter = 0 ; niter < Nt ; niter++) 
    {

        for(j = 0 ; j < N ; j++)
        {
            vel[j] += 0.5*dt * (-E[j]); // (1/2) kick  //particle velocity gets updated

            //  drift (and apply periodic boundary conditions)
            pos[j] += vel[j] * dt;          //particle position gets updated
            pos[j] = fmod(pos[j] , boxsize);    // if the particle moves out of box , it throws it to the intial as periodic does 
                                                //i.e if new ps is 51 it will be updated as 1
            if(pos[j] < 0)
                pos[j] = pos[j] + 50.0;
        }                                      // if new posn is -ve it will throw correspondinly from other side of box

        // update accelerations
        getAcc( E, phi_grid, pos, N, Nx, boxsize, n0, Gmtx );

        for(j = 0 ; j < N ; j++)
        {
            vel[j] += 0.5*dt * (-E[j]); // (1/2) kick  // Giving two kicks, used as leap frog method.

            // update time
            t += dt;
        }
    }
    auto stop=high_resolution_clock::now();
    auto duration=duration_cast<milliseconds>(stop-start); //stopping wall clock

    #ifdef device_GPU
    cout<<"Time taken by GPU "<<duration.count() <<" ms" <<endl;
    #endif

    #ifdef device_CPU
    cout<<"Time taken by CPU "<<duration.count() <<" ms" <<endl;
    #endif

    // Create and open a text file
    ofstream MyFile("PICresult.txt");

    // Write to the file
    cout << "Writing data"<<endl;
    for(j = 0 ; j < N ; j++)
        MyFile << pos[j] << "\t" << vel[j] << endl;
    
    // Close the file
    MyFile.close();
    
    // cleanup
    for(i = 0; i < Nx; ++i)
        delete[] Gmtx[i];
        
    delete[] Gmtx;
    delete[] pos;
    delete[] vel;
    delete[] E;
    delete[] bvec;
    delete[] phi_grid;

} 

double getAcc( float* E, double* phi_grid, double* pos, int N, int Nx, double boxsize, int n0, double** Gmtx )
{   
    /*
    Calculate the acceleration on each particle due to electric field
	pos      is an Nx1 matrix of particle positions
	Nx       is the number of mesh cells
	boxsize  is the domain [0,boxsize]
	n0       is the electron number density
	Gmtx     is an Nx x Nx matrix for calculating the gradient on the grid
	Lmtx     is an Nx x Nx matrix for calculating the laplacian on the grid
	a        is an Nx1 matrix of accelerations
	*/

    // Calculate Electron Number Density on the Mesh by 
	// placing particles into the 2 nearest bins (j & j+1, with proper weights)
	// and normalizing
    
    int i, j, pidx[N], pidx1[N];
    double n[Nx], E_grid[Nx], phi_grid_old[Nx], error, Eg_temp[Nx], weight_j[N], weight_jp1[N], bvec[Nx],Ep_temp[N];

    double dx  = (double)boxsize / Nx;  // mesh cell dim
    for (i = 0 ; i < Nx ; i++)
        n[i] = 0;

    // distributing charge from particles to the grid based on weights
    for (i = 0 ; i < N ; i++)
    {
        pidx[i] = (int)floor(pos[i]/dx);                    //Getting left node id for each particle
        pidx1[i] = pidx[i] + 1;                             //Getting right node id for each particle
        weight_j[i]   = ( pidx1[i]*dx - pos[i]  )/dx;       //Getting weights for left node based on position of particle
        weight_jp1[i] = ( pos[i]      - pidx[i]*dx )/dx;    //Getting weights for Right node based on position of particle
        pidx1[i] = pidx1[i] % Nx;                           //right most node is updated to zero

        n[pidx[i]] += weight_j[i];                          // n is grid dimensions, output of Pidx would be between the grid 0 to Nx and thus we summ  
        n[pidx1[i]] += weight_jp1[i];                       // weights on each node based on particle posn here. We have now taken charge from particle to grid.
    }

   // for solving laplacian where 2nd order grad potential 
   //is equaivalent to n-n0 (net charge density) 
    for (i = 0 ; i < Nx ; i++)              
    {
        n[i] *= (double)n0 * boxsize / N / dx;              
        bvec[i] = (n[i] - n0)*dx*dx;
        phi_grid_old[i]=phi_grid[i]; // so that we use Jacobi, previous values are used to update.
    }

    // Solve Poisson's Equation: laplacian(phi) = n-n0
#ifdef device_GPU

    get_grid_Efield(E, phi_grid ,bvec, Gmtx, weight_j, weight_jp1, pidx, pidx1);  

  
#endif

#ifdef device_CPU    // preprocessor wont compile this part if device_CPU not defined in input file
    for (j = 0 ; j < 10000 ; j++)
    //Performin Jacobi Iteratively to find electric field from 2nd order potential
    {
        error = 0;
        for (i = 1 ; i < Nx-1 ; i++)
        {   
            phi_grid[i] = 0.5*(phi_grid_old[i-1] + phi_grid_old[i+1] - bvec[i]);
            error += abs(phi_grid_old[i] - phi_grid[i]);
        }
        error += abs(phi_grid_old[0] - 0.5*(phi_grid_old[Nx-1] + phi_grid_old[1] - bvec[0])) + abs(phi_grid_old[Nx-1] - 0.5*(phi_grid_old[Nx-2] + phi_grid_old[0] - bvec[Nx-1]));
        phi_grid[0]    = 0.5*(phi_grid_old[Nx-1] + phi_grid_old[1] - bvec[0]);
        phi_grid[Nx-1] = 0.5*(phi_grid_old[Nx-2] + phi_grid_old[0] - bvec[Nx-1]);

        // cout << phi_grid[1] << " " << error << endl;
        for (i = 0 ; i < Nx ; i++)
            phi_grid_old[i] = 0.7*phi_grid[i] + (1-0.7)*phi_grid_old[i];

        if(error < 1e-6) break;

    }
    cout << "Jacobi_iter_stopped_at" << j << endl;



    //updating E field from Phi obtained above
    for (i = 0 ; i < Nx ; i++)
    {
        Eg_temp[i] = 0;
        for (j = 0 ; j < Nx ; j++)
        {
            Eg_temp[i] += Gmtx[i][j]*phi_grid[j];
        }
        // cout<< Eg_temp[i] <<"  " << E_grid[i]<<endl;
        E_grid[i] = Eg_temp[i];
    }
    // cin.get();

    // Moving E from nodes to particles
    for (i = 0 ; i < N ; i++)
    {
        Ep_temp[i] = weight_j[i] * E_grid[pidx[i]] + weight_jp1[i] * E_grid[pidx1[i]];
       // cout<< Ep_temp[i] <<"  " << E[i]<<endl;
        E[i] = Ep_temp[i];
    }
   // cin.get();

#endif

    return 0;
}

