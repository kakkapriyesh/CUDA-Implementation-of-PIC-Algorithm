
#define device_GPU  // device_GPU or device_CPU (or Both)
//#define device_CPU

// Thread block size
#define Thread_grid  100  // number of threads in a block, number of blocks is mesh_size/no of threads
#define Thread_particle 1000 //thread size based on particle, (dont go beyond 1000 threads/GPU capacity)
#define M_HEIGHT    1 // number of rows

// Simulation parameters
#define No_Of_Particles   	 40000     // Number of particles
#define No_Of_Mesh_Cell         100      // Number of mesh cells
#define sim_end_time	         50      // time at which simulation ends
#define timestep		 1       // timestep
#define electron_no_density	 1       // electron number density
#define beam_vel		 3       // beam velocity
#define beam_width		 1       // beam width
#define pertub			 0.1     // perturbation
#define box_size    		 50      // periodic domain [0,boxsize]
#define iterations_Jacobi           10000  // iterations to be done in gauss siedel
	
	
