import numpy as np
import matplotlib.pyplot as plt

"""
Create Your Own Plasma PIC Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the 1D Two-Stream Instability
Code calculates the motions of electron under the Poisson-Maxwell equation
using the Particle-In-Cell (PIC) method

"""


	


def main():
	""" Plasma PIC simulation """

	# Simulation parameters
	# N         = 100   # Number of particles
	# tEnd      = 50      # time at which simulation ends
	# dt        = 1       # timestep
	boxsize   = 50

	# plotRealTime = False
	
	# # number of timesteps
	# Nt = int(np.ceil(tEnd/dt))

	f = open('PICresult.txt', 'r')
	pos, vel = [], []
	N = 0
	for line in f.readlines():
		fields = line.split('\t')
		pos.append(np.double(fields[0]))
		vel.append(np.double(fields[1]))
		N += 1

	Nh = int(N/2)
	

	# with open('PICresult.txt','r') as f:
	# 	for i in range(N):
	# 		lines = f.readlines(1)
	# 		# pos[i], vel[i] = lines
	# 		print(i, lines)


	
	# prep figure
	fig = plt.figure(figsize=(5,4), dpi=80)
	
	# Simulation Main Loop
	# for i in range(Nt):
		
	# 	# plot in real time - color 1/2 particles blue, other half red
	# 	if plotRealTime or (i == Nt-1):
	# 		plt.cla()
	# 		plt.scatter(pos[0:Nh],vel[0:Nh],s=.4,color='blue', alpha=0.5)
	# 		plt.scatter(pos[Nh:], vel[Nh:], s=.4,color='red',  alpha=0.5)
	# 		plt.axis([0,boxsize,-6,6])
			
	# 		plt.pause(0.001)
			
	
	plt.cla()
	plt.scatter(pos[0:Nh],vel[0:Nh],s=.4,color='blue', alpha=0.5)
	plt.scatter(pos[Nh:], vel[Nh:], s=.4,color='red',  alpha=0.5)
	plt.axis([0,boxsize,-6,6])

	# Save figure
	plt.xlabel('x')
	plt.ylabel('v')
	plt.savefig('pic_output_t50.png',dpi=240)
	plt.show()
	    
	return 0
	


  
if __name__== "__main__":
  main()
