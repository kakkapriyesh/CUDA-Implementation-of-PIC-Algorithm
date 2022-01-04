# CUDA-Implementation-of-PIC-Algorithm
This is a learning exercise to use GPU for scientific computing. The code is written in CUDA and C++ to simulate two electron beams passing through each other in opposite directions. This interesting phenomenon is called the Two-Stream Instability.

# Details
The code is direct implemntation of [blog](https://medium.com/swlh/create-your-own-plasma-pic-simulation-with-python-39145c66578b) which is written in python.

Extensive discription of the files and the results are given in the documentation.pdf attached with this repository. 

# Building

The dependencies are- Cuda , gcc and python 3.0

Go to the directory and run the shell script 

```
./run.sh
```

The shell script executes following commands:
a)compile the files 
```
nvcc *.cu main.cpp -std=c++11
```
b)runs the output a.out
```
./a.out
```
c)plots the data from the text file of outputs(PICresult.txt) using python code in (plot.py)
```
python plot.py
```

To run the simulation on CPU:

Just change the input.h to define (device_CPU) file and run the shell script run.sh (./run.sh)
