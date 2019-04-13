# Prallel Programming with MPI in C

compile:
mpicc FILE_NAME -lm
Run:
mpirun -np #process ./Executable

Example
mpicc jacobi.c -lm
mprirun -np 4 ./a.out
