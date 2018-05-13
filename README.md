# Prallel_Programming_MPI_OpenMP_C

compile:
mpicc FILE_NAME -lm
Run:
mpirun -np #process ./Executable

Example
mpicc jacobi.c -lm
mprirun -np 4 ./a.out
