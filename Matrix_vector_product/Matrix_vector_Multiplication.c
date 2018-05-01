/* Author: Suman Kumar
   Date: 24th April 2018 
   Revision 2.0
   Email:suman.kumar1@outlook.com
   
   About the Program:
   Matrix vector multiplication
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void read_file(int row,int col,FILE *file,double *array);
void prod(int mat_row,int vec_row,int p,double *local_mat,double *vec,double *local_res);
void write2file(int mat_row,double *global_res);
int main(int argc,char* argv[])
{
	int p,rank,dest,tag,root=0,source,mat_col,mat_row,vec_row,vec_col,size[3];
	double *mat,*vec,*global_res;
	FILE *fmat,*fvec;
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&p);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	if (rank==0)
	{
		fmat=fopen("matrix.txt","r");
		fvec=fopen("vector.txt","r");
		fscanf(fmat,"%d %d",&mat_row,&mat_col);
		fscanf(fvec,"%d %d",&vec_row,&vec_col);
		mat = (double*) malloc(mat_col*mat_row*sizeof(double));//full matrix array on 0
		read_file(mat_row,mat_col,fmat,mat);fclose(fmat);//read matrix into mat on 0
		size[0]=mat_row;size[1]=mat_col;size[2]=vec_row;//pack size variable to reduce communicaton instance
	}
	MPI_Bcast(size,3,MPI_INT,root,MPI_COMM_WORLD);
	mat_row=size[0];mat_col=size[1];vec_row=size[2];//unpacking
	vec = (double*) malloc(vec_row*sizeof(double));
	if (rank==0){read_file(vec_row,1,fvec,vec); fclose(fvec);}//read vector into vec
	MPI_Bcast(vec,vec_row,MPI_DOUBLE,root,MPI_COMM_WORLD);//broadcast vec
	double *local_mat = (double*) malloc((mat_row*mat_col/p)*sizeof(double));//local matrix
	MPI_Scatter(mat,mat_row*mat_col/p,MPI_DOUBLE,local_mat,mat_row*mat_col/p,MPI_DOUBLE,root,MPI_COMM_WORLD);//distribute full matrix
	if (rank==0){free(mat);}//free memory from full matrix
	double *local_res = (double*) malloc((mat_row/p)*sizeof(double));//local result
	prod(mat_row,vec_row,p,local_mat,vec,local_res);//multiply
	free(local_mat);free(vec);//free local mat and vector memory
	global_res = (double*) malloc(mat_row*sizeof(double));//global result matrix
	MPI_Gather(local_res,mat_row/p,MPI_DOUBLE,global_res,mat_row/p,MPI_DOUBLE,root,MPI_COMM_WORLD);//gather into global result matrix
	if(rank==0){write2file(mat_row,global_res);}//print into file
	free(global_res);
	MPI_Finalize();
}
//All Function def
void read_file(int row,int col,FILE *file,double *array)
{
	int i,j;
	for (i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			fscanf(file,"%lf",&array[i*col+j]);
		}
	}
}

void prod(int mat_row,int vec_row,int p,double *local_mat,double *vec,double *local_res)
{
	int i,j,index;
	for (i=0;i<(mat_row/p);i++)
	{
		for(j=0;j<vec_row;j++)
		{	
			index=i*vec_row+j;
			local_res[i]=local_res[i]+local_mat[index]*vec[j];
		}
	}
}

void write2file(int mat_row,double *global_res)
{
	int i;
	FILE *fres;
	fres = fopen("result.txt","w");
	for (i=0;i<mat_row;i++)
	{
		fprintf(fres,"%lf\n",global_res[i]);
	}
	fclose(fres);
}









