/*  Author:Suman Kumar
	Date: 25th April 2018
	Problem: Matrix-Matrix Multiplication */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void read_file(int row,int col,FILE *fp,double *array);
void prod(int p,int *size,double *mat1,double *mat2,double *res,int rank);
void write2file(int *size,double *global_res);
int main(int argc,char* argv[])
{
	int p,rank,root=0,size[4];
	FILE *fmat1,*fmat2;
	double *mat1,*mat2,*local_mat1,*local_mat2,*local_res,*global_res;
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&p);
	if (rank==root)
	{
		fmat1 = fopen("matrix1.txt","r");
		fmat2 = fopen("matrix2.txt","r");
		fscanf(fmat1,"%d %d",&size[0],&size[1]);//size[0]=row1;size[1]=col1
		fscanf(fmat2,"%d %d",&size[2],&size[3]);//size[2]=row2;size[3]=col2
		mat1= (double*) malloc(size[0]*size[1]*sizeof(double));
		mat2= (double*) malloc(size[2]*size[3]*sizeof(double));
		read_file(size[0],size[1],fmat1,mat1);
		read_file(size[2],size[3],fmat2,mat2);
		fclose(fmat1);fclose(fmat2);
	}
	MPI_Bcast(size,4,MPI_INT,root,MPI_COMM_WORLD);
	local_mat1 = (double*) malloc(sizeof(double)*size[0]*size[1]/p);//local matrix 1
	local_mat2 = (double*) malloc(sizeof(double)*size[2]*size[3]/p);//locam matrix 2
	MPI_Scatter(mat1,size[0]*size[1]/p,MPI_DOUBLE,local_mat1,size[0]*size[1]/p,MPI_DOUBLE,root,MPI_COMM_WORLD);//distributr mat1
	MPI_Scatter(mat2,size[2]*size[3]/p,MPI_DOUBLE,local_mat2,size[2]*size[3]/p,MPI_DOUBLE,root,MPI_COMM_WORLD);//distribute mat2
	if(rank==0) {free(mat1);free(mat2);}//No Longer Required so free it
	local_res = (double*) malloc(sizeof(double)*size[0]*size[3]/p);//local_res
	prod(p,size,local_mat1,local_mat2,local_res,rank);//Multiplication
	free(local_mat1);free(local_mat2);//free
	global_res =(double*) malloc(sizeof(double)*size[0]*size[3]);//Global Result
	MPI_Gather(local_res,size[0]*size[3]/p,MPI_DOUBLE,global_res,size[0]*size[3]/p,MPI_DOUBLE,root,MPI_COMM_WORLD);//Gather all Result
	if(rank==0){write2file(size,global_res);}
	free(local_res);
	free(global_res);
	MPI_Finalize();
}

void read_file(int row,int col,FILE *fp,double *array)
{
	int i,j;
	for(i=0;i<row;i++){
		for(j=0;j<col;j++){
			fscanf(fp,"%lf",&array[i*col+j]);}}
			
}

void prod(int p,int *size,double *mat1,double *mat2,double *res,int rank)
{
	int i,j,k,l,circle_iter,dest,send_tag=0,source,recv_tag=0,index2;
	MPI_Status status;
	dest=rank-1;
	source=rank+1;
	if (rank==0){dest=p-1;}
	if (rank==p-1){source=0;}
	for(circle_iter=0;circle_iter<p;circle_iter++)//p-1 rotation of local_mat2 and corresponding cicle_iter'th piece of row of local_mat1
	{
	index2=rank+circle_iter;
	if (index2>p-1){index2=index2-p;}
	for(i=0;i<size[0]/p;i++)//for ith row of local_res and corresponding ith row of local_mat1
	{
		for(j=0;j<size[3];j++)//for jth col of local_res
		{
			for(k=0;k<size[1]/p;k++)//for kth row of local_mat2 and corresponding kth col of local_mat1
			{
				res[i*size[1]+j]=res[i*size[1]+j]+mat1[i*size[1]+k+index2*size[1]/p]*mat2[k*size[3]+j];
					//printf("%lf\n",res[i*size[1]+j]);
					//if(rank==0){printf("%d %d %d\n",i*size[1]+j,i*size[1]+k+index2*size[1]/p,k*size[3]+j);}
			}
		}
	}
	MPI_Sendrecv_replace(mat2,size[2]*size[3]/p,MPI_DOUBLE,dest,send_tag,source,recv_tag,MPI_COMM_WORLD,&status);//circular shift
	}
}

void write2file(int *size,double *global_res)
{
	int i;
	FILE *fp;
	fp = fopen("result.txt","w");
	for(i=0;i<size[0]*size[3];i++)
	{
		fprintf(fp,"%lf\n",global_res[i]);
	}
	fclose(fp);
}






















	
