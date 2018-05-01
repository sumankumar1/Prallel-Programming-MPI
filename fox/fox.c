/*  Author:Suman Kumar
	Date: 28th April 2018
	Revison: 2.0
	Problem: Matrix-Matrix Multiplication using Fox Algorithm
	Tested on MPI Version:1.10.2
	gcc version:4.8.5*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

typedef struct{
	int p;
	int rank;
	int dest;
	int source;
	int root;
	int size[4];
	int dimension[2];
	int my_row;
	int my_col;
	MPI_Comm cart_comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;} grid_info;
void read_size(FILE *fp,FILE *fp1,int *size);//just to read matrix size
void read_matrix(FILE *fp,int row,int col,double *mat);//read matrix into array from file
void setup_grid(grid_info* grid);
void fox_distribute(double *mat,double *local_mat,int row,int col,int p,int my_row,int my_col);
void fox_prod(double *mat1,double *mat2,double *res,grid_info *grid);
void fox_collect_print(double *local_mat,double *global_res,grid_info *grid);

int main(int argc,char* argv[])
{
	grid_info grid;grid.root=0;
	double *mat1,*mat2,*local_mat1,*local_mat2,*local_res,*global_res;
	FILE *fmat1,*fmat2;//File Pointer Declaration
	fmat1 = fopen("mat1.txt","r");//File Pointer initialization
	fmat2 = fopen("mat2.txt","r");//File Pointer initialization
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&grid.p);
	grid.dimension[0]=sqrt(grid.p);grid.dimension[1]=sqrt(grid.p);
	setup_grid(&grid);
	//int i;MPI_Comm_rank(MPI_COMM_WORLD,&i);
	
	if(grid.rank==0){read_size(fmat1,fmat2,grid.size);}//Reading size....
	MPI_Bcast(grid.size,4,MPI_INT,grid.root,grid.cart_comm);//Broadcasting array size[4]....
	
	mat1 = (double*) malloc(grid.size[0]*grid.size[1]*sizeof(double));//mat1
	if(grid.rank==0){read_matrix(fmat1,grid.size[0],grid.size[1],mat1);fclose(fmat1);}//Declare mat1 and read into it
	MPI_Bcast(mat1,grid.size[0]*grid.size[1],MPI_DOUBLE,grid.root,grid.cart_comm);//Broadcast full matrix
	local_mat1 = (double*) malloc((grid.size[0]/grid.p)*grid.size[1]*sizeof(double));//Declare local_mat1
	fox_distribute(mat1,local_mat1,grid.size[0],grid.size[1],grid.p,grid.my_row,grid.my_col);free(mat1);//R
	
	mat2 = (double*) malloc(grid.size[2]*grid.size[3]*sizeof(double));//mat2
	if(grid.rank==0){read_matrix(fmat2,grid.size[2],grid.size[3],mat2);fclose(fmat2);}//Declare mat2 and read into it
	MPI_Bcast(mat2,grid.size[2]*grid.size[3],MPI_DOUBLE,grid.root,grid.cart_comm);//Broadcast full matrix
	local_mat2 = (double*) malloc((grid.size[2]/grid.p)*grid.size[3]*sizeof(double));//Declare local_mat2
	fox_distribute(mat2,local_mat2,grid.size[2],grid.size[3],grid.p,grid.my_row,grid.my_col);free(mat2);//Read into local_mat1 and free memory from full matrix
	
	local_res = (double*) malloc((grid.size[0]/grid.p)*grid.size[3]*sizeof(double));//Declare local_res
	fox_prod(local_mat1,local_mat2,local_res,&grid);//fox algorithm for multiplication	
	free(local_mat1);free(local_mat2);//free memory from unused array 
	global_res = (double*) malloc(grid.size[0]*grid.size[3]*sizeof(double));//Declare global result array
	
	fox_collect_print(local_res,global_res,&grid);//collect into global result & print into file
	free(local_res);free(global_res);
	MPI_Finalize();
}

void read_size(FILE *fp,FILE *fp1,int *size)
{
	int i;
	for (i=0;i<2;i++){fscanf(fp,"%d",&size[i]);}
	for (i=2;i<4;i++){fscanf(fp1,"%d",&size[i]);}
}

void read_matrix(FILE *fp,int row,int col,double *mat)
{
	int i,j;
	for(i=0;i<row;i++){for(j=0;j<col;j++){fscanf(fp,"%lf",&mat[i*col+j]);}}
}

void setup_grid(grid_info* grid)
{
	MPI_Comm comm1;
	int wrap_around[2];//circular shift allowed if value is 1
	int free_cords[2];//row or column grid
	int coordinates[2];
	wrap_around[0]=1;
	wrap_around[1]=1;
	MPI_Cart_create(MPI_COMM_WORLD,2,grid->dimension,wrap_around,1,&grid->cart_comm);//grid->cart_comm=comm1;
	MPI_Comm_rank(grid->cart_comm,&grid->rank);
	MPI_Cart_coords(grid->cart_comm,grid->rank,2,coordinates);grid->my_row=coordinates[0];grid->my_col=coordinates[1];
	free_cords[0]=0;//row comm by fixing row
	free_cords[1]=1;//Not Fixed
	MPI_Cart_sub(grid->cart_comm,free_cords,&grid->row_comm);
	free_cords[0]=1;//Not Fixed
	free_cords[1]=0;//col comm by fixing col
	MPI_Cart_sub(grid->cart_comm,free_cords,&grid->col_comm);
}

void fox_distribute(double *mat,double *local_mat,int row,int col,int p,int my_row,int my_col)
{
	int i,j,k,row1,col1;
	row1=row/sqrt(p);col1=col/sqrt(p);

	for(i=0;i<row1;i++)
	{
		for(j=0;j<col1;j++)
		{
			local_mat[i*col1+j]=mat[(i+my_row*row1)*col+j+(my_col*col1)];//simply copy into local
		}
	}
}


void fox_prod(double *mat1,double *mat2,double *res,grid_info *grid)
{
	int row1,col1,row2,col2,col3,p,rank,i,j,k,l,index,source,dest;
	p=sqrt(grid->p);
	row1=grid->size[0]/p;col1=grid->size[1]/p;row2=grid->size[2]/p;col2=grid->size[3]/p;
	double *temp =(double*) malloc(row1*col1*sizeof(double));
	MPI_Status status;
	dest=grid->my_row-1;if(dest<0){dest=p-1;}//calculating dest for circular shift
	source=grid->my_row+1;if(source>p-1){source=0;}//calculating source for circular shift
	for(k=0;k<p;k++)
	{	
		col3=grid->my_row+k;if(col3>p-1){col3=col3-p;}//which local_mat1 should be sent to its row_comm
		if(grid->my_col==col3){memcpy(temp,mat1,row1*col1*sizeof(double));}//temporary matrix for local_mat1 broadcasting
		MPI_Bcast(temp,row1*col1,MPI_DOUBLE,col3,grid->row_comm);//Broadcasting required local_mat1
		for(i=0;i<row1;i++){
			for(j=0;j<col2;j++){index =i*col2+j;
				for(l=0;l<col1;l++)//col1=row2=number of multiplication per element of res
				{
				res[index]=res[index]+temp[i*col1+l]*mat2[l*col2+j];//product
				}
			}
		}
		
	if(k!=p-1){MPI_Sendrecv_replace(mat2,row2*col2,MPI_DOUBLE,dest,0,source,0,grid->col_comm,&status);}//circular shift of local_mat2
	}
	free(temp);
}

void fox_collect_print(double *local_res,double *global_res,grid_info *grid)
{
	int i,j,k,row1,col1,my_row,my_col,g;
	MPI_Status status;
	row1=grid->size[0]/sqrt(grid->p);
	col1=grid->size[3]/sqrt(grid->p);
	double *temp =(double*) malloc((row1*col1+3)*sizeof(double));
	if(grid->rank!=0)
	{
		memcpy(temp,local_res,row1*col1*sizeof(double));//packing for least communication 
		temp[row1*col1+1]=grid->my_row;//packing for least communication 
		temp[row1*col1+2]=grid->my_col;//packing for least communication 
		MPI_Send(temp,row1*col1+3,MPI_DOUBLE,0,0,grid->cart_comm);}//sending packet
	else
	{ 
		for(k=1;k<grid->p;k++)
		{
			MPI_Recv(temp,row1*col1+3,MPI_DOUBLE,k,0,grid->cart_comm,&status);//packet recieved
			my_row=temp[row1*col1+1];my_col=temp[row1*col1+2];//unpacked
			for(i=0;i<row1;i++){for(j=0;j<col1;j++){global_res[(i+my_row*row1)*grid->size[3]+j+(my_col*col1)]=temp[i*col1+j];}}//copying into global result array
			//printf("%lf %lf %d %d %d\n",global_res[(i+my_row*row1)*grid->size[3]+j+(my_col*col1)],temp[i*col1+j],(i+my_row*row1)*grid->size[3]+j+(my_col*col1),i*col1+j,k);}}
		}
		for(i=0;i<row1;i++){for(j=0;j<col1;j++){global_res[(i+grid->my_row*row1)*grid->size[3]+j+(grid->my_col*col1)]=local_res[i*col1+j];}}
		//printf("%lf %lf %d %d %d\n",global_res[(i+grid->my_row*row1)*grid->size[3]+j+(grid->my_col*col1)],local_res[i*col1+j],(i+grid->my_row*row1)*grid->size[3]+j+(grid->my_col*col1),i*col1+j,grid->rank);}}
	}	
	FILE *fp;
	fp=fopen("result.txt","w");
	if(grid->rank==0){for(i=0;i<row1*col1*grid->p;i++){fprintf(fp,"%lf\n",global_res[i]);}}//printing
	fclose(fp);
}







