/*  Author: Suman Kumar
	Revision: 1.0
	Date: 11 May 2018
	Problem: Solution of 2D Poisson's equation with SOR 
	Programmed on MPI Version:1.10.2
	gcc version:4.8.5
	**Test Pending*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include "func.h"
#include "boundary.h"

typedef struct 
{
	int p;
	int rank;
	int source;
	int dest;
	int root;
	int x_steps;
	int y_steps;
	int dim[2];
	int ndims;
	int periodic[2];
	int reorder;
	int my_row;
	int my_col;
	MPI_Comm cart_comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;
} grid_info_t;

void create_2D_comm(grid_info_t *grid);
void sor(grid_info_t *grid,double *u,double (*f)(),double (*g)(),float w,double tol,double *res);


int main(int agrc,char* argv[])
{
	int xs,ys,tot,i;
	double *u,*res,tol=pow(10,-3),w=1.5;
	double (*f)(double,double)=&func;
	double (*g)(double,double)=&boundary;
	grid_info_t grid;
	grid.root=0;grid.ndims=2;grid.reorder=1;
	grid.periodic[0]=grid.periodic[1]=1;
	MPI_Status status;
	MPI_Init(&agrc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&grid.p);
	grid.dim[0]=grid.dim[1]=sqrt(grid.p);
	create_2D_comm(&grid);
	grid.x_steps=4;//total discetized point in x without boundary
	grid.y_steps=4;//total discetized point in y without boundary
	xs=grid.x_steps/sqrt(grid.p);
	ys=grid.y_steps/sqrt(grid.p);
	tot=(xs+2)*(ys+2);//4 corner element + 2 xs +2 ys + total without bounday
	u = (double*) malloc(tot*sizeof(double));
	res=(double*) malloc(grid.x_steps*grid.y_steps*sizeof(double));
	for(i=0;i<tot;i++){u[i]=0;}
	sor(&grid,u,f,g,w,tol,res);
	FILE *fp;
	fp = fopen("result.txt","w");
	if(grid.rank==0){for(i=0;i<grid.y_steps*grid.x_steps;i++){fprintf(fp,"%lf\n",res[i]);}}
	fclose(fp);
	free(u);
	free(res);
	MPI_Finalize();
}

void create_2D_comm(grid_info_t *grid)
{
	int free_coord[2];int my_coord[2];
	MPI_Cart_create(MPI_COMM_WORLD,grid->ndims,grid->dim,grid->periodic,grid->reorder,&grid->cart_comm);
	MPI_Comm_rank(grid->cart_comm,&grid->rank);
	MPI_Cart_coords(grid->cart_comm,grid->rank,2,my_coord);
	grid->my_row = my_coord[0];grid->my_col = my_coord[1];
	free_coord[0]=0;
	free_coord[1]=1;
	MPI_Cart_sub(grid->cart_comm,free_coord,&grid->row_comm);
	free_coord[0]=1;
	free_coord[1]=0;
	MPI_Cart_sub(grid->cart_comm,free_coord,&grid->col_comm);
}

void sor(grid_info_t *grid,double *u,double (*f)(),double (*g)(),float w,double tol,double *res)
{
	int i,j,k,xs,ys,p1,count=1,flag=0;
	double xh,yh;
	MPI_Status status;
	p1=sqrt(grid->p);
	xs=grid->x_steps/p1;
	ys=grid->y_steps/p1;
	double error=10,error_prev,*temp,*up,*down,*left,*right,*error_vec;
	xh=1.0/(grid->x_steps+1);
	yh=1.0/(grid->y_steps+1);
	
	if (grid->my_row==0){for(i=1;i<xs+1;i++){u[i]=g((grid->my_col*xs+i)*xh,0.0);}}//creating local_red_black
	if (grid->my_row==p1-1){for(i=1;i<xs+1;i++){u[(xs+2)*(ys+1)+i]=g((grid->my_col*xs+i)*xh,1.0);}}//creating local_red_black
	if (grid->my_col==0){for(i=1;i<xs+1;i++){u[(xs+2)*i]=g(0.0,(grid->my_row*ys+i)*yh);}}//creating local_red_black
	if (grid->my_col==p1-1){for(i=1;i<xs+1;i++){u[(xs+2)*(i+1)-1]=g(1.0,(grid->my_row*ys+i)*yh);}}//creating local_red_black
	
	temp = (double*) malloc((xs+2)*(ys+2)*sizeof(double));//to store previous value
	
	up = (double*) malloc(xs*sizeof(double));//for exchange of boundary elements
	down = (double*) malloc(xs*sizeof(double));//for exchange of boundary elements
	left = (double*) malloc(ys*sizeof(double));//for exchange of boundary elements
	right = (double*) malloc(ys*sizeof(double));//for exchange of boundary elements
	error_vec =(double*) malloc(grid->p*sizeof(double));//To Gather Error from all processor for termination of loop correctly w.r.t tolerance
	
	while (flag==0)
	{
		flag=1;//Terminator
		memcpy(temp,u,(xs+2)*(ys+2)*sizeof(double));//copying matrix
		//for(i=0;i<(xs+2)*(ys+2);i++){printf("%d  %d %lf\n",i,grid->rank,temp[i]);}
		
		for(i=1;i<ys+1;i++){right[i]=u[(xs+2)*(i+1)-2];}//boundary values to be exchanged
		for(i=1;i<ys+1;i++){left[i]=u[(xs+2)*i+1];}//boundary values to be exchanged
		for(i=1;i<xs+1;i++){down[i]=u[(xs+2)*ys+i];}//boundary values to be exchanged
		for(i=1;i<xs+1;i++){up[i]=u[xs+2+i];/*printf("%lf  %d\n",up[i],grid->rank);*/}//boundary values to be exchanged
		//for(i=0;i<(xs+2)*(ys+2);i++){printf("%d  %d %lf\n",i,grid->rank,u[i]);}
		
		if(grid->my_row!=0){MPI_Send(up,xs,MPI_DOUBLE,grid->my_row-1,0,grid->col_comm);}//handshake
		if(grid->my_row!=p1-1){MPI_Send(down,xs,MPI_DOUBLE,grid->my_row+1,1,grid->col_comm);}//handshake
		if(grid->my_row!=p1-1){MPI_Recv(down,xs,MPI_DOUBLE,grid->my_row+1,0,grid->col_comm,&status);}//handshake
		if(grid->my_row!=0){MPI_Recv(up,xs,MPI_DOUBLE,grid->my_row-1,1,grid->col_comm,&status);}//handshake

		if(grid->my_col!=0){MPI_Send(left,ys,MPI_DOUBLE,grid->my_col-1,0,grid->row_comm);}//handshake
		if(grid->my_col!=p1-1){MPI_Send(right,ys,MPI_DOUBLE,grid->my_col+1,1,grid->row_comm);}//handshake
		if(grid->my_col!=p1-1){MPI_Recv(right,ys,MPI_DOUBLE,grid->my_col+1,0,grid->row_comm,&status);}//handshake
		if(grid->my_col!=0){MPI_Recv(left,ys,MPI_DOUBLE,grid->my_col-1,1,grid->row_comm,&status);}//handshake
		
		if(grid->my_col!=p1-1){for(i=1;i<ys+1;i++){u[(xs+2)*(i+1)-1]=right[i];}}//exchange included in local_red_black
		if(grid->my_col!=0){for(i=1;i<ys+1;i++){u[(xs+2)*i]=left[i];}}//exchange included in local_red_black
		if(grid->my_row!=p1-1){for(i=1;i<xs+1;i++){u[(xs+2)*(ys+1)+i]=down[i];}}//exchange included in local_red_black
		if(grid->my_row!=0){for(i=1;i<xs+1;i++){u[i]=up[i];}}//exchange included in local_red_black
		//for(i=0;i<(xs+2)*(ys+2);i++){printf("%d  %d %lf\n",i,grid->rank,u[i]);}
		/**************SOR CALCULATION STARTED*************************/
		for(i=1;i<ys+1;i++)
		{
			for(j=1;j<xs+1;j=j+2)
			{
				xs=xs+2;
				u[i*xs+j] = u[i*xs+j]+ 0.25*w*(u[(i-1)*xs+j] + u[(i+1)*xs+j] + u[i*xs+(j-1)] + u[i*xs+(j+1)] - 4*u[i*xs+j] + f(j*xh,i*yh));
				xs=xs-2;
			}
		}
		for(i=1;i<ys+1;i++)
		{
			for(j=2;j<xs+1;j=j+2)
			{
				xs=xs+2;
				u[i*xs+j] = u[i*xs+j]+ 0.25*w*(u[(i-1)*xs+j] + u[(i+1)*xs+j] + u[i*xs+(j-1)] + u[i*xs+(j+1)] - 4*u[i*xs+j] + f(j*xh,i*yh));
				xs=xs-2;
			}
		}
		error=tol;
		for(i=0;i<(xs+2)*(ys+2);i++)
		{
			error_prev=fabs(temp[i]-u[i]);
			if (error_prev>error){error=error_prev;}
			//if (grid->rank==0 && count==1){printf("%d %lf\n",count,error_prev);}
		}
		count+=1;
		//if (grid->rank==0){printf("%d %lf\n",count,error);}
		//if (grid->rank==0){printf("I am here\n");}
		MPI_Gather(&error,1,MPI_DOUBLE,error_vec,1,MPI_DOUBLE,0,grid->cart_comm);
		if (grid->rank==0){for(i=0;i<grid->p;i++){if(error_vec[i]>tol){flag=0;}}}
		MPI_Bcast(&flag,1,MPI_INT,0,grid->cart_comm);
		/**********************SOR CALCULATION COMPLETED********************************/
	}
	// Gathering Local Values................................................
	u[0]=grid->my_row;u[xs+1]=grid->my_col;//both corner are not used in calculation so used in transportation
	int my_row,my_col;
	if(grid->rank!=0){MPI_Send(u,(xs+2)*(ys+2),MPI_DOUBLE,0,0,grid->cart_comm);}
	else{for(i=0;i<grid->p;i++)
	{
		if(i!=0){MPI_Recv(u,(xs+2)*(ys+2),MPI_DOUBLE,i,0,grid->cart_comm,&status);}
		my_row=u[0];my_col=u[xs+1];
		for(j=1;j<ys+1;j++)
		{
			for(k=1;k<xs+1;k++)
			{
				res[(j-1)*grid->x_steps+k-1+my_row*grid->x_steps*ys+my_col*xs]=u[j*(xs+2)+k];
				//printf("%lf\n",res[(j-1)*grid->x_steps+k-1+my_row*grid->x_steps*ys+my_col*xs]);
				//printf("%d  %d  %d\n",(j-1)*grid->x_steps+k-1+my_row*grid->x_steps*ys+my_col*xs,j*(xs+2)+k,i);
			}
		}
	}}
	// Gathering Finished!!
	free(temp);free(up);free(down);free(left);free(right);free(error_vec);//Free Memory
}
	















