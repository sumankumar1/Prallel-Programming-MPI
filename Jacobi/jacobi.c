/* Author: Suman Kumar
   Date: 18th January 2018 */

#include<stdio.h>
#include<mpi.h>
#include<math.h>
#include<stdlib.h>
#include "f.h"
#include "g.h"

/*********************Function Declaration********************/
void initialize(double (*gunc)(),int my_rank,int p,int N,int n,double* u0,double* u1,double* bc0,double* bc1,double xa,double ya,double h);
void jacobi(double (*func)(),int my_rank,int p,int N,int n,double* u0,double* u1,double xa,double ya,double h,double tol);
void write2file(FILE* fp,double* out,int count);
/*********************Main Function***************************/
int main(int argc,char* argv[])
{
MPI_Init(&argc, &argv);
MPI_Status status;
double (*func)(double,double)=&f;//function pointer for function in poission's equation
double (*gunc)(double,double)=&g;//function pointer for boundary values
int N=8,n=0,p,my_rank,i,j;//N=n*p//Total Elements on Grid=(N+2)^2
double h,runtime;//Step Size hx=(xb-xa)/nx+1, hy=(yb-ya)/ny 
double xa=0,xb=1,ya=0,yb=1;//Domain
double *u0,*u1,*bc0,*bc1,*result,tol=pow(10,-10);//solution
MPI_Comm_size(MPI_COMM_WORLD,&p);
MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
n=N/p;//strong scaling
h=(xb-xa)/(N+1);ya=0+h;yb=1-h;//Domain1=Domain excluding boundary along y
ya=ya+h*n*my_rank;//left side(a) of domain [a,b] for each processor
yb=ya+h*(n-1);//right side(b) of domain [a,b] for each processor
u0 = (double *) malloc((N+2)*(n+2)*sizeof(double));//ny+2=2 extra row for boundary values from neighbouring processor
u1 = (double *) malloc((N+2)*(n)*sizeof(double));//ny+2=2 extra row for boundary values from neighbouring processor
bc0 = (double *) malloc((N+2)*sizeof(double));//boundary value at ymin
bc1 = (double *) malloc((N+2)*sizeof(double));//boundary value at ymax
result = (double *) malloc((N+2)*N*sizeof(double));//all values except boundary at x=xmin and x=xmax
/*****All Calculation*****/
initialize(*gunc,my_rank,p,N,n,u0,u1,bc0,bc1,xa,ya,h);//calculate boundary values and Initialize other elements
if (my_rank==p-1){MPI_Send(bc1,N+2, MPI_DOUBLE,0, 0, MPI_COMM_WORLD);}//send to processor 0
if (my_rank==0)  {MPI_Recv(bc1,N+2, MPI_DOUBLE,p-1, 0,MPI_COMM_WORLD, &status);}//recieved at processor 0 for printing
MPI_Barrier(MPI_COMM_WORLD);//Barrier
runtime=MPI_Wtime();//start time
jacobi(*func,my_rank,p,N,n,u0,u1,xa,ya,h,tol);//Jacobi iteration
MPI_Barrier(MPI_COMM_WORLD);//barrier
runtime=MPI_Wtime()-runtime;//End Time
if (my_rank==0){printf("%lf\n",runtime);}//printing runtime
MPI_Gather(u1,(N+2)*n,MPI_DOUBLE,result,(N+2)*n,MPI_DOUBLE,0,MPI_COMM_WORLD);//gathering result to processor 0 without bc0 and bc1
/*****printing******/
if (my_rank==0)
	{
	FILE *res;
	if(fopen("result.txt","w")){remove("result.txt");}
	res=fopen("result.txt","w");
	write2file(res,bc0,N+2);
	write2file(res,result,(N+2)*N);
	write2file(res,bc1,N+2);
	fclose(res);
	}
free(u0);
free(u1);
free(bc0);
free(bc1);
free(result);
MPI_Finalize();
}
/**************************Intialize Function************************/
void initialize(double (*gunc)(),int my_rank,int p,int N,int n,double* u0,double* u1,double* bc0,double* bc1,double xa,double ya,double h)
{
int i,j;
for(i=0;i<n+2;i++)
{
	for(j=0;j<N+2;j++)
	{
		if (my_rank==0)
		{
			if (i==0){u0[i*(N+2)+j]=gunc(xa+j*h,ya+(i-1)*h);bc0[j]=u0[i*(N+2)+j];}
			else{if((j==0 || j==N+1)&& i!=n+1){u0[i*(N+2)+j]=gunc(xa+j*h,ya+(i-1)*h);}else{u0[i*(N+2)+j]=1;}}
		}
		else if(my_rank==p-1)
		{
			if (i==n+1){u0[i*(N+2)+j]=gunc(xa+j*h,ya+(i-1)*h);bc1[j]=u0[i*(N+2)+j];}
			else{if((j==0||j==N+1)&& i!=0){u0[i*(N+2)+j]=gunc(xa+j*h,ya+(i-1)*h);}else{u0[i*(N+2)+j]=1;}}
		}
		else
		{
			if((i!=0 && i!=n+1)&&(j==0||j==N+1)){u0[i*(N+2)+j]=gunc(xa+j*h,ya+(i-1)*h);}else{u0[i*(N+2)+j]=1;} 
			//printf("%d %d %d %lf\n",i,j,i*(N+2)+j,u0[i*(N+2)+j]);
			//printf("x:%lf y:%lf\n",xa+j*h,ya+(i-1)*h);
		}
		if(i!=0&&i!=n+1){u1[(i-1)*(N+2)+j]=u0[i*(N+2)+j];}//copy
	}
}
}
/*******************Jacobi*******************************************/
void jacobi(double (*func)(),int my_rank,int p,int N,int n,double* u0,double* u1,double xa,double ya,double h,double tol)
{
int i,j,k,hsquare,center1,center0,left,right,up,down,flag=1,iter=0;double error;
int *flagr = (int *) malloc(p*sizeof(int));
double *lower = (double *) malloc((N)*sizeof(double));
double *upper = (double *) malloc((N)*sizeof(double));
hsquare=h*h;
MPI_Status status;
while(flag==1)
{
	flag=0;
	for(i=1;i<n+1;i++)
	{
		for(j=1;j<N+1;j++)
		{	
			left=i*(N+2)+(j-1);//for u0
			right=i*(N+2)+(j+1);//for u0
			center1=(i-1)*(N+2)+j;//for u1
			center0=i*(N+2)+j;//for u0
			up=(i-1)*(N+2)+j;//for u0
			down=(i+1)*(N+2)+j;//for u0
			u1[center1]=0.25*(u0[left]+u0[right]+u0[up]+u0[down]+hsquare*func(xa+j*h,ya+(i-1)*h));
			error=fabs(u1[center1]-u0[center0]);//Estimate error
			if ((error>tol) && (flag==0)) {flag=1;}//Set flag to decide termination of jacob iteration
			u0[center0]=u1[center1];//set u0 to new values
			if (i==1){upper[j-1]=u0[center0];}//my upper row
			if (i==n){lower[j-1]=u0[center0];}//my lower row
		}
	}
	iter+=1;
	MPI_Gather(&flag,1,MPI_INT,flagr,1,MPI_INT,0,MPI_COMM_WORLD);//Gather flag corresponding to errors from all processor
	if(my_rank==0){for (i=0;i<p;i++){if (flagr[i]==1){flag=1;}}}//set flag=1 if any of the flag is 1
	MPI_Bcast(&flag,1,MPI_INT,0,MPI_COMM_WORLD);//Broadcast flag 
	if (my_rank!=p-1){MPI_Send(lower,N, MPI_DOUBLE,my_rank+1, 0, MPI_COMM_WORLD);}//sending my lower row(nth) to to (my_rank+1)th processor's 0th row
	if (my_rank!=0)  {MPI_Send(upper,N, MPI_DOUBLE,my_rank-1, 0, MPI_COMM_WORLD);}//sending my upper row(0th) to (my_rank-1)th processor's (n+1)th row
	if (my_rank!=p-1){MPI_Recv(lower,N, MPI_DOUBLE,my_rank+1, 0,MPI_COMM_WORLD, &status);}//recieving into n+1  row from my_rank+1 processor
	if (my_rank!=0)  {MPI_Recv(upper,N, MPI_DOUBLE,my_rank-1, 0,MPI_COMM_WORLD, &status);}//recieving into 0the  row from my_rank-1 processor
	if (my_rank!=0)  {for(k=1;k<N+1;k++){u0[k]=upper[k-1];}}//update upper row of my_matrix with lower row of my_rank-1
	if (my_rank!=p-1){for(k=1;k<N+1;k++){u0[(n+1)*(N+2)+k]=lower[k-1];}}//update lower row of my_matrix with upper row of my_rank+1
}
free(lower); 
free(upper);
free(flagr);
//printf("total iteration:%d\n",iter);
}
/**********************Write to File*************************************/
void write2file(FILE* fp,double* out,int count)
{
	int i=0;
	for (i=0;i<count;i++)
	{
	fprintf(fp,"%f\n",out[i]);
	}
}
