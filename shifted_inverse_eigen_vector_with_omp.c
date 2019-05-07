#include<stdio.h>
#include<math.h>
#include<omp.h>
#include<stdlib.h>
#include<time.h>
#define N 1000

double A[N+1][N+1],l[N+1][N+1],u[N+1][N+1],C[N+1][N+1];
long long int i,j,k;
void mylu(){
	for(k=1;k<=N-1;k++){
		long long row,col;
		#pragma omp parallel for shared(k) private(row,col) schedule(static)		
		for(row=k+1;row<=N;row++){
			double factor = A[row][k]/A[k][k];
			for(col=k+1;col<=N;col++){
				A[row][col]=A[row][col]-factor*A[k][col];
			
			}
			A[row][k]=factor;
		}
	}
}

void forward(double b[],double x[]){
	for(i=1;i<=N;i++){
		x[i]=b[i];
	long long row;
		for(row=1;row<=i-1;row++)
			x[i]=x[i]-l[i][row]*x[row];
		x[i]=x[i]/l[i][i];
	}
}

void backward(double b[],double x[]){
	for(i=N;i>=1;i--){
		x[i]=b[i];
		for(j=i+1;j<=N;j++)
			x[i]=x[i]-u[i][j]*x[j];
		x[i]=x[i]/u[i][i];
	}
}



int main()
{
	double d1=0.0,d2=0.0,eps=1e-6,lambdacurr=0.0,lambdabefore=0.0,s=-1,b[N+1],x[N+1];
	/*x[1] = 1;
	  x[2] = 0;
	  x[3] = 0;

	  A[1][1]=-2;
	  A[1][2]=-2;
	  A[1][3]=3;
	  A[2][1]=-10;
	  A[2][2]=-1;
	  A[2][3]=6;
	  A[3][1]=10;
	  A[3][2]=-2;
	  A[3][3]=-9;
	  */
	//srand(time(0));
	x[1]=1;
	for(i=2;i<=N;i++)
		x[i]=0;
	for(i=1;i<=N;i++)
		for(j=1;j<=N;j++)
			if(i==j)
				A[i][j]=1;
			else
				A[i][j]=0;
	//A[i][j]=rand()%1000;
	double st=omp_get_wtime();
	#pragma omp parallel for
	for(i=1;i<=N;i++)
	{
		A[i][i]-=s;
	}
	mylu();
	double st2 = omp_get_wtime();
	printf("%lf",st2-st);
	for(i=1;i<=N;i++)
	{
		#pragma omp parallel for
		for(j=1;j<=N;j++){
			if(i==j)
				l[i][j]=1;
			else if(i<j)
				l[i][j]=0;
			else
				l[i][j]=A[i][j];
		}
	}
	for(i=1;i<=N;i++)
	{
		#pragma omp parallel for
		for(j=1;j<=N;j++){
			if(i>j)
				u[i][j]=0;
			else
				u[i][j]=A[i][j];
		}
	}

	//normalization of x
	for(i=1; i<=N; i++){
		d1+=x[i]*x[i];
	}
	d1 = 1.0/sqrt(d1);
	for(i=1;i<=N;i++){
		x[i]=x[i]*d1;
	}

	double y[N+1],z[N+1];
	for(k=0;k<100000;k++){
		#pragma omp parallel for	
		for(i=1;i<=N;i++){
			y[i]=0;
			z[i]=0;
		}
		#pragma omp barrier
		forward(x,z);
		backward(z,y);
		d1=0.0;
		for(i=1;i<=N;i++){
			d1+=y[i]*y[i];
		}
		d1=1.0/sqrt(d1);
		for(i=1;i<=N;i++){
			y[i]=y[i]*d1;
		}
		lambdacurr = 0.0;
		#pragma parallel for
		for(i=1;i<=N;i++){
			lambdacurr=lambdacurr+((y[i]-x[i])*(y[i]-x[i]));
		}
		lambdacurr = sqrt(lambdacurr);
		if(lambdacurr<eps)
			break;
		#pragma omp parallel for
		for(i=1;i<=N;i++){
			x[i]=y[i];
		}
	}
	double en=omp_get_wtime();
	for(i=1;i<=N;i++)
		printf("%lf ",y[i]);
	printf("\n%lf",en-st);
	return 0;
}
