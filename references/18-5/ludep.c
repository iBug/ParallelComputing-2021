#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#define a(x,y) a[x*M+y]
/*AΪM*M����*/
#define A(x,y) A[x*M+y]
#define l(x,y) l[x*M+y]
#define u(x,y) u[x*M+y]
#define floatsize sizeof(float)
#define intsize sizeof(int)

int M,N;
int m;
float *A;
int my_rank;
int p;
MPI_Status status;

void fatal(char *message)
{
    printf("%s\n",message);
    exit(1);
}

void Environment_Finalize(float *a,float *f)
{
    free(a);
    free(f);
}

int main(int argc, char **argv)
{
    int i,j,k,my_rank,group_size;
    int i1,i2;
    int v,w;
    float *a,*f,*l,*u;
    FILE *fdA;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    p=group_size;

    if (my_rank==0)
    {
        fdA=fopen("dataIn.txt","r");
        fscanf(fdA,"%d %d", &M, &N);
        if(M != N)
        {
            puts("The input is error!");
            exit(0);
        }
        A=(float *)malloc(floatsize*M*M);
        for(i = 0; i < M; i ++)
            for(j = 0; j < M; j ++)
                fscanf(fdA, "%f", A+i*M+j);
        fclose(fdA);
    }

    /*0�Ž��̽�M�㲥�����н���*/
    MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);
    m=M/p;
    if (M%p!=0) m++;

    /*�����������̵��Ӿ����СΪm*M*/
    a=(float*)malloc(floatsize*m*M);

    /*������Ϊ����Ԫ�ؽ������ͺͽ��ջ�����*/
    f=(float*)malloc(floatsize*M);

    /*0�Ž���Ϊl��u��������ڴ棬�Է���������任���A�����е�l��u����*/
    if (my_rank==0)
    {
        l=(float*)malloc(floatsize*M*M);
        u=(float*)malloc(floatsize*M*M);
    }

    /*0�Ž��̲����н��滮�ֽ�����A����Ϊ��Сm*M��p���Ӿ������η��͸�1��p-1�Ž���*/
    if (a==NULL) fatal("allocate error\n");

    if (my_rank==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<M;j++)
                a(i,j)=A((i*p),j);
        for(i=0;i<M;i++)
            if ((i%p)!=0)
        {
            i1=i%p;
            i2=i/p+1;
            MPI_Send(&A(i,0),M,MPI_FLOAT,i1,i2,MPI_COMM_WORLD);
        }
    }
    else
    {
        for(i=0;i<m;i++)
            MPI_Recv(&a(i,0),M,MPI_FLOAT,0,i+1,MPI_COMM_WORLD,&status);
    }

    for(i=0;i<m;i++)
        for(j=0;j<p;j++)
    {
        /*j�Ž��̸���㲥����Ԫ��*/
        if (my_rank==j)
        {
            v=i*p+j;
            for (k=v;k<M;k++)
                f[k]=a(i,k);

            MPI_Bcast(f,M,MPI_FLOAT,my_rank,MPI_COMM_WORLD);
        }
        else
        {
            v=i*p+j;
            MPI_Bcast(f,M,MPI_FLOAT,j,MPI_COMM_WORLD);
        }

        /*���С��my_rank�Ľ��̣�����my_rank�����������ж����i+1,��,m-1���������б任*/
        if (my_rank<=j)
            for(k=i+1;k<m;k++)
        {
            a(k,v)=a(k,v)/f[v];
            for(w=v+1;w<M;w++)
                a(k,w)=a(k,w)-f[w]*a(k,v);
        }

        /*��Ŵ���my_rank�Ľ����������ж����i,��,m-1���������б任*/
        if (my_rank>j)
            for(k=i;k<m;k++)
        {
            a(k,v)=a(k,v)/f[v];
            for(w=v+1;w<M;w++)
                a(k,w)=a(k,w)-f[w]*a(k,v);
        }
    }

    /*0�Ž��̴�����������н����Ӿ���a���õ������任�ľ���A*/
    if (my_rank==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<M;j++)
                A(i*p,j)=a(i,j);
    }
    if (my_rank!=0)
    {
        for(i=0;i<m;i++)
            MPI_Send(&a(i,0),M,MPI_FLOAT,0,i,MPI_COMM_WORLD);
    }
    else
    {
        for(i=1;i<p;i++)
            for(j=0;j<m;j++)
        {
            MPI_Recv(&a(j,0),M,MPI_FLOAT,i,j,MPI_COMM_WORLD,&status);
            for(k=0;k<M;k++)
                A((j*p+i),k)=a(j,k);
        }
    }

    if (my_rank==0)
    {
        for(i=0;i<M;i++)
            for(j=0;j<M;j++)
                u(i,j)=0.0;
        for(i=0;i<M;i++)
            for(j=0;j<M;j++)
                if (i==j)
                    l(i,j)=1.0;
        else
            l(i,j)=0.0;
        for(i=0;i<M;i++)
            for(j=0;j<M;j++)
                if (i>j)
                    l(i,j)=A(i,j);
        else
            u(i,j)=A(i,j);
        printf("Input of file \"dataIn.txt\"\n");
        printf("%d\t %d\n",M, N);
        for(i=0;i<M;i++)
        {
            for(j=0;j<N;j++)
                printf("%f\t",A(i,j));
            printf("\n");
        }
        printf("\nOutput of LU operation\n");
        printf("Matrix L:\n");
        for(i=0;i<M;i++)
        {
            for(j=0;j<M;j++)
                printf("%f\t",l(i,j));
            printf("\n");
        }
        printf("Matrix U:\n");
        for(i=0;i<M;i++)
        {
            for(j=0;j<M;j++)
                printf("%f\t",u(i,j));
            printf("\n");
        }
    }
    MPI_Finalize();
    Environment_Finalize(a,f);
    return(0);
}
