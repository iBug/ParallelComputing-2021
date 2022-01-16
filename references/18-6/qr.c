#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "mpi.h"

#define a(x,y) a[x*M+y]
#define q(x,y) q[x*M+y]
#define A(x,y) A[x*M+y]
#define Q(x,y) Q[x*M+y]
#define R(x,y) R[x*M+y]

float temp;
float *A;
float *R;
float *Q;
double starttime;
double time1;
double time2;
int p;
MPI_Status status;

void Environment_Finalize(float *a,float *q,float *v,float *f,float *R,
                          float *Q,float *ai,float *aj,float *qi,float *qj)
{
    free(a);
    free(q);
    free(v);
    free(f);
    free(R);
    free(Q);
    free(ai);
    free(aj);
    free(qi);
    free(qj);
}


int main(int argc, char **argv)
{
    int M,N,m;
    int z;
    int i,j,k,my_rank,group_size;
    float *ai,*qi,*aj,*qj;
    float c,s,sp;
    float *f,*v;
    float *a,*q;
    FILE *fdA;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    p=group_size;

    starttime=MPI_Wtime();

    if(my_rank==p-1)
    {
        fdA=fopen("dataIn.txt","r");
        fscanf(fdA,"%d %d", &M, &N);

        if(M != N)
        {
            puts("The input is error!");
            exit(0);
        }

        A=(float*)malloc(sizeof(float)*M*M);
        Q=(float*)malloc(sizeof(float)*M*M);
        R=(float*)malloc(sizeof(float)*M*M);

        for(i = 0; i < M; i ++)
        {
            for(j = 0; j < M; j ++) fscanf(fdA, "%f", A+i*M+j);
        }
        fclose(fdA);

        for(i=0;i<M;i++)
            for(j=0;j<M;j++)
                if (i==j)
                    Q(i,j)=1.0;
        else
            Q(i,j)=0.0;
    }

    MPI_Bcast(&M,1,MPI_INT,p-1,MPI_COMM_WORLD);
    m=M/p;
    if (M%p!=0) m++;

    qi=(float*)malloc(sizeof(float)*M);
    qj=(float*)malloc(sizeof(float)*M);
    aj=(float*)malloc(sizeof(float)*M);
    ai=(float*)malloc(sizeof(float)*M);
    v=(float*)malloc(sizeof(float)*M);
    f=(float*)malloc(sizeof(float)*M);
    a=(float*)malloc(sizeof(float)*m*M);
    q=(float*)malloc(sizeof(float)*m*M);

    if (a==NULL||q==NULL||f==NULL||v==NULL||qi==NULL||qj==NULL||ai==NULL||aj==NULL)
        printf("memory allocation is wrong\n");

    if (my_rank==p-1)
    {
        for(i=0;i<m;i++)
            for(j=0;j<M;j++)
        {
            a(i,j)=A((my_rank*m+i),j);
            q(i,j)=Q((my_rank*m+i),j);
        }
    }

    if (my_rank==p-1)
    {
        for(i=0;i<p-1;i++)
        {
            MPI_Send(&A(m*i,0),m*M,MPI_FLOAT,i,i,MPI_COMM_WORLD);
            MPI_Send(&Q(m*i,0),m*M,MPI_FLOAT,i,i,MPI_COMM_WORLD);
        }
        free(A);
    }
    else
    {
        MPI_Recv(a,m*M,MPI_FLOAT,p-1,my_rank,MPI_COMM_WORLD,&status);
        MPI_Recv(q,m*M,MPI_FLOAT,p-1,my_rank,MPI_COMM_WORLD,&status);
    }

    time1=MPI_Wtime();

    if (p>1)
    {
        if (my_rank==0)
        {
            for(j=0;j<m-1;j++)
            {
                for(i=j+1;i<m;i++)
                {
                    sp=sqrt(a(j,j)*a(j,j)+a(i,j)*a(i,j));
                    c=a(j,j)/sp;  s=a(i,j)/sp;

                    for(k=0;k<M;k++)
                    {
                        aj[k]=c*a(j,k)+s*a(i,k);
                        qj[k]=c*q(j,k)+s*q(i,k);
                        ai[k]=-s*a(j,k)+c*a(i,k);
                        qi[k]=-s*q(j,k)+c*q(i,k);
                    }

                    for(k=0;k<M;k++)
                    {
                        a(j,k)=aj[k];
                        q(j,k)=qj[k];
                        a(i,k)=ai[k];
                        q(i,k)=qi[k];
                    }
                }                                 /*  i */

                for(k=0;k<M;k++)
                {
                    f[k]=a(j,k);
                    v[k]=q(j,k);
                }

                MPI_Send(&f[0],M,MPI_FLOAT,1,j,MPI_COMM_WORLD);
                MPI_Send(&v[0],M,MPI_FLOAT,1,j,MPI_COMM_WORLD);
            }                                     /* for j */
            for(k=0;k<M;k++)
            {
                f[k]=a((m-1),k);
                v[k]=q((m-1),k);
            }

            MPI_Send(&f[0],M,MPI_FLOAT,1,m-1,MPI_COMM_WORLD);
            MPI_Send(&v[0],M,MPI_FLOAT,1,m-1,MPI_COMM_WORLD);
        }                                         /* my_rank==0 */

        else                                      /* my_rank!=0 */
        {
            if (my_rank!=(group_size-1))
            {
                for(j=0;j<my_rank*m;j++)
                {
                    MPI_Recv(&f[0],M,MPI_FLOAT,(my_rank-1),j,MPI_COMM_WORLD,&status);
                    MPI_Recv(&v[0],M,MPI_FLOAT,(my_rank-1),j,MPI_COMM_WORLD,&status);

                    for(i=0;i<m;i++)
                    {
                        sp=sqrt(f[j]*f[j]+a(i,j)*a(i,j));
                        c=f[j]/sp;  s=a(i,j)/sp;

                        for(k=0;k<M;k++)
                        {
                            aj[k]=c*f[k]+s*a(i,k);
                            qj[k]=c*v[k]+s*q(i,k);
                            ai[k]=-s*f[k]+c*a(i,k);
                            qi[k]=-s*v[k]+c*q(i,k);
                        }

                        for(k=0;k<M;k++)
                        {
                            f[k]=aj[k];
                            v[k]=qj[k];
                            a(i,k)=ai[k];
                            q(i,k)=qi[k];
                        }
                    }

                    MPI_Send(&f[0],M,MPI_FLOAT,(my_rank+1),j,MPI_COMM_WORLD);
                    MPI_Send(&v[0],M,MPI_FLOAT,(my_rank+1),j,MPI_COMM_WORLD);
                }                                 /* for j */

                for(j=0;j<m-1;j++)
                {
                    for(i=j+1;i<m;i++)
                    {
                        sp=sqrt(a(j,(my_rank*m+j))*a(j,(my_rank*m+j))+a(i,(my_rank*m+j))*a(i,(my_rank*m+j)));
                        c=a(j,(my_rank*m+j))/sp;
                        s=a(i,(my_rank*m+j))/sp;

                        for(k=0;k<M;k++)
                        {
                            aj[k]=c*a(j,k)+s*a(i,k);
                            qj[k]=c*q(j,k)+s*q(i,k);
                            ai[k]=-s*a(j,k)+c*a(i,k);
                            qi[k]=-s*q(j,k)+c*q(i,k);
                        }

                        for(k=0;k<M;k++)
                        {
                            a(j,k)=aj[k];
                            q(j,k)=qj[k];
                            a(i,k)=ai[k];
                            q(i,k)=qi[k];
                        }
                    }

                    for(k=0;k<M;k++)
                    {
                        f[k]=a(j,k);
                        v[k]=q(j,k);
                    }

                    MPI_Send(&f[0],M,MPI_FLOAT,my_rank+1,my_rank*m+j,MPI_COMM_WORLD);
                    MPI_Send(&v[0],M,MPI_FLOAT,my_rank+1,my_rank*m+j,MPI_COMM_WORLD);
                }                                 /* for j */

                for(k=0;k<M;k++)
                {
                    f[k]=a((m-1),k);
                    v[k]=q((m-1),k);
                }

                MPI_Send(&f[0],M,MPI_FLOAT,my_rank+1,my_rank*m+m-1,MPI_COMM_WORLD);
                MPI_Send(&v[0],M,MPI_FLOAT,my_rank+1,my_rank*m+m-1,MPI_COMM_WORLD);

            }                                     /* my_rank !=groupsize -1 */

            if (my_rank==(group_size-1))
            {
                for(j=0;j<my_rank*m;j++)
                {
                    MPI_Recv(&f[0],M,MPI_FLOAT,(my_rank-1),j,MPI_COMM_WORLD,&status);
                    MPI_Recv(&v[0],M,MPI_FLOAT,(my_rank-1),j,MPI_COMM_WORLD,&status);

                    for(i=0;i<m;i++)
                    {
                        sp=sqrt(f[j]*f[j]+a(i,j)*a(i,j));
                        c=f[j]/sp;  s=a(i,j)/sp;

                        for(k=0;k<M;k++)
                        {
                            aj[k]=c*f[k]+s*a(i,k);
                            qj[k]=c*v[k]+s*q(i,k);
                            ai[k]=-s*f[k]+c*a(i,k);
                            qi[k]=-s*v[k]+c*q(i,k);
                        }

                        for(k=0;k<M;k++)
                        {
                            f[k]=aj[k];
                            v[k]=qj[k];
                            a(i,k)=ai[k];
                            q(i,k)=qi[k];
                        }
                    }                             /* for i */

                    for(k=0;k<M;k++)
                    {
                        Q(j,k)=v[k];
                        R(j,k)=f[k];
                    }

                }                                 /* for j */

                for(j=0;j<m-1;j++)
                {
                    for(i=j+1;i<m;i++)
                    {
                        sp=sqrt(a(j,(my_rank*m+j))*a(j,(my_rank*m+j))+a(i,(my_rank*m+j))*a(i,(my_rank*m+j)));
                        c=a(j,(my_rank*m+j))/sp;
                        s=a(i,(my_rank*m+j))/sp;

                        for(k=0;k<M;k++)
                        {
                            aj[k]=c*a(j,k)+s*a(i,k);
                            qj[k]=c*q(j,k)+s*q(i,k);
                            ai[k]=-s*a(j,k)+c*a(i,k);
                            qi[k]=-s*q(j,k)+c*q(i,k);
                        }

                        for(k=0;k<M;k++)
                        {
                            a(j,k)=aj[k];
                            q(j,k)=qj[k];
                            a(i,k)=ai[k];
                            q(i,k)=qi[k];
                        }
                    }                             /*  for i */

                    for(k=0;k<M;k++)
                    {
                        Q((my_rank*m+j),k)=q(j,k);
                        R((my_rank*m+j),k)=a(j,k);
                    }
                }                                 /* for j */

                for(k=0;k<M;k++)
                {
                    Q((my_rank*m+m-1),k)=q((m-1),k);
                    R((my_rank*m+m-1),k)=a((m-1),k);
                }

            }                                     /* for my_rank==groupsize -1 */

        }                                         /*    else my_rank!=0           */
    }                                             /*    if p >1          */

    if (p==1)
    {
        for (j=0;j<M;j++)
            for (i=j+1;i<M;i++)
        {
            sp=sqrt(a(j,j)*a(j,j) + a(i,j)*a(i,j));
            c=a(j,j)/sp;
            s=a(i,j)/sp;

            for (k=0;k<M;k++)
            {
                aj[k]=c*a(j,k) + s*a(i,k);
                qj[k]=c*q(j,k) + s*q(i,k);
                ai[k]=(-s)*a(j,k) + c*a(i,k);
                qi[k]=(-s)*q(j,k) + c*q(i,k);
            }

            for (k=0;k<M;k++)
            {
                a(j,k)=aj[k];
                q(j,k)=qj[k];
                a(i,k)=ai[k];
                q(i,k)=qi[k];
            }
        }                                         /* for   */

        for(i=0;i<M;i++)
            for(j=0;j<M;j++)
                R(i,j)=a(i,j);

        for(i=0;i<M;i++)
            for(j=0;j<M;j++)
                Q(i,j)=q(i,j);
    }                                             /*  if   p==1 */

    if (my_rank==p-1)
    {
        printf("Input of file \"dataIn.txt\"\n");
        printf("%d\t %d\n",M, N);
        for(i=0;i<M;i++)
        {
            for(j=0;j<N;j++) printf("%f\t",A(i,j));
            printf("\n");
        }

        printf("\nOutput of QR operation\n");

        printf("Matrix R:\n");
        for(i=0;i<M;i++)
        {
            for(j=0;j<M;j++)
                printf("%f\t",R(i,j));
            printf("\n");
        }

        for(i=0;i<M;i++)
            for(j=i+1;j<M;j++)
        {
            temp=Q(i,j);
            Q(i,j)=Q(j,i);
            Q(j,i)=temp;
        }

        printf("Matrix Q:\n");
        for(i=0;i<M;i++)
        {
            for(j=0;j<M;j++)
                printf("%f\t",Q(i,j));
            printf("\n");
        }
    }

    time2 = MPI_Wtime();
    if (my_rank==0)
    {
        printf("\n");
        printf("Whole running time    = %f seconds\n",time2-starttime);
        printf("Distribute data time  = %f seconds\n",time1-starttime);
        printf("Parallel compute time = %f seconds\n",time2-time1);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    Environment_Finalize(a,q,v,f,R,Q,ai,aj,qi,qj);
    return(0);
}
