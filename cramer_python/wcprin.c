#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <time.h>

void prini(ifprint,ifinit)
int ifprint;
int ifinit;
{
    static FILE *st1;
    int a, ka, n, itype;
    static int ifsave;
    char *mes, *fname;

//        printf("ifprint=\n%d\n",ifprint);

        fname="out13";

        princore(fname,ifinit,mes,a,ka,n,itype,ifprint);

}



void princore(fname,ifinit,mes,a,ka,n,itype,ifprint)
FILE *fname;
char *mes;
double *a;
int n, *ka, itype, ifinit, ifprint;
{
    static FILE *st1;
    static int ifprint_save;
    int i;

//        printf("ifprint, in princore=\n%d\n",ifprint);


    if (ifprint == 1) ifprint_save = 1;
    if (ifprint == 0) ifprint_save = 0;


    if (ifprint_save == 0) return;


        if(ifinit==1) {
            ifprint=1;
            st1=fopen(fname,"w");
            return;
        }

/*
/        print double precision array
*/
    if (itype==11) {
/*
/        . . . print to standard output
*/
        fprintf(stdout,"%s\n",mes);
        for (i=0; i<n; i=i+1) {
            fprintf(stdout,"  %11.4le", a[i]);
            if(i%6==5 || i==n-1) fprintf(stdout,"\n");
        }

/*
/    . . . print to the file
*/
        fprintf(st1,"%s\n",mes);
        for (i=0; i<n; i=i+1) {
            fprintf(st1,"  %11.4le", a[i]);
            if(i%6==5 || i==n-1) fprintf(st1,"\n");
        }
        return;
    }


/*
/        print integer array
*/
    if (itype==21) {
/*
/        . . . print to standard output
*/
        fprintf(stdout,"%s\n",mes);
        for (i=0; i<n; i=i+1) {
            fprintf(stdout," %7d", ka[i]);
            if(i%10==9 || i==n-1) fprintf(stdout,"\n");
        }
/*
/        . . . print to the file
*/
        fprintf(st1,"%s\n",mes);
        for (i=0; i<n; i=i+1) {
            fprintf(st1," %7d", ka[i]);
            if(i%10==9 || i==n-1) fprintf(st1,"\n");
        }
    }

}



void prin2(mes,a,n)
double *a;
char *mes;
{

    int fname, ifinit, itype, ka, ifprint;

        ifinit=0;
        itype=11;
        ifprint=-100;
        princore(fname,ifinit,mes,a,ka,n,itype,ifprint);
}



void prinf(mes,ka,n)
char *mes;
int *ka;
{
    int i, itype, fname, ifinit, ifprint;
    double *a;


        itype=21;
        ifinit =0;
        ifprint=-100;
        princore(fname,ifinit,mes,a,ka,n,itype,ifprint);
}



