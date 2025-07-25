#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <time.h>

///////////////////////////////////////////////////////////////////////
//
//   DEBUGGING CODE ENDS HERE. CDFINVS CODE PROPER STARTS BELOW.
//
///////////////////////////////////////////////////////////////////////



void cdfinvs_all(ys,ccs,n,m,k,ivals)
int m, n, k, ivals[][m];
double *ys, ccs[][n+1];
{

    int i,j,ijk,iii;

        for (j=0; j<k; j=j+1){
            cdfinvs_march(ys,&ccs[j][0],n,m,&ivals[j][0]);
        }

        prinf("m=",&m,1);
        prinf("n=",&n,1);

        prin2("ys=",ys,m);
        prin2("ccs, 0=",&ccs[0][0],n+1);
        prin2("ccs, 1=",&ccs[1][0],n+1);

        return;

}



void cdfinvs_march(ys,ccs,n,m,ivals)
double *ys, *ccs;
int m, n, *ivals;
{

    int i,j,ival,istar;
    double vals_grid[n+1];

        prin2("ys, in march=",ys,m);
        prin2("ccs, in march=",ccs,n+1);

        istar=0;

        for (j=0; j<m; j=j+1) {

            vals_grid[istar] = ccs[istar] - ys[j];
            for (i=istar+1; i<n+1; i=i+1) {

                prinf("i=",&i,1);
                vals_grid[i] = ccs[i] - ys[j];
            #
                if (vals_grid[i]*vals_grid[i-1] <= 0) {
                    ivals[j] = i-1;
                    istar = ivals[j];
                    prinf("exiting loop at i=",&i,1);
                    break;
                }
            }



            prinf("j=",&j,1);
            prinf("ivals[j]=",&ivals[j],1);
        }



        return;
}



int cdfinv_march(yy,ccs,n)
double yy, *ccs;
int n;
{
    int i,j,k,l,ival;
    double vals_grid[n+1];

        vals_grid[0] = ccs[0] - yy;
        for (i=1; i<n+1; i=i+1) {
            vals_grid[i] = ccs[i] - yy;
        #
            if (vals_grid[i]*vals_grid[i-1] <= 0) {
                ival = i-1;
                prinf("exiting loop at i=",&i,1);
                break;
            }
        }

        prinf("n=",&n,1);

        prin2("ccs=",ccs,n+1);
        prin2("yy=",&yy,1);
        prinf("ival=",&ival,1);


        prin2("vals_grid=",&vals_grid[ival],1);
        prin2("vals_grid=",&vals_grid[ival+1],1);


        prin2("ccs=",&ccs[ival],1);
        prin2("ccs=",&ccs[ival+1],1);

        return ival;
}



