#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double Ds,Di,Dv;
double ds,di,dv;
double beta,alpha,p;
double T,L;
double dt,dx;
double theta;

#include "matrix.h"
solution *sol;
gsl_vector *lambdafunc(int n){
    gsl_vector *v=gsl_vector_alloc(n);
    int i;
    for(i=0;i<n;i++) gsl_vector_set(v,i,sin(i*dx/(2*M_PI)*L)+1.0);
    return v;
}

void setvalue(){
    Ds=0.01;
    Di=0.02;
    Dv=0.1;
    ds=0.1;
    di=0.2;
    dv=1;
    beta=0.2;
    alpha=0.3;
    p=10;
    T=200;
    L=2;
    dt=0.001;
    dx=0.1;
    theta=0.8;
}
void calculate(){
    
    FILE *fpv=fopen("vxt.csv","w+");
    FILE *fpS=fopen("sxt.csv","w+");
    FILE *fpI=fopen("Ixt.csv","w+");

    int i;
    int Tm=T/dt;
    int Lm=L/dx;
    gsl_matrix *diffS=diffusion(Ds/dx/dx,Lm);
    gsl_matrix *diffI=diffusion(Di/dx/dx,Lm);
    gsl_matrix *diffv=diffusion(Dv/dx/dx,Lm);
    gsl_vector *lam=lambdafunc(Lm);
    gsl_matrix *Im=eye(Lm);
    sol=malloc(sizeof(solution)*(Tm+1));

    sol[0].S=onesa(Lm,10.0);
    sol[0].I=gsl_vector_alloc(Lm);
    sol[0].v=gsl_vector_alloc(Lm);
    gsl_vector_set(sol[0].v,10,1.0);
    gsl_vector_set(sol[0].v,11,1.0);
    sol[0].t=0.0;
    gsl_matrix *L;
    gsl_vector *tmp;
    gsl_vector *t2;

    for (i=1;i<Tm;i++){
        //S
        L=MatMinus(diffS,MatPlus(infect(sol[i-1].S,sol[i-1].I,sol[i-1].v),diaga(ds,Lm),1,1),0,1);
        t2=productva(lam,dt,0);
        tmp=VecPlus(productmv(MatPlus(Im,productma(L,dt*(1-theta),0),0,1),sol[i-1].S,1,0),t2,1,1);
        sol[i].S=productmv(inverse(MatMinus(Im,productma(L,dt*theta,1),0,1),1),tmp,1,1);
        //I
        L=MatMinus(diffI,diaga(di,Lm),0,1);
        t2=infect2(sol[i].S,sol[i-1].I,sol[i-1].v,dt);
        tmp=VecPlus(productmv(MatPlus(Im,productma(L,dt*(1-theta),0),0,1),sol[i-1].I,1,0),t2,1,1);
        sol[i].I=productmv(inverse(MatMinus(Im,productma(L,theta*dt,1),0,1),1),tmp,1,1);
        //v
        L=MatMinus(MatPlus(diffv,mechtaxi(sol[i].S,sol[i].I,alpha/dx),0,1),diaga(dv,Lm),1,1);
        t2=virusbirth(sol[i].I);
        tmp=VecPlus(productmv(MatPlus(Im,productma(L,dt*(1-theta),0),0,1),sol[i-1].v,1,0),t2,1,1);
        sol[i].v=productmv(inverse(MatMinus(Im,productma(L,dt*theta,1),0,1),1),tmp,1,1);
        sol[i].t=sol[i-1].t+dt;
        int j;
        if(i%100==0){
            for (j=0;j<Lm-1;j++){
                fprintf(fpS,"%lf,",gsl_vector_get(sol[i].S,j));
                fprintf(fpI,"%lf,",gsl_vector_get(sol[i].I,j));
                fprintf(fpv,"%lf,",gsl_vector_get(sol[i].v,j));
            }
            fprintf(fpS,"%lf\n",gsl_vector_get(sol[i].S,j));
            fprintf(fpI,"%lf\n",gsl_vector_get(sol[i].I,j));
            fprintf(fpv,"%lf\n",gsl_vector_get(sol[i].v,j));
        }
        for(j=0;j<Lm;j++)
            if (gsl_vector_get(sol[i].S,j)<=0.0)
                gsl_vector_set(sol[i].S,j,1e-260);
    }

    for(i=0;i<Tm;i++){
        gsl_vector_free(sol[i].S);
        gsl_vector_free(sol[i].I);
        gsl_vector_free(sol[i].v);
    }
    free(sol);
    gsl_vector_free(lam);
    gsl_matrix_free(diffS);
    gsl_matrix_free(diffI);
    gsl_matrix_free(diffv);
    gsl_matrix_free(Im);
    fclose(fpv);
    fclose(fpS);
    fclose(fpI);
}
int main(){
    setvalue();
    calculate();
    return 0;
}
