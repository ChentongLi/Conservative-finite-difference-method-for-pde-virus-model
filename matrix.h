#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#define MAX(x,y)  (((x)>(y)) ?  (x) : (y))

typedef struct solution{
    gsl_vector *S;
    gsl_vector *I;
    gsl_vector *v;
    double t;
}solution;

gsl_matrix *inverse(gsl_matrix *m,int flag){
    
    int s;
    int n=m->size1;
    gsl_matrix *invm=gsl_matrix_alloc(n,n);

    gsl_permutation * p = gsl_permutation_alloc (n);
    gsl_linalg_LU_decomp (m, p, &s);
    gsl_linalg_LU_invert (m, p, invm);
    gsl_permutation_free(p);
    if(flag) gsl_matrix_free(m);
    
    return invm;
    
}
double betafunc(int n){
    double l=n*dx;
    return beta;
}
double pfunc(int n){
    double l=n*dx;
    return p;
}
int VecCopy(gsl_vector *v1,gsl_vector *v2){
    int n=v1->size;
    int i;
    for(i=0;i<n;i++){
        gsl_vector_set(v1,i,gsl_vector_get(v2,i));
    }
    gsl_vector_free(v2);
    return 1;
}
gsl_vector *onesa(int n,double a){
    gsl_vector *v=gsl_vector_alloc(n);
    int i;
    for(i=0;i<n;i++) gsl_vector_set(v,i,a);
    return v;
}
gsl_matrix *eye(int n){
    
    gsl_matrix *m=gsl_matrix_alloc(n,n);
    int i;
    for(i=0;i<n;i++) gsl_matrix_set(m,i,i,1.0);
    return m;
}
gsl_vector *productva(gsl_vector *v,double a,int flag){
    int n=v->size;
    gsl_vector *m=gsl_vector_alloc(n);
    int i;
    for(i=0;i<n;i++) gsl_vector_set(m,i,gsl_vector_get(v,i)*a);
    if(flag) gsl_vector_free(v);
    return m;
}
gsl_vector *virusbirth(gsl_vector *v){
    int n=v->size;
    gsl_vector *m=gsl_vector_alloc(n);
    int i;
    for(i=0;i<n;i++) gsl_vector_set(m,i,gsl_vector_get(v,i)*pfunc(i)*dt*di);
    return m;
}
gsl_matrix *MatPlus(gsl_matrix *m1,gsl_matrix *m2,int f1,int f2){
    
    int i,j;
    int n=m1->size1;
    gsl_matrix *result=gsl_matrix_alloc(n,n);
    for(i=0;i<n;i++)
        for(j=0;j<n;j++){
            gsl_matrix_set(result,i,j,gsl_matrix_get(m1,i,j)+gsl_matrix_get(m2,i,j));
    }
    if(f1) gsl_matrix_free(m1);
    if(f2) gsl_matrix_free(m2);
    return result;
}
double infectfunc(double S,double I,double v,double b){
    return b*v/(S+I);
}
gsl_matrix *infect(gsl_vector *S,gsl_vector *I,gsl_vector *v){
    int n=I->size;
    gsl_matrix *m=gsl_matrix_alloc(n,n);
    int i;
    for(i=0;i<n;i++) gsl_matrix_set(m,i,i,infectfunc(gsl_vector_get(S,i),gsl_vector_get(I,i),
                                                     gsl_vector_get(v,i),betafunc(i)));
    return m;
}
double infectfunc2(double S,double I,double v,double b){
    return b*v*S/(S+I);
}
gsl_vector *infect2(gsl_vector *S,gsl_vector *I,gsl_vector *v,double h){
    int n=I->size;
    gsl_vector *m=gsl_vector_alloc(n);
    int i;
    for(i=0;i<n;i++) gsl_vector_set(m,i,infectfunc2(gsl_vector_get(S,i),gsl_vector_get(I,i),
                                                    gsl_vector_get(v,i),betafunc(i))*h);
    return m;
}
gsl_matrix *diaga(double a,int n){
    gsl_matrix *m=gsl_matrix_alloc(n,n);
    int i;
    for(i=0;i<n;i++) gsl_matrix_set(m,i,i,a);
    return m;
}
gsl_matrix *MatMinus(gsl_matrix *m1,gsl_matrix *m2,int f1,int f2){
    
    int i,j;
    int n=m1->size1;
    gsl_matrix *result=gsl_matrix_alloc(n,n);
    for(i=0;i<n;i++)
        for(j=0;j<n;j++){
            gsl_matrix_set(result,i,j,gsl_matrix_get(m1,i,j)-gsl_matrix_get(m2,i,j));
        }
    if(f1) gsl_matrix_free(m1);
    if(f2) gsl_matrix_free(m2);
    return result;
}
gsl_vector *VecPlus(gsl_vector *v1, gsl_vector *v2,int f1,int f2){
    int n=v1->size;
    int i;
    gsl_vector *v=gsl_vector_alloc(n);
    for(i=0;i<n;i++) gsl_vector_set(v,i,gsl_vector_get(v1,i)+gsl_vector_get(v2,i));
    if(f1) gsl_vector_free(v1);
    if(f2) gsl_vector_free(v2);
    return v;
}
gsl_vector *productmv(gsl_matrix *m,gsl_vector *v,int f1,int f2){
    int n=v->size;
    gsl_vector *vr=gsl_vector_alloc(n);
    gsl_blas_dgemv (CblasNoTrans,
                    1.0, m, v,
                    0.0, vr);
    if(f1) gsl_matrix_free(m);
    if(f2) gsl_vector_free(v);
    return vr;
}
gsl_matrix *productmm(gsl_matrix *m1,gsl_matrix *m2,int f1,int f2){
    int n=m1->size1;
    gsl_matrix *m=gsl_matrix_alloc(n,n);
    gsl_blas_dgemm (CblasNoTrans,CblasNoTrans,
                    1.0, m1, m2,
                    0.0, m);
    if(f1) gsl_matrix_free(m1);
    if(f2) gsl_matrix_free(m2);
    return m;
}
gsl_matrix *productma(gsl_matrix *m,double a,int flag){
    
    int i,j;
    int n=m->size1;
    gsl_matrix *m0=gsl_matrix_alloc(n,n);
    for(i=0;i<n;i++)
        for(j=0;j<n;j++){
            gsl_matrix_set(m0,i,j,a*gsl_matrix_get(m,i,j));
        }
    if(flag) gsl_matrix_free(m);
    return m0;
}

gsl_matrix *diffusion(double D,int n){
    
    gsl_matrix *m=gsl_matrix_alloc(n,n);
    int i,j;
    for(i=0;i<n;i++)
        for(j=0;j<n;j++){
            if (i==j) gsl_matrix_set(m,i,j,-2*D);
            else if(j==(i+1) || j==(i-1)) gsl_matrix_set(m,i,j,D);
            else gsl_matrix_set(m,i,j,0.0);
        }
    gsl_matrix_set(m,0,n-1,D);  //boundary
    gsl_matrix_set(m,n-1,0,D);
    return m;
}
gsl_matrix *mechtaxi(gsl_vector *S,gsl_vector *I,double a){
    
    int n=S->size;
    gsl_matrix *m=gsl_matrix_alloc(n,n);
    gsl_vector *bp=gsl_vector_alloc(n);
    gsl_vector *bm=gsl_vector_alloc(n);
    int i,j;

    for(i=0;i<n;i++){
        double tmp;
        if(i==0)tmp=((gsl_vector_get(S,i)+gsl_vector_get(I,i))-(gsl_vector_get(S,n-1)+gsl_vector_get(I,n-1)))/dx;
        else tmp=((gsl_vector_get(S,i)+gsl_vector_get(I,i))-(gsl_vector_get(S,i-1)+gsl_vector_get(I,i-1)))/dx;
        gsl_vector_set(bp,i,MAX(tmp,0.0));
        gsl_vector_set(bm,i,MAX(-tmp,0.0));
    }
    for (i=0;i<n;i++)
        for (j=0;j<n;j++){
            if(i==j) gsl_matrix_set(m,i,j,-(gsl_vector_get(bp,i)+gsl_vector_get(bm,j))*a);
            else if(i==(j+1)) gsl_matrix_set(m,i,j,gsl_vector_get(bp,j)*a);
            else if(i==(j-1)) gsl_matrix_set(m,i,j,gsl_vector_get(bm,j)*a);
            else gsl_matrix_set(m,i,j,0.0);
        }
    gsl_matrix_set(m,0,n-1,gsl_vector_get(bp,n-1)*a); //boundary
    gsl_matrix_set(m,n-1,0,gsl_vector_get(bm,0)*a);
    gsl_vector_free(bp);
    gsl_vector_free(bm);
    return m;
}

double maxeigen(gsl_matrix *m){
    
    int n=m->size1;
    gsl_matrix *evecm = gsl_matrix_alloc(n,n);
    gsl_vector *eval = gsl_vector_alloc (n);
    gsl_eigen_symmv_workspace * w =gsl_eigen_symmv_alloc (n);
    gsl_eigen_symmv (m, eval, evecm, w);
    int i;
    double max=-1000;
    double tmp;
    for(i=0;i<n;i++){
        tmp=gsl_vector_get(eval,i);
        if (tmp>max) max=tmp;
    }
    gsl_eigen_symmv_free (w);
    gsl_matrix_free (evecm);
    gsl_vector_free (eval);
    return max;
}

