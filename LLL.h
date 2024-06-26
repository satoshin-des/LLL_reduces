#ifndef __LLL_C__
#define __LLL_C__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* 内積 */
double dot_dbl_dbl(double *x, double *y, const int n){
    double s = 0.0;
    for(int i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}
double dot_int_dbl(int *x, double *y, const int n){
    double s = 0.0;
    for(int i = 0; i < n; ++i) s += y[i] * x[i];
    return s;
}
double dot_int_int(int *x, int *y, const int n){
    double s = 0.0;
    for(int i = 0; i < n; ++i) s += y[i] * x[i];
    return s;
}


/* Gram-Schmidtの直交化 */
void GSO(int **b, double *B, double **mu, const int n, const int m){
    int i, j, k;
    double t, s, **GSOb;
    GSOb = (double **)malloc(n * sizeof(double *));
    for(i = 0; i < n; ++i) GSOb[i] = (double *)malloc(m * sizeof(double));

    for(i = 0; i < n; ++i){
        mu[i][i] = 1.0;
        for(j = 0; j < m; ++j) GSOb[i][j] = b[i][j];
        for(j = 0; j < i; ++j){
            mu[i][j] = dot_int_dbl(b[i], GSOb[j], m) / dot_dbl_dbl(GSOb[j], GSOb[j], m);
            for(k = 0; k < m; ++k) GSOb[i][k] -= mu[i][j] * GSOb[j][k];
        }
        B[i] = dot_dbl_dbl(GSOb[i], GSOb[i], m);
    }
}


/* 部分サイズ基底簡約 */
void SizeReduce(int **b, double **mu, const int i, const int j, const int m){
    int k;
    if(mu[i][j] > 0.5 || mu[i][j] < -0.5){
        const int q = round(mu[i][j]);
        for(k = 0; k < m; ++k) b[i][k] -= q * b[j][k];
        for(k = 0; k <= j; ++k) mu[i][k] -= mu[j][k] * q;
    }
}

/* LLL基底簡約 */
void LLLReduce(int **b, const double d, const int n, const int m){
    int j, i, h;
    double **mu, *B, nu, BB, C, t;
    mu = (double **)malloc(n * sizeof(double *));
    B = (double *)malloc(n * sizeof(double));
    for(i = 0; i < n; ++i) mu[i] = (double *)malloc(n * sizeof(double));
    GSO(b, B, mu, n, m);

    int tmp;
    for(int k = 1; k < n;){
        h = k - 1;
        for(j = h; j > -1; --j) SizeReduce(b, mu, k, j, m);

        if(k > 0 && B[k] < (d - mu[k][h] * mu[k][h]) * B[h]){
            for(i = 0; i < m; ++i){tmp = b[h][i]; b[h][i] = b[k][i]; b[k][i] = tmp;}
            
            nu = mu[k][k - 1]; BB = B[k] + nu * nu * B[k - 1]; C = 1.0 / BB;
            mu[k][k - 1] = nu * B[k - 1] * C; B[k] *= B[k - 1] * C; B[k - 1] = BB;

            for(i = 0; i <= k - 2; ++i){
                t = mu[k - 1][i]; mu[k - 1][i] = mu[k][i]; mu[k][i] = t;
            }
            for(i = k + 1; i < n; ++i){
                t = mu[i][k]; mu[i][k] = mu[i][k - 1] - nu * t;
                mu[i][k - 1] = t + mu[k][k - 1] * mu[i][k];
            }
            
            k = h;
        }else ++k;
    }
}
#endif