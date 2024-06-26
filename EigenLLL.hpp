#ifndef __EigenLLL__
#define __EigenLLL__

#include <iostream>
#include <eigen3/Eigen/Dense>

/* Gram-Schmidtの直交化 */
void GSO(const Eigen::MatrixXi b, Eigen::VectorXd& B, Eigen::MatrixXd& mu, const int n, const int m){
    int j;
    Eigen::MatrixXd GSOb(n, m);

    for(int i = 0; i < n; ++i){
        mu.coeffRef(i, i) = 1.0;
        GSOb.row(i) = b.row(i).cast<double>();
        for(j = 0; j < i; ++j){
            mu.coeffRef(i, j) = b.row(i).cast<double>().dot(GSOb.row(j)) / GSOb.row(j).dot(GSOb.row(j));
            GSOb.row(i) -= mu.coeff(i, j) * GSOb.row(j);
        }
        B.coeffRef(i) = GSOb.row(i).dot(GSOb.row(i));
    }
}

/* サイズ基底簡約 */
void SizeReduce(Eigen::MatrixXi& b, Eigen::MatrixXd& mu, const int i, const int j, const int m){
    if(mu.coeff(i, j) > 0.5 || mu.coeff(i, j) < -0.5){
        const int q = round(mu.coeff(i, j));
        b.row(i) -= q * b.row(j);
        mu.row(i).head(j + 1) -= (double)q * mu.row(j).head(j + 1);
    }
}

/* LLL簡約 */
void LLLReduce(Eigen::MatrixXi& b, const long double d, const int n, const int m){
    double nu, BB, C, t;
    Eigen::VectorXd B(n), logB(n);
    Eigen::MatrixXd mu(n, n);
    GSO(b, B, mu, n, m);
    
    for(int k = 1, j, i, k1; k < n;){
        k1 = k - 1;
        for(j = k1; j > -1; --j) SizeReduce(b, mu, k, j, m);

        if(k > 0 && B.coeff(k) < (d - mu.coeff(k, k1) * mu.coeff(k, k1)) * B.coeff(k1)){
            b.row(k).swap(b.row(k1));
            
            nu = mu.coeff(k, k1); BB = B.coeff(k) + nu * nu * B.coeff(k1); C = 1.0 / BB;
            mu.coeffRef(k, k1) = nu * B.coeff(k1) * C;
            B.coeffRef(k) *= B.coeff(k1) * C; B.coeffRef(k1) = BB;

            mu.row(k1).head(k - 1).swap(mu.row(k).head(k - 1));
            for(i = k + 1; i < n; ++i){
                t = mu.coeff(i, k); mu.coeffRef(i, k) = mu.coeff(i, k1) - nu * t;
                mu.coeffRef(i, k1) = t + mu.coeff(k, k1) * mu.coeff(i, k);
            }
            
            k = k1;
        }else ++k;
    }
}

#endif
