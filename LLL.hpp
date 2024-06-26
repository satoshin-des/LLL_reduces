#ifndef __CPP_LLL__
#define __CPP_LLL__

#include <iostream>
#include <vector>
#include <tuple>

/* 内積 */
double dot(const std::vector<int> x, const std::vector<double> y){
	double z = 0.0;
	const int n = x.size();
	for(int i = 0; i < n; ++i) z += x.at(i) * y.at(i);
	return z;
}
double dot(const std::vector<double> x, const std::vector<double> y){
	double z = 0.0;
	const int n = x.size();
	for(int i = 0; i < n; ++i) z += x.at(i) * y.at(i);
	return z;
}
double dot(const std::vector<int> x, const std::vector<int> y){
	double z = 0.0;
	const int n = x.size();
	for(int i = 0; i < n; ++i) z += x.at(i) * y.at(i);
	return z;
}


/* Gram-Schmidtの直交化 */
std::tuple<std::vector<double>, std::vector<std::vector<double>>> Gram_Schmidt_squared(const std::vector<std::vector<int>> b){
	const int n = b.size(), m = b.at(0).size(); int i, j, k;
    std::vector<double> B(n);
	std::vector<std::vector<double>> GSOb(n, std::vector<double>(m)), mu(n, std::vector<double>(n));
	for(i = 0; i < n; ++i){
		mu.at(i).at(i)= 1.0;
		for(j = 0; j < m; ++j) GSOb.at(i).at(j) = b.at(i).at(j);
		for(j = 0; j < i; ++j){
			mu.at(i).at(j) = dot(b.at(i), GSOb.at(j)) / dot(GSOb.at(j), GSOb.at(j));
			for(k = 0; k < m; ++k) GSOb.at(i).at(k) -= mu.at(i).at(j) * GSOb.at(j).at(k);
		}
        B.at(i) = dot(GSOb.at(i), GSOb.at(i));
	}
	return std::forward_as_tuple(B, mu);
}


/* 部分サイズ基底簡約 */
void SizeReduce(std::vector<std::vector<int>>& b, std::vector<std::vector<double>>& mu, const int i, const int j){
	int q;
	const int m = b.at(0).size();
	if(mu.at(i).at(j) > 0.5 || mu.at(i).at(j) < -0.5){
		q = round(mu.at(i).at(j));
		for(int k = 0; k < m; ++k) b.at(i).at(k) -= q * b.at(j).at(k);
		for(int k = 0; k <= j; ++k) mu.at(i).at(k) -= mu.at(j).at(k) * q;
	}
}


/* LLL基底簡約 */
void LLLReduce(std::vector<std::vector<int>>& b, const float d = 0.99){
	const int n = b.size(), m = b.at(0).size(); int j, i, h;
	double t, nu, BB, C;
	std::vector<std::vector<double>> mu;
	std::vector<double> B; std::tie(B, mu) = Gram_Schmidt_squared(b);
	int tmp;
	for(int k = 1; k < n;){
		h = k - 1;

		for(j = h; j > -1; --j) SizeReduce(b, mu, k, j);

		//Checks if the lattice basis matrix b satisfies Lovasz condition.
		if(k > 0 && B.at(k) < (d - mu.at(k).at(h) * mu.at(k).at(h)) * B.at(h)){
			for(i = 0; i < m; ++i){tmp = b.at(h).at(i); b.at(h).at(i) = b.at(k).at(i); b.at(k).at(i) = tmp;}

			nu = mu.at(k).at(h); BB = B.at(k) + nu * nu * B.at(h); C = 1.0 / BB;
            mu.at(k).at(h) = nu * B.at(h) * C; B[k] *= B.at(h) * C; B.at(h) = BB;

            for(i = 0; i <= k - 2; ++i){
                t = mu.at(h).at(i); mu.at(h).at(i) = mu.at(k).at(i); mu.at(k).at(i) = t;
            }
            for(i = k + 1; i < n; ++i){
                t = mu.at(i).at(k); mu.at(i).at(k) = mu.at(i).at(h) - nu * t;
                mu.at(i).at(h) = t + mu.at(k).at(h) * mu.at(i).at(k);
            }

			--k;
		}else ++k;
	}
}
#endif