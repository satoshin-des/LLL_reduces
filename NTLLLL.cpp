#include <iostream>
#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/LLL.h>

int main(){
    NTL::ZZ _;
    int t[][40] = {
        {-1, 3, 0, -1, -1, 0, -6, 0, -28, -2, 0, -1, 0, -1, 0, -15, 0, -6, 0, -1, -1, -1, -1, -28, -99, -2, 2, -1, -1, -1, 1, 11, 1, -3, 0, 0, 2, 1, -2, 0 },
        {1, -1, 13, 1, -1, -1, 3, -10, 1, 1, -3, 1, 1, 2, 2, -1, -1, -1, 1, 0, 2, -2, -1, 1, 1, 3, 1, -1, 2, -2, -9, -3, 1, -3, 0, -2, -1, -4, -1, 1 },
        {-3, 0, -3, 1, 1, -3, -3, 6, 2, -1, 1, 0, 0, -4, -1, -1, -4, 1, 1, 3, 0, 2, -1, -1, -9, -1, -1, -1, 1, 0, 0, 1, 0, -1, 1, -1, 0, 2, 13, 0 },
        {-1, -1, 1, 1, 0, 0, -2, 0, -9, 2, 1, -1, -1, 1, 2, -2, 1, 1, -1, -20, 0, 1, 0, -1, -1, -1, -1, 296, -3, -1, 1, 2, -11, 2, 8, 14, -1, 0, -3, 1 },
        {-2, 0, -1, 1, -1, -1, 1, -1, 1, 6, -1, 0, -1, 1, -11, 1, 1, -1, 1, 1, -3, -1, -1, -1, 4, 5, -4, -1, 3, 0, 1, -5, -1, -2, 1, -1, 1, -3, 0, -5 },
        {0, 0, 6, 2, -1, 1, 1, 0, 1, 12, 1, -6, 3, 1, 0, -1, 0, 0, 0, 2, 0, 3, -1, 0, 0, -1, 1, 1, 0, 12, -1, 0, 119, -2, -11, 29, 0, 1, 2, -2 },
        {0, 1, 14, -2, 1, 2, 1, -3, -1, 0, 1, 1, -3, 0, 2, -2, -1, -1, -2, 0, 1, 15, 14, -4, 1, 0, -1, 1, 1, 1, 4, -1, 17, 1, -2, 1, 3, 0, 3, -1 },
        {1, -3, -1, 0, -1, 1, -3, 13, -1, -1, 3, -2, 1, 1, 1, 1, -3, 2, -6, -13, 1, 0, 1, -2, -1, 2, 0, -1, -3, 0, -1, 11, -3, 11, -4, 1, 1, -2, -4, -3 },
        {0, -7, -1, 0, 1, 0, -37, 0, 0, 1, -16, 0, 1, -2, 0, -7, 5, 2, 2, -1, 1, 2, -12, 0, 0, 4, -1, -1, 1, 17, 13, 0, 2, 2, -1, 0, 0, -1, 5, -8 },
        {1, 1, -1, 1, 3, 0, -1, 2, -1, 2, -1, -7, 2, -1, 1, -4, 3, 1, 2, -1, 0, 4, 1, -11, 6, -144, 1, -3, 0, -35, 2, 3, 1, -1, 1, 4, 12, 3, 0, 2 },
        {-1, -1, 1, -1, 9, 0, 1, 1, 0, 2, 6, -2, -2, 1, -1, 142, 0, 1, 0, 3, 2, -1, 1, 0, -1, -2, -12, -13, -1, 3, 5, 2, -4, -1, -2, -2, -3, 0, 0, 1 },
        {0, 17, 0, 2, -16, -3, 0, 0, 1, 26, 1, -3, 3, -2, -4, 0, -2, 0, 1, 1, -1, -6, 1, -5, 1, 1, 1, 0, -1, -3, 18, 4, 2, -3, -107, -5, -6, 0, -3, 1 },
        {2, -2, 3, 1, 3, -7, -1, 1, 1, 0, 7, -5, 0, -1, 0, 68, 2, 1, -1, 4, 0, 7, 0, 3, 11, 1, -1, -2, 0, 0, 26, -1, 5, -4, -1, 4, -1, -3, 1, -1 },
        {1, 1, -5, 1, -1, -1, 0, 1, -40, 1, -1, -1, 1, 1, 2, 1, 0, -30, 1, -1, 0, 2, 1, 0, -2, -1, 0, -6, 1, 2, 1, 2, 1, -1, 0, 0, 1, -2, 1, 3 },
        {-7, 5, -2, 2, 3, 0, -1, 1, 1, -218, 2, -1, -1, 2, -2, -1, 0, -1, 0, 5, 1, -1, -1, 1, 0, 1, 1, 1, 1, 1, 0, -3, 3, -1, 1, -1, 0, 1, -2, 1 },
        {0, 0, -3, -4, 1, -1, -12, -3, 9, 2, 0, -1, -15, -1, -1, 1, -1, -3, 0, -1, -1, 0, 1, 0, 1, 0, -9, 0, 0, 6, 0, 1, -1, -1, -10, -1, 23, 8, 0, 5 },
        {2, 1, -4, 0, 0, 1, 1, -1, 2, 6, 1, 2, 4, 0, 2, -1, 22, -1, 0, 1, -3, 2, 1, 0, 1, 0, -3, -2, 0, 4, 0, -2, 1, 3, 1, 1, 0, 1, 1, 0 },
        {-12, -1, 0, 1, 0, 1, 1, -1, 1, -27, -1, 2, 1, 1, 1, 5, -1, 2, 1, 0, 2, -8, 0, 0, 11, 95, 2, 1, -1, 1, -1, -2, -3, 0, -2, -2, 2, -1, -3, -2 },
        {9, -1, -1, 1, -1, 4, -1, -1, 1, 0, -3, 1, 1, 6, 1, -1, -1, 1, 1, 0, 0, 0, 0, -1, 3, 2, 0, -1, -1, -1, 3, 1, 0, -1, 0, -1, -1, -3, 1, -1 },
        {-5, -4, -7, -2, 1, -1, 3, 2, -3, -1, 1, 0, 14, 1, 0, 0, 0, 1, -1, 3, -1, -1, 2, -1, -4, 1, -4, 1, 0, 0, 142, -1, 2, -1, 0, -10, 3, -4, 1, -2 },
        {0, 1, 0, 1, -2, -1, 0, 0, 1, 0, -1, 1, 2, 0, 0, 3, -1, 1, -1, 0, -1, 0, 156, 5, 0, -3, -3, 7, 1, 184, 4, 1, -33, 1, -5, 0, 8, 0, 1, 0 },
        {-1, -1, 1, -1, 2, 0, 0, 1, 1, 5, -47, 1, 1, -3, -15, 0, 3, 1, 2, 3, 1, -2, -9, -15, 49, -3, 0, -16, -1, 0, -1, 1, 0, -1, 3, 1, -1, 6, -29, -1 },
        {0, -7, -1, -1, -3, 2, -2, 20, 0, -1, 170, 30, 0, -1, 1, 1, -1, 3, 1, 2, -1, 2, 4, 1, -2, 3, -1, 0, 1, 0, -7, 1, -1, 323, -1, -1, 0, 0, 8, -22 },
        {-1, -18, 1, -2, -4, -1, 1, 0, 1, -2, 1, -1, 7, -1, 1, -1, 3, -1, -3, 0, 3, 2, 3, -1, -1, 1, 0, -1, 1, 3, -1, 0, -12, -1, 4, 0, -3, 1, -1, 0 },
        {1, 1, -17, 0, -1, 3, 4, -1, -4, -1, 2, 1, -2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 3, 5, 1, 2, -1, 2, 1, -4, 0, 0, 4, 1, 24, 0, -1, 2 },
        {0, -2, 8, 2, -1, 4, 1, -1, 10, 1, -1, 2, 3, 0, 7, -1, 8, -1, -6, 5, -1, 1, -123, -1, 1, 3, -1, -2, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, -1 },
        {3, 1, 7, -1, -3, -1, 3, -1, 3, -1, -65154, -1, 1, 2, -2, 3, 1, -23, 10, -2, 0, 0, 0, 3, -1, 2, 0, -1, -1, 8, -5, 1, 0, -2, 0, 0, 0, 1, -1, 3 },
        {0, -1, -4, 2, 4, 19, 3, 0, -1, 1, -2, -2, 2, -1, 0, -1, 0, -1, -1, 0, 1, -1, -1, -2, 0, 8, 0, 0, 0, -8, -1, -1, 1, 2, 1, 3, -1, 9, -2, 1 },
        {1, -1, 1, 1, 25, 2, -2, 0, 4, 1, 1, 0, -1, 0, 1, -6, 0, 1, -1, -1, -3, 0, 1, -2, 59, 0, 1, -2, 1, 1, -1, -1, 0, 2, 1, -1, 1, 0, 1, -3 },
        {1, 4, 1, 4, 1, -2, -58, 0, -9, -1, -1, -1, -1, 0, -20, 0, -1, 0, -2, 0, 1, 4, 2, 5, 19, -1, -3, 1, 140, 1, -2, -1, 0, -1, 3, 2, -1, 0, -2, -1 },
        {18, -1, 1, -3, -1, -1, 0, 2, 14, -2, -1, 1, 0, -1, 2, -1, -1, 0, -4, 1, 1, -7, -2, -1, -4, 2, 3, -1, -2, -1, -6, -1, -1, 0, -39, 3, 0, 0, -2, 40 },
        {-5, 1, 0, -1, 2, 0, 3, 0, 1, 6, 1, 0, 2, 0, -3, -4, -1270, 1, -1, 1, -2, 2, -2, -1, -4, 1, -1, -2, 1, -2, 2, 0, -3, 2, 0, 1, -1, 0, -1, 1 },
        {-49, 0, 0, -1, 1, 1, -3, -9, 0, 0, 0, -2, 1, 1, -47, 0, 0, 0, 4, 6, -2, -2, 0, 2, 1, 2, 1, -1, 1, 4, 7, 0, 1, -1, 0, -1, -1, 1, 0, -1 },
        {-2, -1, -2, 1, 17, 1, 10, 6, 0, -1, -1, -2, -3, 1, 1, 1, 5, 18, 1, 0, 49, 7, 0, 2, 0, 0, -1, 0, 0, 0, 2, -2, 1, 0, 0, -1, 0, 0, -2, 3 },
        {-1, -1, 166, 0, -46, 0, -2, 0, 1, 1, 15, -2, 1, 12, 1, 4, 0, -2, 1, -1, 0, -1, 0, 0, -25, -6, 0, -1, -4, 0, 1, 2, 78, -15, 2, 0, 0, -1, 17, 17 },
        {-13, 4, -16, -1, 0, 1, 0, 1222, 1, -1, 1, 0, 0, 1, 1, 1, 0, 0, -1, 1, 0, 0, 0, 0, -1, 0, -5, 3, 1, 1, -1, -1, -1, -1, 3, 0, 0, -4, 2, -1 },
        {1, 2, -2, 0, 0, -4, 1, -1, 9, -1, -1, -13, 24, -1, -1, -8, 1, -1, -29, 0, 1, 0, -1, 1, 1, 1, 1, 4, 9, -1, -1, 0, -18, -1, -1, 1, 6, -1, -4, 0 },
        {6, -13, 0, 0, 0, 15, 0, -3, 0, -2, 0, -2, 1, -1, -2, -1, 0, 1, -2, 0, 1, -1, 1, -1, -9, 1, 6, -5, -9, 0, 1, -2, -1, -2, 9, 1, 1, 3, 0, 1 },
        {1, 3, -2, -1, -3, 1, -2, -1, -1, -2, 460, 1, 2, -8, 0, 2, 0, 4, 0, 1, -1, 0, 0, -1, -1, 4, 23, -1, -1, -1, -1, -4, 2, 15, -2, 3, 0, -1, 2, -1 },
        {0, 0, -1, 1, 69, 4, 0, -2, 0, 8, 1, 4, 1, 2, -2, 0, -2, -1, 8, -4, 1, -11, 0, -1, -1, 0, -1, 0, 1, -1, -11, 0, 505, 3, 8, -1, -1, 1, -3, 1}
    };
    NTL::mat_ZZ b; b.SetDims(40, 40);
    for(int i = 0, j; i < 40; ++i){
        for(j = 0; j < 40; ++j)
            b[i][j] = t[i][j];
    }

    NTL::LLL(_, b);
    std::cout << b << std::endl << "norm = " << NTL::sqrt(NTL::to_RR(b[0] * b[0])) << std::endl;
}