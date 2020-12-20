//
// Created by ME on 20.12.2020.
//

#pragma once

#include "base.h"

matrix GenerateReedMullerCode(int r, int m) {
    int n = 1 << m;
    matrix res(1, std::vector<unsigned char>(n, 1));
    if (r == 0)
        return res;
    std::vector<unsigned char> temp(n);
    for (unsigned i = 0; i < m; i++) {
        for (unsigned mask = 0; mask < n; mask++) {
            temp[mask] = (mask >> i) & 1U;
        }
        res.push_back(temp);
    }
    if (r == 1)
        return res;
    for (int bits = 2; bits <= r; bits++) {
        for (unsigned mask = 0; mask < n; mask++) {
            temp.assign(n, 1);
            int bitcount = 0;
            for (unsigned i = 0; i < m; i++) {
                if ((mask >> i) & 1U) {
                    bitcount++;
                    AndVectors(temp, res[i + 1]);
                }
            }
            if (bitcount == bits)
                res.push_back(temp);
        }
    }
    return res;
}
