//
// Created by ME on 20.12.2020.
//

#pragma once

#include "base.h"

void GenerateMinimalSpanMatrixForwardPass(matrix &gen_matrix, int n, int k, matrix* key_matrix = nullptr) {
    int current_base_row = 0;
    for (int column = 0; column < n && current_base_row < k; column++) {
        int found_base = k;
        for (auto i = current_base_row; i < k && found_base == k; i++) {
            if (gen_matrix[i][column] == 1) {
                found_base = i;
                swap(gen_matrix[i], gen_matrix[current_base_row]);
                if (key_matrix) {
                    swap(key_matrix->at(i), key_matrix->at(current_base_row));
                }
                current_base_row++;
            }
        }
        for (auto i = found_base + 1; i < k; i++) {
            if (gen_matrix[i][column] == 1) {
                XorVectors(gen_matrix[i], gen_matrix[current_base_row - 1]);
                if (key_matrix) {
                    XorVectors(key_matrix->at(i), key_matrix->at(current_base_row - 1));
                }
            }
        }
    }
}

void GenerateMinimalSpanMatrixBackwardPass(matrix &gen_matrix, int n, int k, matrix* key_matrix = nullptr) {
    std::vector<unsigned char> found_rows(k);
    int found_ending_rows = 0;
    for (int base = n - 1; base >= 0 && found_ending_rows < k; base--) {
        int found_index = -1;
        for (auto i = k - 1; i >= 0 && found_index == -1; i--) {
            if (found_rows[i])
                continue;
            if (gen_matrix[i][base] == 1) {
                found_index = i;
                found_ending_rows++;
                found_rows[i] = true;
            }
        }
        for (auto i = found_index - 1; i >= 0; i--) {
            if (found_rows[i])
                continue;
            if (gen_matrix[i][base] == 1) {
                XorVectors(gen_matrix[i], gen_matrix[found_index]);
                if (key_matrix) {
                    XorVectors(key_matrix->at(i), key_matrix->at(found_index));
                }
            }
        }
    }
}

void GenerateMinimalSpanMatrix(matrix &gen_matrix, int n, int k, matrix *key_matrix = nullptr) {
    if (key_matrix) {
        key_matrix->assign(k, std::vector<uint8_t>(k, 0));
        for (int i = 0; i < k; i++) {
            (*key_matrix)[i][i] = 1;
        }
    }
    GenerateMinimalSpanMatrixForwardPass(gen_matrix, n, k, key_matrix);
    GenerateMinimalSpanMatrixBackwardPass(gen_matrix, n, k, key_matrix);
}

void
FindActiveRows(int n, int k, const matrix &gen_matrix, std::vector<int> &starting_row, std::vector<int> &ending_row) {
    for (int row_index = 0; row_index < k; row_index++) {
        for (int i = 0; i < n; i++) {
            if (gen_matrix[row_index][i] == 1) {
                starting_row[i] = row_index;
                break;
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            if (gen_matrix[row_index][i] == 1) {
                ending_row[i] = row_index;
                break;
            }
        }
    }
}
