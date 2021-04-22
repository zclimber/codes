//
// Created by ME on 20.12.2020.
//

#pragma once

#include "msf.h"

#include <iostream>


struct BlockCodeTrellis {
    struct TrellisCell {
        float self_value = 0.;
        int prev_cells[2] = {-1, -1};
        int selected_cell = -1;
    };
    mutable std::vector<TrellisCell> data;
    std::vector<int> layer_start;
    std::vector<int> layer_end;
};

void dig_trellis_rec(std::vector<std::set<unsigned long long>> &set, const BlockCodeTrellis &tr, unsigned begin_layer,
                     int index, unsigned long long num,
                     unsigned long long depth) {
    if (index == -1)
        return;
    if (index < tr.layer_end[begin_layer]) {
        set[index - tr.layer_start[begin_layer]].insert(num);
        return;
    }
    dig_trellis_rec(set, tr, begin_layer, tr.data[index].prev_cells[0], num * 2, depth + 1);
    dig_trellis_rec(set, tr, begin_layer, tr.data[index].prev_cells[1], num * 2 + 1, depth + 1);
}

// result[finish][start]
std::vector<std::vector<std::set<unsigned long long>>>
dig_trellis_pos(unsigned start, unsigned finish, const BlockCodeTrellis &tr) {
    int finish_size = tr.layer_end[finish] - tr.layer_start[finish];
    int start_size = tr.layer_end[start] - tr.layer_start[start];
    std::vector<std::vector<std::set<unsigned long long>>> res;
    res.resize(finish_size);
    for (int fin = 0; fin < finish_size; fin++) {
        res[fin].resize(start_size);
        dig_trellis_rec(res[fin], tr, start, tr.layer_start[finish] + fin, 0, 0);
    }
    return res;
}

std::set<unsigned long long> dig_trellis(BlockCodeTrellis &tr) {
    return dig_trellis_pos(0, tr.layer_start.size() - 1, tr)[0][0];
}

void dig_trellis_rec(std::vector<unsigned long long>&set, const BlockCodeTrellis &tr, unsigned begin_layer,
                     int index, unsigned long long num,
                     unsigned long long depth) {
    if (index == -1)
        return;
    if (index < tr.layer_end[begin_layer]) {
      int set_index = index - tr.layer_start[begin_layer];
        set[set_index] = std::min(num, set[set_index]);
        return;
    }
    dig_trellis_rec(set, tr, begin_layer, tr.data[index].prev_cells[0], num * 2, depth + 1);
    dig_trellis_rec(set, tr, begin_layer, tr.data[index].prev_cells[1], num * 2 + 1, depth + 1);
}

// result[finish][start]
void 
dig_trellis_pos(unsigned start, unsigned finish, const BlockCodeTrellis &tr, std::vector<std::vector<unsigned long long>>& res) {
    int finish_size = tr.layer_end[finish] - tr.layer_start[finish];
    int start_size = tr.layer_end[start] - tr.layer_start[start];
    res.resize(finish_size);
    for (int fin = 0; fin < finish_size; fin++) {
        res[fin].assign(start_size, std::numeric_limits<unsigned long long>::max());
        dig_trellis_rec(res[fin], tr, start, tr.layer_start[finish] + fin, 0, 0);
    }
}

BlockCodeTrellis CreateCodeTrellisFromGenMatrix(const matrix &orig_gen_matrix) {
    int n = orig_gen_matrix.front().size();
    int k = orig_gen_matrix.size();
    matrix gen_matrix(orig_gen_matrix);
    std::vector<int> starting_row(n, -1), ending_row(n, -1);
    FindActiveRows(n, k, gen_matrix, starting_row, ending_row);

    int first_index = 0;
    std::vector<BlockCodeTrellis::TrellisCell> data(1);
    std::set<int> active_rows_set;
    std::vector<unsigned> previous_mask_value(k);
    std::vector<int> layer_start, layer_end;
    for (int index = 0; index < n; index++) {
        if (starting_row[index] > -1) {
            active_rows_set.insert(starting_row[index]);
        }
        if (ending_row[index] > -1) {
            active_rows_set.erase(ending_row[index]);
        }
        std::vector<unsigned char> active_rows(active_rows_set.begin(), active_rows_set.end());
        if (active_rows.size() > 30) {
            std::cout << "Logarithmic difficulty of trellis is >30, aborting";
            exit(1);
        }
        layer_start.push_back(first_index);
        first_index = data.size();
        layer_end.push_back(first_index);
        data.resize(data.size() + (1U << active_rows.size()));
        for (unsigned mask = 0; mask < (1U << active_rows.size()); mask++) {
            unsigned char this_sym = 0;
            unsigned prev_index = 0;
            for (unsigned i = 0; i < active_rows.size(); i++) {
                if (mask & (1U << i)) {
                    this_sym ^= gen_matrix[active_rows[i]][index];
                    prev_index |= previous_mask_value[active_rows[i]];
                }
            }
            data[first_index + mask].prev_cells[this_sym] = layer_start.back() + (int) prev_index;
            if (ending_row[index] != -1) {
                auto another_prev = prev_index | previous_mask_value[ending_row[index]];
                data[first_index + mask].prev_cells[this_sym ^ 1U] =
                        layer_start.back() + (int) another_prev;
            }
        }
        for (unsigned i = 0; i < active_rows.size(); i++) {
            previous_mask_value[active_rows[i]] = 1U << i;
        }
    }
    layer_start.push_back(first_index);
    layer_end.push_back(data.size());
    return BlockCodeTrellis{data, layer_start, layer_end};
}

unsigned long long vit_comps = 0, vit_adds = 0;

class ViterbiSoftDecoder {
public:
    explicit ViterbiSoftDecoder(const matrix &orig_gen_matrix) {
        gen_matrix_ = orig_gen_matrix;
        GenerateMinimalSpanMatrix(gen_matrix_, orig_gen_matrix[0].size(), orig_gen_matrix.size(), &info_transform_matrix_);
        if (gen_matrix_ == orig_gen_matrix) {
            info_transform_matrix_.clear();
        }
        else {
            info_transform_matrix_inverse_ = InverseMatrix(info_transform_matrix_);
        }
        for (auto row : gen_matrix_) {
            row_starts_.push_back(std::find(row.begin(), row.end(), 1) - row.begin());
        }
        trellis_ = CreateCodeTrellisFromGenMatrix(orig_gen_matrix);
    }

    void Encode(const std::vector<uint8_t>& info, std::vector<uint8_t>& encoded) {
        if (info_transform_matrix_.empty()) {
            MultiplyVectorByMatrix(info, gen_matrix_, encoded);
        }
        else {
            MultiplyVectorByMatrix(info, info_transform_matrix_inverse_, temp_vector_);
            MultiplyVectorByMatrix(temp_vector_, gen_matrix_, encoded);
        }
    }

    float DecodeInputToCodeword(const std::vector<float> &data, std::vector<unsigned char> &res) const {
        res.resize(data.size());
        int current_symbol_num = 0;
        float current_symbol_value;
        for (int i = 1; i < trellis_.data.size(); i++) {
            if (i == trellis_.layer_end[current_symbol_num]) {
                current_symbol_num++;
                current_symbol_value = data[current_symbol_num - 1];
            }
            BlockCodeTrellis::TrellisCell &cur = trellis_.data[i];
            vit_adds++;
            if (cur.prev_cells[0] == -1) {
                cur.selected_cell = 1;
                cur.self_value = trellis_.data[cur.prev_cells[1]].self_value + current_symbol_value;
            } else if (cur.prev_cells[1] == -1) {
                cur.selected_cell = 0;
                cur.self_value = trellis_.data[cur.prev_cells[0]].self_value - current_symbol_value;
            } else {
                vit_adds++;
                vit_comps++;
                float value_0 = trellis_.data[cur.prev_cells[0]].self_value - current_symbol_value;
                float value_1 = trellis_.data[cur.prev_cells[1]].self_value + current_symbol_value;
                if (value_0 >= value_1) {
                    cur.self_value = value_0;
                    cur.selected_cell = 0;
                } else {
                    cur.self_value = value_1;
                    cur.selected_cell = 1;
                }
            }
        }
        int cur_cell = trellis_.layer_start.back();
        for (int sym = (int) data.size() - 1; sym >= 0; sym--) {
            res[sym] = trellis_.data[cur_cell].selected_cell;
            cur_cell = trellis_.data[cur_cell].prev_cells[res[sym]];
        }
        return trellis_.data.back().self_value;
    }

    void DecodeMessageFromCodeword(std::vector<unsigned char> &data, std::vector<unsigned char> &res) {
        res.resize(gen_matrix_.size());
        for (int row = 0; row < gen_matrix_.size(); row++) {
            res[row] = data[row_starts_[row]];
            if (res[row])
                XorVectors(data, gen_matrix_[row]);
        }
    }

    std::vector<unsigned char> decode(const std::vector<float> &data) const {
        std::vector<unsigned char> res;
        DecodeInputToCodeword(data, res);
        return res;
    }

    matrix gen_matrix_;
    matrix info_transform_matrix_;
    matrix info_transform_matrix_inverse_;
    std::vector<uint8_t> temp_vector_;
    std::vector<int> row_starts_;
    BlockCodeTrellis trellis_;
};
