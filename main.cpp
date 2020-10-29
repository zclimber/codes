#include <iostream>
#include <iomanip>
#include <utility>
#include <vector>
#include <random>
#include <set>

using matrix = std::vector<std::vector<unsigned char>>;

void XorVectors(std::vector<unsigned char> &vec1, const std::vector<unsigned char> &vec2) {
    for (int j = 0; j < vec2.size(); j++) {
        vec1[j] ^= vec2[j];
    }
}

void GenerateMinimalSpanMatrixForwardPass(matrix &gen_matrix, int n, int k) {
    int current_base_row = 0;
    for (int column = 0; column < n && current_base_row < k; column++) {
        int found_base = k;
        for (auto i = current_base_row; i < k && found_base == k; i++) {
            if (gen_matrix[i][column] == 1) {
                found_base = i;
                swap(gen_matrix[i], gen_matrix[current_base_row]);
                current_base_row++;
            }
        }
        for (auto i = found_base + 1; i < k; i++) {
            if (gen_matrix[i][column] == 1) {
                XorVectors(gen_matrix[i], gen_matrix[current_base_row - 1]);
            }
        }
    }
}

void GenerateMinimalSpanMatrixBackwardPass(matrix &gen_matrix, int n, int k) {
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
            }
        }
    }
}

void GenerateMinimalSpanMatrix(matrix &gen_matrix, int n, int k) {
    GenerateMinimalSpanMatrixForwardPass(gen_matrix, n, k);
    GenerateMinimalSpanMatrixBackwardPass(gen_matrix, n, k);
}

class AWGNChannel {
public:
    explicit AWGNChannel(float noise_sigma) : gen(std::random_device{}()), noise(0., noise_sigma) {
    }

    std::vector<float> transmit(const std::vector<unsigned char> &data) {
        std::vector<float> result;
        result.reserve(data.size());
        for (auto bit : data) {
            result.push_back(noise(gen) + (bit == 1 ? 1.f : -1.f));
        }
        return result;
    }

private:
    std::mt19937 gen;
    std::normal_distribution<float> noise;
};

class SimpleEncoder {
public:
    explicit SimpleEncoder(matrix gen_matrix) : n(gen_matrix.front().size()), k(gen_matrix.size()),
                                                gen_matrix(std::move(gen_matrix)) {
    }

    std::vector<unsigned char> encode(const std::vector<unsigned char> &data) {
        std::vector<unsigned char> res(gen_matrix.front().size(), 0);
        for (auto row = 0U; row < k; row++) {
            if (data[row] == 1) {
                XorVectors(res, gen_matrix[row]);
            }
        }
        return res;
    }

private:
    int n, k;
    matrix gen_matrix;
};

struct BlockCodeTrellis {
    struct TrellisCell {
        float self_value = 0.;
        int prev_cells[2] = {-1, -1};
        int selected_cell;
    };
    mutable std::vector<TrellisCell> data;
    std::vector<int> layer_start;
};

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
    std::vector<int> layer_start;
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
    return BlockCodeTrellis{data, layer_start};
}

class ViterbiSoftDecoder {
public:
    explicit ViterbiSoftDecoder(const matrix &orig_gen_matrix) {
        gen_matrix = orig_gen_matrix;
        for (auto row : gen_matrix) {
            row_starts.push_back(std::find(row.begin(), row.end(), 1) - row.begin());
        }
        trellis = CreateCodeTrellisFromGenMatrix(orig_gen_matrix);
    }

    std::vector<unsigned char> DecodeInputToCodeword(const std::vector<float> &data, float &probability_log) const {
        std::vector<unsigned char> res(data.size());
        const float *current_sym = data.data() - 1;
        const int *next_section = trellis.layer_start.data() + 1;
        for (int i = 1; i < trellis.data.size(); i++) {
            if (i == *next_section) {
                next_section++;
                current_sym++;
            }
            BlockCodeTrellis::TrellisCell &cur = trellis.data[i];
            if (cur.prev_cells[0] == -1) {
                cur.selected_cell = 1;
                cur.self_value = trellis.data[cur.prev_cells[1]].self_value + *current_sym;
            } else if (cur.prev_cells[1] == -1) {
                cur.selected_cell = 0;
                cur.self_value = trellis.data[cur.prev_cells[0]].self_value - *current_sym;
            } else {
                float value_0 = trellis.data[cur.prev_cells[0]].self_value - *current_sym;
                float value_1 = trellis.data[cur.prev_cells[1]].self_value + *current_sym;
                if (value_0 >= value_1) {
                    cur.self_value = value_0;
                    cur.selected_cell = 0;
                } else {
                    cur.self_value = value_1;
                    cur.selected_cell = 1;
                }
            }
        }
        probability_log = trellis.data.back().self_value;
        int cur_cell = (int) trellis.data.size() - 1;
        for (int sym = (int) data.size() - 1; sym >= 0; sym--) {
            res[sym] = trellis.data[cur_cell].selected_cell;
            cur_cell = trellis.data[cur_cell].prev_cells[res[sym]];
        }
        return res;
    }

    std::vector<unsigned char> DecodeMessageFromCodeword(std::vector<unsigned char> &data) {
        std::vector<unsigned char> res(gen_matrix.size());
        for (int row = 0; row < gen_matrix.size(); row++) {
            res[row] = data[row_starts[row]];
            if (res[row])
                XorVectors(data, gen_matrix[row]);
        }
        return res;
    }

    std::vector<unsigned char> decode(const std::vector<float> &data, float &probability_log) const {
        std::vector<unsigned char> res;
        return res;
    }

private:
    matrix gen_matrix;
    std::vector<int> row_starts;
    BlockCodeTrellis trellis;
};

std::string PrintVector(const std::vector<unsigned char> &data) {
    std::string res;
    res.resize((data.size() * 2), '0');
    for (int i = 0; i < data.size(); i++) {
        res[i * 2] += data[i];
        res[i * 2 + 1] = ' ';
    }
    res.pop_back();
    return res;
}

int main() {
    int n, k, words;
    float noise_sigma;
    std::cin >> n >> k >> noise_sigma >> words;
    matrix code_gen_matrix(k);
    for (int i = 0; i < k; i++) {
        code_gen_matrix[i].resize(n);
        for (int j = 0; j < n; j++) {
            int x;
            std::cin >> x;
            code_gen_matrix[i][j] = x;
        }
    }

    // to minimal span form
    GenerateMinimalSpanMatrix(code_gen_matrix, n, k);

    std::cout << "Converted code matrix to minimal span form:\n";
    // output minimal span matrix
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << (int) code_gen_matrix[i][j] << " ";
        }
        std::cout << "\n";
    }

    SimpleEncoder encoder(code_gen_matrix);
    AWGNChannel channel(0.8);
    ViterbiSoftDecoder decoder(code_gen_matrix);

    std::vector<unsigned char> input(k);
    std::mt19937 rand_gen(std::random_device{}());
    for (int i = 0; i < words; i++) {
        for (auto &bit : input) {
            bit = rand_gen() & 1U;
        }
        std::cout << "\nGenerated random input:\n" << PrintVector(input) << "\n";
        auto encoded = encoder.encode(input);
        std::cout << "Encoded it into:\n" << PrintVector(encoded) << "\n";
        auto transmitted = channel.transmit(encoded);
        std::cout << "Data was transmitted as:\n";
        std::cout << std::fixed << std::setprecision(2);
        for (float data : transmitted)
            std::cout << data << " ";
        std::cout << "\n";
        float prob_log;
        auto codeword = decoder.DecodeInputToCodeword(transmitted, prob_log);
        std::cout << "Transmitted data was restored with cumulative ML of " << prob_log << ":\n"
                  << PrintVector(codeword)
                  << "\n";
        auto decoded = decoder.DecodeMessageFromCodeword(codeword);
        std::cout << "Codeword was decoded into:\n" << PrintVector(decoded) << "\n";
        if (decoded != input) {
            std::cout << "Message was transmitted incorrectly!\n";
        }
    }
    return 0;
}
