#include <iostream>
#include <iomanip>
#include <utility>
#include <vector>
#include <random>
#include <set>
#include <map>
#include <thread>
#include <mutex>
#include <string>

using matrix = std::vector<std::vector<unsigned char>>;

void XorVectors(std::vector<unsigned char> &vec1, const std::vector<unsigned char> &vec2) {
    for (int j = 0; j < vec2.size(); j++) {
        vec1[j] ^= vec2[j];
    }
}

void AndVectors(std::vector<unsigned char> &vec1, const std::vector<unsigned char> &vec2) {
    for (int j = 0; j < vec2.size(); j++) {
        vec1[j] &= vec2[j];
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

    void transmit(const std::vector<unsigned char> &data, std::vector<float> &result) {
        result.resize(data.size());
        for (auto i = 0U; i < data.size(); i++) {
            result[i] = noise(gen) + (data[i] == 1 ? 1.f : -1.f);
        }
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

    void encode(const std::vector<unsigned char> &data, std::vector<unsigned char> &res) {
        res.assign(gen_matrix.front().size(), 0U);
        for (int row = 0; row < k; row++) {
            if (data[row] == 1) {
                XorVectors(res, gen_matrix[row]);
            }
        }
    }

private:
    int n = 0, k = 0;
    matrix gen_matrix;
};

struct BlockCodeTrellis {
    struct TrellisCell {
        float self_value = 0.;
        int prev_cells[2] = {-1, -1};
        int selected_cell = -1;
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

    float DecodeInputToCodeword(const std::vector<float> &data, std::vector<unsigned char> &res) const {
        res.resize(data.size());
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
        int cur_cell = (int) trellis.data.size() - 1;
        for (int sym = (int) data.size() - 1; sym >= 0; sym--) {
            res[sym] = trellis.data[cur_cell].selected_cell;
            cur_cell = trellis.data[cur_cell].prev_cells[res[sym]];
        }
        return trellis.data.back().self_value;
    }

    void DecodeMessageFromCodeword(std::vector<unsigned char> &data, std::vector<unsigned char> &res) {
        res.resize(gen_matrix.size());
        for (int row = 0; row < gen_matrix.size(); row++) {
            res[row] = data[row_starts[row]];
            if (res[row])
                XorVectors(data, gen_matrix[row]);
        }
    }

    std::vector<unsigned char> decode(const std::vector<float> &data, float &probability_log) const {
        std::vector<unsigned char> res;
        return res;
    }

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

void dig_trellis_rec(std::set<unsigned long long> &set, BlockCodeTrellis &tr, int index, unsigned long long num,
                     unsigned long long depth) {
    if (index == -1)
        return;
    if (index == 0) {
        set.insert(num);
        return;
    }
    dig_trellis_rec(set, tr, tr.data[index].prev_cells[0], num, depth + 1);
    dig_trellis_rec(set, tr, tr.data[index].prev_cells[1], num + (1UL << depth), depth + 1);
}

std::set<unsigned long long> dig_trellis(BlockCodeTrellis &tr) {
    std::set<unsigned long long> res;
    dig_trellis_rec(res, tr, tr.data.size() - 1, 0, 0);
    return res;
}

unsigned long long vector_to_code(const std::vector<unsigned char> &vec) {
    unsigned long long res = 0;
    for (auto c : vec) {
        res = (res << 1U) + (c & 1U);
    }
    return res;
}

std::vector<unsigned char> code_to_vector(unsigned long long code, int size) {
    std::vector<unsigned char> res(size);
    for (int i = size - 1; i >= 0; i--) {
        res[i] = code & 1U;
        code >>= 1U;
    }
    return res;
}

std::set<unsigned long long> gen_all_codewords(const matrix &gen_matrix) {
    std::vector<unsigned long long> base;
    base.reserve(gen_matrix.size());
    for (const auto &vec: gen_matrix) {
        base.push_back(vector_to_code(vec));
    }
    std::set<unsigned long long> res;
    for (unsigned num = 0; num < (1U << base.size()); num++) {
        unsigned long long cur = 0;
        for (unsigned i = 0; i < base.size(); i++) {
            cur ^= (num & (1U << i)) ? base[i] : 0;
        }
        res.insert(cur);
    }
    return res;
}

[[maybe_unused]] void TestGenerateMinSpan() {
    // test minspan
    std::random_device rd{};
    std::mt19937 gen{rd()};
    int n = 14, k = 6;
    matrix code_gen_matrix(k, std::vector<unsigned char>(n));
    for (int id = 0; id < 100000; id++) {
        auto rnd = gen();
        if (vector_to_code(code_to_vector(rnd, 32)) != rnd) {
            std::cerr << "ERROR RECODE at " << id << "\n";
            std::cerr << rnd << " " << vector_to_code(code_to_vector(rnd, 32)) << "\n";
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                code_gen_matrix[i][j] = gen() & 1U;
            }
        }
        auto before = gen_all_codewords(code_gen_matrix);
        GenerateMinimalSpanMatrix(code_gen_matrix, n, k);
        auto after = gen_all_codewords(code_gen_matrix);
        if (before != after) {
            std::cerr << "ERROR at " << id << "\n";
        }
    }
}

void CheckViterbiDecoderOnce(std::mt19937 &gen, int n, int k, matrix &code_gen_matrix, int id) {
    auto rnd = gen();
    if (vector_to_code(code_to_vector(rnd, 32)) != rnd) {
        std::cerr << "ERROR RECODE at " << id << "\n";
        std::cerr << rnd << " " << vector_to_code(code_to_vector(rnd, 32)) << "\n";
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            code_gen_matrix[i][j] = gen() & 1U;
        }
    }
    auto orig_matrix = code_gen_matrix;
    auto before = gen_all_codewords(code_gen_matrix);
    GenerateMinimalSpanMatrix(code_gen_matrix, n, k);
    auto after = gen_all_codewords(code_gen_matrix);
    if (before != after) {
        std::cerr << "ERROR at " << id << "\n";
    }

    for (auto &row: code_gen_matrix) {
        if (std::find(row.begin(), row.end(), 1) == row.end()) {
            CheckViterbiDecoderOnce(gen, n, k, code_gen_matrix, id);
            return;
        }
    }


    std::vector<unsigned char> input(k, 0);
    SimpleEncoder enc(code_gen_matrix);
    AWGNChannel channel(0.0001);
    ViterbiSoftDecoder dec(code_gen_matrix);

    auto codes = dig_trellis(dec.trellis);
    if (codes != before) {
        std::cerr << "ERROR TRELLIS at " << id << "\n";
        ViterbiSoftDecoder dec2(code_gen_matrix);
    }
    std::vector<unsigned char> encoded, restored, decoded;
    std::vector<float> transmitted;
    enc.encode(input, encoded);
    channel.transmit(encoded, transmitted);
    auto prob_log = dec.DecodeInputToCodeword(transmitted, restored);
    auto restored_copy = restored;
    dec.DecodeMessageFromCodeword(restored, decoded);
    if (decoded != input) {
        std::cout << "ERROR DECODING at " << id << "!!!\n";
        prob_log = dec.DecodeInputToCodeword(transmitted, restored);
        dec.DecodeMessageFromCodeword(restored, decoded);
        exit(1);
    }
}

[[maybe_unused]] void TestViterbiDecoderRandom() {
    // test minspan
    std::random_device rd{};
    std::mt19937 gen{rd()};
    int n = 20, k = 12;
    matrix code_gen_matrix(k, std::vector<unsigned char>(n));
    for (int id = 0; id < 100000; id++) {
        if (id % 100 == 0)
            std::cout << id << "\n";
        CheckViterbiDecoderOnce(gen, n, k, code_gen_matrix, id);
    }
}

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

struct ReedMullerChecker {
    std::map<int, std::map<std::string, double>> reed_muller_measures;
    std::set<std::string> reed_muller_names;
    std::mutex reed_muller_mutex;

    void SingleThreadReedMullerCheck(int code_r, int code_m, int iter, int snr_db_start, int snr_db_finish,
                                     int snr_db_step = 1) {
        matrix code_gen_matrix = GenerateReedMullerCode(code_r, code_m);
        int k = code_gen_matrix.size();
        int n = code_gen_matrix[0].size();
        GenerateMinimalSpanMatrix(code_gen_matrix, n, k);
        std::vector<unsigned char> input(k);
        std::vector<unsigned char> encoded, codeword, decoded;
        std::vector<float> transmitted;
        std::mt19937 rand_gen(std::random_device{}());

        SimpleEncoder encoder(code_gen_matrix);
        ViterbiSoftDecoder decoder(code_gen_matrix);
        std::string name = "RM(" + std::to_string(code_r) + "," + std::to_string(code_m) + ")";
        for (int snr_db = snr_db_start; snr_db <= snr_db_finish; snr_db += snr_db_step) {
            AWGNChannel channel((float) pow(10., 0.1 * snr_db));
            int correct = 0;
            for (int index = 0; index < iter; index++) {
                unsigned val = rand_gen();
                for (unsigned i = 0; i < k; i++) {
                    input[i] = (val >> i) & 1U;
                }
                encoder.encode(input, encoded);
                channel.transmit(encoded, transmitted);
                decoder.DecodeInputToCodeword(transmitted, codeword);
                if (codeword == encoded) {
                    correct++;
                }
            }
            std::lock_guard guard(reed_muller_mutex);
            reed_muller_measures[snr_db][name] = double(correct) / iter;
        }
        std::lock_guard guard(reed_muller_mutex);
        reed_muller_names.insert(std::move(name));
    }

    void MultiThreadedReedMullerCheck() {
        std::vector<std::thread> threads;
        threads.emplace_back([&]() {
            SingleThreadReedMullerCheck(1, 3, 10000000, -10, 10);
            SingleThreadReedMullerCheck(2, 3, 10000000, -10, 10);
        });
        threads.emplace_back([&]() {
            SingleThreadReedMullerCheck(1, 4, 10000000, -10, 10);
            SingleThreadReedMullerCheck(2, 4, 10000000, -10, 10);
        });
        threads.emplace_back([&]() {
            SingleThreadReedMullerCheck(1, 5, 10000000, -10, 0);
        });
        threads.emplace_back([&]() {
            SingleThreadReedMullerCheck(2, 5, 10000000, -10, 0);
        });
        threads.emplace_back([&]() {
            SingleThreadReedMullerCheck(1, 5, 10000000, 1, 10);
        });
        threads.emplace_back([&]() {
            SingleThreadReedMullerCheck(2, 5, 10000000, 1, 10);
        });
        for (auto &thr: threads) {
            thr.join();
        }
    }

    void PrintDataAsCSV() {
        for (auto &name: reed_muller_names) {
            std::cout << ";" << name;
        }
        std::cout << "\n";
        for (auto &[snr, measures]: reed_muller_measures) {
            std::cout << (double) snr / 10;
            for (auto &name: reed_muller_names) {
                if (auto it = measures.find(name); it != measures.end()) {
                    std::cout << it->second;
                }
                std::cout << ";";
            }
            std::cout << "\n";
        }
    }
};

int main() {
    ReedMullerChecker checker;
    checker.MultiThreadedReedMullerCheck();
    checker.PrintDataAsCSV();
    return 0;
}
