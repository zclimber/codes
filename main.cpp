#include <iostream>
#include <iomanip>
#include <utility>
#include <vector>
#include <random>
#include <set>
#include <map>
#include <thread>
#include <sstream>
#include <mutex>
#include <atomic>
#include <string>
#include <algorithm>
#include <numeric>

using matrix = std::vector<std::vector<unsigned char>>;

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

std::string PrintMatrix(const matrix &data) {
    std::stringstream ss;
    for (auto &row : data) {
        ss << PrintVector(row) << "\n";
    }
    return ss.str();
}

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

void MultiplyVectorByMatrix(const std::vector<unsigned char> &data, const matrix &gen_matrix,
                            std::vector<unsigned char> &res) {
    res.assign(gen_matrix.front().size(), 0U);
    for (int row = 0; row < gen_matrix.size(); row++) {
        if (data[row] == 1) {
            XorVectors(res, gen_matrix[row]);
        }
    }
}

void SwapColumns(matrix &m, int col1, int col2) {
    if (col1 == col2)
        return;
    for (int i = 0; i < m.size(); i++) {
        std::swap(m[i][col1], m[i][col2]);
    }
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
        MultiplyVectorByMatrix(data, gen_matrix, res);
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
    std::vector<int> layer_end;
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

unsigned long long vector_to_code(const std::vector<unsigned char> &vec) {
    unsigned long long res = 0;
    for (unsigned i = 0; i < vec.size(); i++) {
        res |= static_cast<unsigned long long>(vec[i]) << i;
    }
    return res;
}

std::vector<unsigned char> code_to_vector(unsigned long long code, int size) {
    std::vector<unsigned char> res(size);
    for (int i = 0; i < size; i++) {
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
            cur ^= (num & (1U << i)) ? base[base.size() - i - 1] : 0;
        }
        res.insert(cur);
    }
    return res;
}

void GetSystematicLikeMatrix(matrix &temp, std::vector<int> &columns) {
    int n = temp[0].size();
    int k = temp.size();

    columns.resize(n);
    std::iota(columns.begin(), columns.end(), 0);

    int pos = 0;
    for (int i = 0; i < k && pos < n; i++) {
        auto one = std::find(temp[i].begin() + pos, temp[i].end(), 1);
        if (one == temp[i].end()) {
            continue;
        }
        int one_pos = one - temp[i].begin();
        SwapColumns(temp, one_pos, pos);
        std::swap(columns[one_pos], columns[pos]);
        for (int j = 0; j < k; j++) {
            if (temp[j][pos] && j != i) {
                XorVectors(temp[j], temp[i]);
            }
        }
        pos++;
    }
}

matrix RevertColumnSwappedMatrix(const matrix &temp, const std::vector<int> &columns) {
    matrix res(temp.size());
    for (int i = 0; i < temp.size(); i++) {
        res[i].resize(columns.size());
        for (int j = 0; j < columns.size(); j++) {
            res[i][columns[j]] = temp[i][j];
        }
    }
    return res;
}

matrix GenerateTransposedCheckMatrix(const matrix &gen_matrix, int n, int k) {
    matrix temp = gen_matrix;
    std::vector<int> columns;
    GetSystematicLikeMatrix(temp, columns);

    int r = n - k;
    matrix res(n);
    for (int i = 0; i < n; i++) {
        if (i >= k) {
            res[i].resize(r);
            res[i][i - k] = 1;
        } else {
            res[i] = {temp[i].begin() + k, temp[i].end()};
        }
    }
    matrix resres(n);
    for (int i = 0; i < n; i++) {
        resres[columns[i]] = std::move(res[i]);
    }
    return resres;
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

void CheckVectorToCode(std::mt19937 &gen, int id) {
    auto rnd = gen();
    if (vector_to_code(code_to_vector(rnd, 32)) != rnd) {
        std::cerr << "ERROR RECODE at " << id << "\n";
        std::cerr << rnd << " " << vector_to_code(code_to_vector(rnd, 32)) << "\n";
    }
}

void CheckCheckMatrix(int n, int k, const matrix &code_gen_matrix) {
    auto check_matrix = GenerateTransposedCheckMatrix(code_gen_matrix, n, k);
    std::vector<unsigned char> checks;
    for (auto &code : code_gen_matrix) {
        MultiplyVectorByMatrix(code, check_matrix, checks);
        if (std::find(checks.begin(), checks.end(), 1) != checks.end()) {
            std::cerr << "WRONG CHECK MATRIX!!\n";
            std::cerr << PrintMatrix(code_gen_matrix) << "\n" << PrintMatrix(check_matrix) << "\n"
                      << PrintVector(checks)
                      << "\n\n";
            //GenerateTransposedCheckMatrix(gen_matrix, n, k);
            exit(1);
        }
    }
}

void GenerateRandomCode(std::mt19937 &gen, int n, int k, int id, matrix &code_gen_matrix,
                        std::set<unsigned long long int> &before) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            code_gen_matrix[i][j] = gen() & 1U;
        }
    }
    before = gen_all_codewords(code_gen_matrix);
    GenerateMinimalSpanMatrix(code_gen_matrix, n, k);
    if (before != gen_all_codewords(code_gen_matrix)) {
        std::cerr << "ERROR at " << id << "\n";
    }

    for (auto &row: code_gen_matrix) {
        if (std::find(row.begin(), row.end(), 1) == row.end()) {
            GenerateRandomCode(gen, n, k, id, code_gen_matrix, before);
        }
    }
}

void CheckViterbiDecoder(std::mt19937 &gen, int n, int k, int id, const matrix &code_gen_matrix,
                         std::set<unsigned long long> &before) {
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

void CheckSubsets(int n, int k, int id, const matrix &code_gen_matrix) {
    std::vector<int> row_starts, row_ends;
    for (auto row : code_gen_matrix) {
        row_starts.push_back(std::find(row.begin(), row.end(), 1) - row.begin());
        int one_from_end = std::find(row.rbegin(), row.rend(), 1) - row.rbegin();
        row_ends.push_back(n - one_from_end);
    }
    for (int st = 0; st < n; st++) {
        for (int fin = st + 1; fin <= n; fin++) {
            auto trellis = CreateCodeTrellisFromGenMatrix(code_gen_matrix);
            // generate all compound branches
            auto subsets = dig_trellis_pos(st, fin, trellis);
            // collect different branches
            std::set<std::set<unsigned long long>> cosets;
            for (auto &fins : subsets) {
                for (auto &starts: fins) {
                    if (!starts.empty()) {
                        cosets.insert(starts);
                    }
                }
            }
            // check all branches have same number of words
            auto size = cosets.begin()->size();
            for (auto &coset : cosets) {
                if (coset.size() != size) {
                    std::cerr << "DIFFERENT COSET SIZES\n";
                }
            }
            // check all words are only encountered once
            std::set<unsigned long long> all_words;
            for (auto &coset : cosets) {
                for (auto word : coset) {
                    all_words.insert(word);
                }
            }
            if (all_words.size() != size * cosets.size()) {
                std::cerr << "DUBLICATE ITEMS IN COSETS\n";
            }
            // find active rows in MSF to collect sets and subsets base vectors
            unsigned active_rows = 0;
            matrix check_matrix;
            for (int row = 0; row < k; row++) {
                if (row_starts[row] >= st && row_ends[row] <= fin) {
                    active_rows++;
                    check_matrix.emplace_back(code_gen_matrix[row].begin() + st, code_gen_matrix[row].begin() + fin);
                }
            }
            for (int row = 0; row < k; row++) {
                if (!(row_starts[row] >= st && row_ends[row] <= fin)) {
                    check_matrix.emplace_back(code_gen_matrix[row].begin() + st, code_gen_matrix[row].begin() + fin);
                }
            }
            // also check prognosed set size values
            if (size != (1ULL << active_rows)) {
                std::cerr << "COSET SIZE DIFFERENT FROM ESTIMATE\n";
            }
            // also check that words from trellis are equal to words from matrix
            if (all_words != gen_all_codewords(check_matrix)) {
                std::cerr << "STRANGE ITEMS IN COSETS\n";
            }
            // find active rows for basis
            matrix check_copy = check_matrix;
            std::vector<int> cols;
            GetSystematicLikeMatrix(check_matrix, cols);
            for(int i = 0; i < check_matrix.size(); i++){
                if(std::find(check_matrix[i].begin(), check_matrix[i].end(), 1) != check_matrix[i].end()){
                    check_matrix[i] = check_copy[i];
                }
            }
            matrix set_matrix(check_matrix.begin(), check_matrix.begin() + active_rows);
            matrix coset_matrix(check_matrix.begin() + active_rows, check_matrix.end());

            auto words_set = gen_all_codewords(set_matrix);
            auto words_coset = gen_all_codewords(coset_matrix);
            // check words_set are set base and words_coset are cosets base
            if (cosets.size() != words_coset.size()){
                std::cerr << "WORDS COSET SIZE NOT EQUAL TO COSETS COUNT\n";
            }
            for(auto coset_word : words_coset){
                std::set<unsigned long long> modified_words;
                for(auto set_word : words_set){
                    modified_words.insert(set_word ^ coset_word);
                }
                if(cosets.count(modified_words) == 0){
                    std::cerr << "DEDUCED SET NOT FOUND IN SETS\n";
                }
            }
        }
    }
}

void RunRandomTestsSingleThread(std::atomic_int &id_atomic, int max_id) {
    // test minspan
    std::random_device rd{};
    std::mt19937 gen{rd()};
    int n = 20, k = 12;
    matrix code_gen_matrix(k, std::vector<unsigned char>(n));
    for (;;) {
        int id = id_atomic.fetch_add(1);
        if (id % 100 == 0)
            std::cout << id << "\n";
        if (id > max_id)
            break;
        CheckVectorToCode(gen, id);
        std::set<unsigned long long> before;
        GenerateRandomCode(gen, n, k, id, code_gen_matrix, before);
        CheckCheckMatrix(n, k, code_gen_matrix);
        CheckViterbiDecoder(gen, n, k, id, code_gen_matrix, before);
        CheckSubsets(n, k, id, code_gen_matrix);
    }
}

void RunRandomTests(int threads_count = 6, int tests_count = 100000) {
    std::vector<std::thread> threads(threads_count);
    std::atomic_int id_atomic = 0;
    for (auto &thr : threads) {
        thr = std::thread([&id_atomic, tests_count]() { RunRandomTestsSingleThread(id_atomic, 100000); });
    }
    for (auto &thr : threads) {
        thr.join();
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
    std::map<int, std::map<std::string, int>> reed_muller_measures;
    std::set<std::string> reed_muller_names;
    std::mutex reed_muller_mutex;
    int check_iter = 10000000;

    static float CalculateAWGNSigmaForTargetSNR(double target_snr_db, int n, int k) {
//        10 * log10(1. / n0 * n / k) == target_snr_db;
        double eb_div_n0 = pow(10, target_snr_db / 10);
        double n0 = 1. * n / k / eb_div_n0;
        return (float) sqrt(n0 / 2);
    }

    struct SingleThreadReedMullerCheck {
        std::pair<int, std::string> Run() {
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
            float sigma = CalculateAWGNSigmaForTargetSNR(.1 * snr_db_10, n, k);
            AWGNChannel channel(sigma);
            std::string name = "RM(" + std::to_string(n) + "," + std::to_string(k) + ")";
            int errors = 0;
            for (int total = 0; total < iter; total++) {
                unsigned val = rand_gen();
                for (unsigned i = 0; i < k; i++) {
                    input[i] = (val >> i) & 1U;
                }
                encoder.encode(input, encoded);
                channel.transmit(encoded, transmitted);
                decoder.DecodeInputToCodeword(transmitted, codeword);
                if (encoded != codeword)
                    errors++;
            }
            return {errors, name};
        }

        int code_r, code_m, iter, snr_db_10;
    };

    std::vector<SingleThreadReedMullerCheck> tasks;

    void RunTasks() {
        std::unique_lock lock(reed_muller_mutex);
        for (;;) {
            if (tasks.empty())
                return;
            SingleThreadReedMullerCheck task = tasks.back();
            tasks.pop_back();

            lock.unlock();
            auto[correct, name] = task.Run();
            lock.lock();

            reed_muller_measures[task.snr_db_10][name] = correct;
            reed_muller_names.insert(std::move(name));
        }
    }

    void MultiThreadedReedMullerCheck(int thread_num) {
        for (int r = 1; r <= 2; r++) {
            for (int m = 3; m <= 5; m++) {
                for (int snr_db_10 = -100; snr_db_10 <= 100; snr_db_10 += 5) {
                    tasks.push_back(SingleThreadReedMullerCheck{r, m, check_iter, snr_db_10});
                }
            }
        }
        std::vector<std::thread> threads;
        threads.reserve(thread_num);
        for (int i = 0; i < thread_num; i++) {
            threads.emplace_back([&]() {
                RunTasks();
            });
        }
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
            std::cout << snr;
            for (auto &name: reed_muller_names) {
                std::cout << ";";
                if (auto it = measures.find(name); it != measures.end()) {
                    std::cout << it->second;
                }
            }
            std::cout << "\n";
        }
    }
};

struct RecursiveTrellisDecoding {
    explicit RecursiveTrellisDecoding(const matrix &code) : code(code) {
        k = code.size();
        n = code[0].size();
        check = GenerateTransposedCheckMatrix(code, n, k);
        auto words = gen_all_codewords(code);
        codewords.assign(words.begin(), words.end());
        PrepareDecoderRec(0, n);
    }

    void Decode(const std::vector<float> &data, std::vector<unsigned char> &codeword) {}

private:
    unsigned long long GetMask(unsigned from, unsigned to) {
        unsigned long long res = 0;
        for (auto i = from; i < to; i++) {
            res |= (1ULL << i);
        }
        return res;
    }

    void PrepareDecoderRec(int from, int to) {
        if (to - from <= 2) {
            PrepareMakeCBT(from, to);
        } else {
            int mid = (from + to) / 2;
            PrepareDecoderRec(from, mid);
            PrepareDecoderRec(mid, to);
            PrepareCombCBT(from, mid, to);
        }
    }

    void DecodeRec(int from, int to) {
        if (to - from <= 2) {
            MakeCBT(from, to);
        } else {
            int mid = (from + to) / 2;
            DecodeRec(from, mid);
            DecodeRec(mid, to);
            CombCBT(from, mid, to);
        }
    }

    void PrepareMakeCBT(int from, int to) {
        auto mask = GetMask(from, to);
        std::map<unsigned long long, unsigned long long> items;
        std::vector<unsigned long long> zeros;
        for (auto word : codewords) {
            if ((word & mask) == word) {
                items[word] = word;
                zeros.push_back(word);
            } else {
                items.insert({word & mask, word});
            }
        }
    }

    void PrepareCombCBT(int from, int mid, int to) {}

    void MakeCBT(int from, int to) {

    }

    void CombCBT(int from, int mid, int to) {}

    void CreateShortenedCode(int x, int y) {
        matrix temp = code;
    }

    matrix code, check;
    std::vector<unsigned long long> codewords;
    int n, k;
};

void checker() {
}

int main() {
    RunRandomTests();
    return 0;
    ReedMullerChecker checker;
    checker.MultiThreadedReedMullerCheck(10);
    checker.PrintDataAsCSV();
    return 0;
}
