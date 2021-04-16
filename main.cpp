#include "testing.h"

#include "base.h"
#include "msf.h"
#include "viterbi.h"
#include "reedmuller.h"
#include "polar_code.h"

#include <iomanip>
#include <thread>
#include <future>
#include <atomic>
#include <array>
#include <cassert>

struct TrellisEdge {
    int from;
    int to;
    int label_id;

    bool operator<(const TrellisEdge& other) const {
        return std::make_tuple(from, to, label_id) < std::make_tuple(other.from, other.to, other.label_id);
    }

    bool operator==(const TrellisEdge& other) const {
        return std::make_tuple(from, to, label_id) == std::make_tuple(other.from, other.to, other.label_id);
    }
};

struct TrellisCompoundBranchRule {
    int first_half;
    int second_half;
    int result;

    bool operator<(const TrellisCompoundBranchRule& other) const {
        return std::make_tuple(first_half, second_half, result) < std::make_tuple(other.first_half, other.second_half, other.result);
    }

    bool operator==(const TrellisCompoundBranchRule& other) const {
        return std::make_tuple(first_half, second_half, result) == std::make_tuple(other.first_half, other.second_half, other.result);
    }
};

struct TrellisEdgeLabel {
    unsigned long long current_label;
    float current_value;
};

using LabelCollection = std::vector<std::vector<std::vector<TrellisEdgeLabel>>>;

using RuleCollection = std::vector<std::vector<std::vector<TrellisCompoundBranchRule>>>;

void SolveLinearSystem(const matrix& coeff, matrix& temp, const std::vector<unsigned char>& results, std::vector<unsigned char>& solution) {
    assert(coeff.size() == results.size());
    solution.assign(coeff[0].size(), 2);

    temp.resize(coeff.size());
    for (int i = 0; i < coeff.size(); i++) {
        temp[i].resize(coeff[0].size() + 1);
        std::copy(coeff[i].begin(), coeff[i].end(), temp[i].begin());
        temp[i].back() = results[i];
    }
    int fixed_rows = 0;
    for (int i = 0; i < coeff[0].size(); i++) {
        for (int j = fixed_rows; j < coeff.size(); j++) {
            if (temp[j][i]) {
                swap(temp[fixed_rows], temp[j]);
                for (int k = fixed_rows + 1; k < coeff.size(); k++) {
                    if (temp[k][i]) {
                        XorVectors(temp[k], temp[fixed_rows]);
                    }
                }
                fixed_rows++;
                break;
            }
        }
    }
    for (int j = fixed_rows; j < coeff.size(); j++) {
        assert(temp[j].back() == 0);
    }
    for (int i = (int)coeff[0].size() - 1; i >= 0; i--) {
        for (int j = (int)coeff.size() - 1; j >= 0; j--) {
            if (temp[j][i]) {
                for (int ii = i - 1; ii >= 0; ii--) {
                    assert(temp[j][ii] == 0);
                }
                for (int k = j - 1; k >= 0; k--) {
                    if (temp[k][i]) {
                        XorVectors(temp[k], temp[j]);
                    }
                }
                solution[i] = temp[j].back();
                break;
            }
        }
    }
    assert(std::find(solution.begin(), solution.end(), 2) == solution.end());

}

template<typename TwoVector>
void ExpandVector(TwoVector& vec, int n) {
    vec.resize(n);
    for (auto& x : vec) {
        x.resize(n + 1);
    }
}

struct RecursiveGenContext {
    RecursiveGenContext(int n) {
        ExpandVector(cbt_values_, n);
        ExpandVector(start_rule_parts_, n);
        ExpandVector(new_rec_rules_, n);
        ExpandVector(predicted_mid_, n);
        ExpandVector(predicted_diff_, n);
    }

    struct CBTData {
        unsigned long long label;
        float value;
    };

    std::vector<std::vector<std::vector<CBTData>>> cbt_values_;
    std::vector<std::vector<std::vector<unsigned long long>>> start_rule_parts_;
    std::vector<std::vector<unsigned long long>> predicted_diff_;
    std::vector<std::vector<int>> predicted_mid_;
    RuleCollection new_rec_rules_;

    void PredictComputationDifficulty(int n, const matrix& code_gen_matrix) {
        for (int st = n - 1; st >= 0; st--) {
            for (int fin = st + 1; fin <= n; fin++) {
                matrix special_matrix;
                std::array<unsigned, 4> active_rows;
                CreateSpecialMatrix(code_gen_matrix, st, fin, fin, active_rows, special_matrix);

                predicted_mid_[st][fin] = -1;
                predicted_diff_[st][fin] = std::numeric_limits<unsigned long long>::max() / 4;
                int l = fin - st;
                if (special_matrix.size() < 60) {
                    // MakeCBT-I
                    //predicted_diff_[st][fin] = (fin - st) * (1ULL << special_matrix.size()) - (1ULL << active_rows[3]);
                    // MakeCBT-G
                    auto adds = (1ULL << (l - 1)) + l - 2;
                    auto muls = (1ULL << special_matrix.size()) / 2 - (1ULL << active_rows[3]);
                    predicted_diff_[st][fin] = adds + muls;
                }

                for (int mid = st + 1; mid < fin; mid++) {
                    CreateSpecialMatrix(code_gen_matrix, st, fin, mid, active_rows, special_matrix);
                    auto possible_diff = predicted_diff_[st][mid] + predicted_diff_[mid][fin];
                    possible_diff += (2ULL << (active_rows[2] + active_rows[3])) - (1ULL << active_rows[3]);
                    if (possible_diff < predicted_diff_[st][fin]) {
                        predicted_diff_[st][fin] = possible_diff;
                        predicted_mid_[st][fin] = mid;
                    }
                }
            }
        }
    }

    void CompareGenNew(int st, int fin, const matrix& code_gen_matrix) {
        unsigned label_size = fin - st;
        // generate all compound branches
        if (predicted_mid_[st][fin] == -1) {
            matrix special_matrix;
            std::array<unsigned, 4> active_rows;
            CreateSpecialMatrix(code_gen_matrix, st, fin, fin, active_rows, special_matrix);
            start_rule_parts_[st][fin].assign(1UL << label_size, -1);
            cbt_values_[st][fin].resize(1UL << active_rows[3]);
            std::vector<unsigned char> mask, word;
            for (unsigned long long w_mask = 0; w_mask < (1ULL << special_matrix.size()); w_mask++) {
                code_to_vector(w_mask, special_matrix.size(), mask);
                std::reverse(mask.begin(), mask.end());
                MultiplyVectorByMatrix(mask, special_matrix, word);
                auto group = w_mask % (1ULL << active_rows[3]);
                start_rule_parts_[st][fin][vector_to_code(word)] = group;
            }
            special_matrices_[st * 1000 + fin] = std::move(special_matrix);
            active_rows_[st * 1000 + fin] = active_rows;
            return;
        }
        int mid = predicted_mid_[st][fin];
        CompareGenNew(st, mid, code_gen_matrix);
        CompareGenNew(mid, fin, code_gen_matrix);

        matrix special_matrix;
        std::array<unsigned, 4> active_rows;
        CreateSpecialMatrix(code_gen_matrix, st, fin, mid, active_rows, special_matrix);

        matrix solutions_l, solutions_r;
        matrix eq_full_l, eq_full_r;
        {
            matrix trans_l = TransposeMatrix(special_matrices_[st * 1000 + mid]);
            matrix trans_r = TransposeMatrix(special_matrices_[mid * 1000 + fin]);
            matrix temp;
            std::vector<unsigned char> solution;
            std::vector<unsigned char> res;
            for (unsigned index = active_rows[0] + active_rows[1]; index < special_matrix.size(); index++) {
                auto& row = special_matrix[index];
                res.assign(row.begin(), row.begin() + (mid - st));
                eq_full_l.push_back(res);
                SolveLinearSystem(trans_l, temp, res, solution);
                solutions_l.push_back(solution);
                res.assign(row.begin() + (mid - st), row.end());
                eq_full_r.push_back(res);
                SolveLinearSystem(trans_r, temp, res, solution);
                solutions_r.push_back(solution);
            }
        }
        unsigned w_pow = active_rows[2] + active_rows[3];
        std::vector<unsigned char> mask(w_pow), res_l, res_r;
        unsigned rows_l = active_rows_[st * 1000 + mid][3];
        unsigned rows_r = active_rows_[mid * 1000 + fin][3];
        cbt_values_[st][fin].resize(1ULL << active_rows[3]);
        std::set<TrellisCompoundBranchRule> cbt_rules_3;
        for (unsigned long long w_mask = 0; w_mask < (1ULL << w_pow); w_mask++) {
            code_to_vector(w_mask, mask);
            std::reverse(mask.begin(), mask.end());
            MultiplyVectorByMatrix(mask, solutions_l, res_l);
            MultiplyVectorByMatrix(mask, solutions_r, res_r);

            std::reverse(res_l.begin(), res_l.end());
            std::reverse(res_r.begin(), res_r.end());
            unsigned long long index_l = vector_to_code(res_l.begin(), res_l.begin() + rows_l);
            unsigned long long index_r = vector_to_code(res_r.begin(), res_r.begin() + rows_r);
            TrellisCompoundBranchRule res{ index_l, index_r, w_mask % (1ULL << active_rows[3]) };
            bool inserted = cbt_rules_3.insert(res).second;
        }
        new_rec_rules_[st][fin].assign(cbt_rules_3.begin(), cbt_rules_3.end());

        special_matrices_[st * 1000 + fin] = std::move(special_matrix);
        active_rows_[st * 1000 + fin] = active_rows;
    }

    void CreateSpecialMatrix(const matrix& code_gen_matrix, int st, int fin, int mid, std::array<unsigned, 4>& parts_sizes, matrix& special_matrix) {

        int n = code_gen_matrix[0].size();
        matrix parts[4];
        for (auto& row : code_gen_matrix) {
            int first_active = std::find(row.begin(), row.end(), 1) - row.begin();
            int last_active = n - (std::find(row.rbegin(), row.rend(), 1) - row.rbegin());
            if (first_active < st || last_active > fin) {
                parts[3].emplace_back(row.begin() + st, row.begin() + fin);
            }
            else if (first_active >= mid) {
                parts[0].emplace_back(row.begin() + st, row.begin() + fin);
            }
            else if (last_active <= mid) {
                parts[1].emplace_back(row.begin() + st, row.begin() + fin);
            }
            else if (first_active != row.size()) {
                parts[2].emplace_back(row.begin() + st, row.begin() + fin);
            }
        }
        for (unsigned i = 0; i < 4; i++) {
            parts_sizes[i] = parts[i].size();
            for (auto& row : parts[i]) {
                special_matrix.push_back(std::move(row));
            }
        }

        unsigned active_rows = parts_sizes[0] + parts_sizes[1] + parts_sizes[2];
        matrix special_copy = special_matrix;
        std::vector<int> cols;
        GetSystematicLikeMatrix(special_copy, cols);
        int special_rows = 0;
        for (int i = 0; i < special_copy.size(); i++) {
            if (std::find(special_copy[i].begin(), special_copy[i].end(), 1) != special_copy[i].end()) {
                special_matrix[special_rows++] = std::move(special_matrix[i]);
            }
        }
        parts_sizes[3] = special_rows - active_rows;
        special_copy.clear();
        special_matrix.resize(special_rows);
    }

    std::map<int, matrix> special_matrices_;
    std::map<int, std::array<unsigned, 4>> active_rows_;

};
unsigned long long rec_comps_2 = 0, rec_adds_2 = 0;

unsigned long long
Decode(int n, RecursiveGenContext& ctx, const std::vector<float>& data) {
    std::vector<bool> set_groups;
    for (int st = n - 1; st >= 0; st--) {
        for (int fin = st + 1; fin <= n; fin++) {
            auto& labels = ctx.cbt_values_[st][fin];
            if (labels.empty())
                continue;
            auto& starts = ctx.start_rule_parts_[st][fin];
            int mid = ctx.predicted_mid_[st][fin];
            if (mid == -1) {
                if (fin - st == 1) {
                    unsigned wcur = 0;
                    float prob = -data[st];
                    if (prob < 0) {
                        prob = -prob;
                        wcur = 1;
                    }
                    labels[starts[wcur]].label = wcur;
                    labels[starts[wcur]].value = prob;
                    if (starts[wcur] != starts[wcur ^ 1]) {
                        wcur ^= 1;
                        labels[starts[wcur]].label = wcur;
                        labels[starts[wcur]].value = -prob;
                    }
                    continue;
                }
                rec_adds_2 += (fin - st - 1);
                float prob2 = -std::accumulate(data.begin() + st + 1, data.begin() + fin, data[st]);

                set_groups.assign(labels.size(), true);
                unsigned inverse_mask = starts.size() - 1;
                for (unsigned long long word = 0; word < starts.size() / 2; word++) {
                    auto wcur = word ^ (word >> 1);
                    if (wcur) {
                        auto prev = (word - 1) ^ ((word - 1) >> 1);
                        auto diff = wcur ^ prev;
                        auto ww = wcur;
                        int difx = 0;
                        for (int i = st; i < fin; i++, ww /= 2, diff /= 2) {
                            if (diff & 1) {
                                rec_adds_2++;
                                difx++;
                                if (ww & 1) {
                                    prob2 += data[i] * 2;
                                }
                                else {
                                    prob2 -= data[i] * 2;
                                }
                            }
                        }
                        if (difx != 1) {
                            std::cerr << "WTF\n";
                        }
                    }
                    if (starts[wcur] == -1) {
                        continue;
                    }
                    float prob = prob2;
                    if (prob < 0) {
                        prob = -prob;
                        wcur ^= inverse_mask;
                    }
                    bool cmp_result;
                    if (set_groups[starts[wcur]]) {
                        set_groups[starts[wcur]] = false;
                        labels[starts[wcur]].label = wcur;
                        labels[starts[wcur]].value = prob;
                    }
                    else {
                        rec_comps_2++;
                        if (labels[starts[wcur]].value < prob) {
                            labels[starts[wcur]].label = wcur;
                            labels[starts[wcur]].value = prob;
                        }
                    }
                    if (starts[wcur] == starts[wcur ^ inverse_mask])
                        continue;
                    wcur ^= inverse_mask;
                    prob = -prob;
                    if (set_groups[starts[wcur]]) {
                        set_groups[starts[wcur]] = false;
                        labels[starts[wcur]].label = wcur;
                        labels[starts[wcur]].value = prob;
                    }
                    else {
                        rec_comps_2++;
                        if (labels[starts[wcur]].value < prob) {
                            labels[starts[wcur]].label = wcur;
                            labels[starts[wcur]].value = prob;
                        }
                    }
                }
            }
            else {
                auto& labels_1 = ctx.cbt_values_[st][mid];
                auto& labels_2 = ctx.cbt_values_[mid][fin];
                set_groups.assign(labels.size(), true);
                for (const auto& rule : ctx.new_rec_rules_[st][fin]) {
                    rec_adds_2++;
                    auto link_val = labels_1[rule.first_half].value + labels_2[rule.second_half].value;
                    auto new_label = labels_1[rule.first_half].label +
                        (labels_2[rule.second_half].label << (mid - st));
                    if (set_groups[rule.result]) {
                        set_groups[rule.result] = false;
                        labels[rule.result].value = link_val;
                        labels[rule.result].label = new_label;
                    }
                    else {
                        rec_comps_2++;
                        if (link_val > labels[rule.result].value) {
                            labels[rule.result].value = link_val;
                            labels[rule.result].label = new_label;
                        }
                    }
                }
            }
        }
    }
    return ctx.cbt_values_[0][n][0].label;
}

unsigned long long rec_comps = 0, rec_adds = 0;

void CheckRecursiveDecoder(std::mt19937& gen, int n, int k, int id, const matrix& code_gen_matrix, std::ostream& out) {
    auto start0 = std::chrono::high_resolution_clock::now();
    RecursiveGenContext ctx2(n);
    ctx2.PredictComputationDifficulty(n, code_gen_matrix);
    out << "Predicting minimal " << ctx2.predicted_diff_[0][n] << " ops\n";

    ctx2.CompareGenNew(0, n, code_gen_matrix);
    auto end0 = std::chrono::high_resolution_clock::now();
    out << "Recurse creation " << std::chrono::duration_cast<std::chrono::duration<double>>(end0 - start0).count() << " s\n";


    std::vector<unsigned char> input(k, 0), encoded;
    SimpleEncoder enc(code_gen_matrix);
    AWGNChannel channel(0.0001);
    std::vector<std::vector<float>> transmits(decode_count);
    std::vector<unsigned long long> codewords(decode_count);
    for (int i = 0; i < decode_count; i++) {
        for (auto& bit : input) {
            bit = gen() % 2;
        }
        enc.encode(input, encoded);
        channel.transmit(encoded, transmits[i]);
        codewords[i] = vector_to_code(encoded);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < decode_count; i++) {
        auto res2 = Decode(n, ctx2, transmits[i]);
        if (res2 != codewords[i]) {
            std::cerr << "INCORRECT NEW RECURSIVE DECODE IN " << id << "\n";
            auto res3 = Decode(n, ctx2, transmits[i]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    out << "Recurse " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << " s\n";
}

void RunRandomTestsSingleThread(std::atomic_int& id_atomic, int max_id) {
    // test minspan
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    int n = 24, k = 12;
    matrix code_gen_matrix(k, std::vector<unsigned char>(n));
    std::ostringstream out;
    for (;;) {
        vit_adds = vit_comps = rec_adds = rec_comps = rec_adds_2 = rec_comps_2 = 0;
        int id = id_atomic.fetch_add(1);
        if (id % 1 == 0)
            out << id << "\n";
        if (id >= max_id)
            break;
        std::set<unsigned long long> before;
        GenerateRandomCode(gen, n, k, id, code_gen_matrix, before);
        CheckVectorToCode(gen, id);
        GenerateMinimalSpanMatrix(code_gen_matrix, n, k);
        CheckCheckMatrix(n, k, code_gen_matrix);
        CheckViterbiDecoder(gen, n, k, id, code_gen_matrix, out);
        CheckRecursiveDecoder(gen, n, k, id, code_gen_matrix, out);
        out << vit_adds / decode_count << "\t" << vit_comps / decode_count << "\t" << (vit_adds + vit_comps) / decode_count << "\n";
        out << rec_adds / decode_count << "\t" << rec_comps / decode_count << "\t" << (rec_adds + rec_comps) / decode_count << "\n";
        out << rec_adds_2 / decode_count << "\t" << rec_comps_2 / decode_count << "\t" << (rec_adds_2 + rec_comps_2) / decode_count << "\n\n";
        std::cout << out.str();
        out.clear();
        //CheckSubsets(n, k, id, code_gen_matrix);
    }
}

void RunRandomTests(int threads_count = 1, int tests_count = 100000) {
    if (threads_count == 1) {
        std::atomic_int id_atomic = 0;
        RunRandomTestsSingleThread(id_atomic, tests_count);
    }
    else {
        std::vector<std::thread> threads(threads_count);
        std::atomic_int id_atomic = 0;
        for (auto& thr : threads) {
            thr = std::thread([&id_atomic, tests_count]() { RunRandomTestsSingleThread(id_atomic, 100000); });
        }
        for (auto& thr : threads) {
            thr.join();
        }
    }
}

struct RecursiveTrellisDecoding {
    explicit RecursiveTrellisDecoding(const matrix& code) : code(code) {
        k = code.size();
        n = code[0].size();
        check = GenerateTransposedCheckMatrix(code, n, k);
        auto words = gen_all_codewords(code);
        codewords.assign(words.begin(), words.end());
        PrepareDecoderRec(0, n);
    }

    void Decode(const std::vector<float>& data, std::vector<unsigned char>& codeword) {}

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
        }
        else {
            int mid = (from + to) / 2;
            PrepareDecoderRec(from, mid);
            PrepareDecoderRec(mid, to);
            PrepareCombCBT(from, mid, to);
        }
    }

    void DecodeRec(int from, int to) {
        if (to - from <= 2) {
            MakeCBT(from, to);
        }
        else {
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
            }
            else {
                items.insert({ word & mask, word });
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

struct PolarRes {
    int list_size_;
    int snr_db_10_;
    int fer_;
};

int main() {
    // 8, 4, 4
    int n = 10;
    int info_length = 512;

    std::random_device rd{};
    std::mt19937 gen{ rd() };
    int iterations = 10000;
    std::mutex vec_mutex;
    std::vector<std::future<void>> tasks;
    std::map<int, std::map<int, int>> results;
    for (int snr_db_10 = -10; snr_db_10 <= 40; snr_db_10 += 5) {
        tasks.push_back(std::async([=, &results, &vec_mutex]() {
            std::random_device rd{};
            std::mt19937 gen{ rd() };
            PolarCode code(n, info_length, 0.5);
            int fer = 0;
            AWGNChannel channel = AWGNChannelFromSNR(snr_db_10 / 10.);
            std::vector<unsigned char> input(512);
            std::vector<double> transmitted;
            std::vector<std::array<double, 2>> probabilities;
            std::vector<unsigned char> codeword, decoded;
            for (int iter = 0; iter < iterations; iter++) {
                for (unsigned i = 0; i < 512; i++) {
                    input[i] = gen() & 1U;
                }
                code.Encode(input, codeword);
                channel.transmit(codeword, transmitted);
                channel.probability(transmitted, probabilities);
                bool prev_decoded = true;
                for (int list_size : {1, 2, 4, 8, 16}) {
                    code.Decode(probabilities, list_size, decoded);
                    std::lock_guard lock(vec_mutex);
                    results[list_size][snr_db_10] = results[list_size][snr_db_10];
                    bool this_decoded = decoded == input;
                    if (!this_decoded && prev_decoded) {
                        std::cerr << "Failed to decode with bigger list\n";
                    }
                    if (!this_decoded) {
                        results[list_size][snr_db_10]++;
                    }
                    prev_decoded = this_decoded;
                }
            }
            }));
    }
    for (auto& task : tasks) {
        task.wait();
    }

    for (auto& [list, result] : results) {
        for (auto& [snr, fer] : result) {
            std::cout << list << "\t" << 0.1 * snr << "\t" << 1. * (fer) / iterations << "\n";
        }
        std::cout << "\n\n";
    }

    return 0;
    matrix code_gen_matrix;
    std::vector<std::pair<int, int>> codes = { {3,1}, {3,2}, {4,2}, {5,2}, {5,3}, {6,2}, { 6, 3 }, {6,4}
    };
    for (auto [m, r] : codes) {
        vit_adds = vit_comps = rec_adds = rec_comps = rec_adds_2 = rec_comps_2 = 0;
        CheckVectorToCode(gen, 0);
        std::set<unsigned long long> before;
        code_gen_matrix = GenerateReedMullerCode(r, m);
        int n = code_gen_matrix[0].size();
        int k = code_gen_matrix.size();
        GenerateMinimalSpanMatrix(code_gen_matrix, n, k);
        CheckCheckMatrix(n, k, code_gen_matrix);
        CheckViterbiDecoder(gen, n, k, 0, code_gen_matrix, std::cout);
        CheckRecursiveDecoder(gen, n, k, 0, code_gen_matrix, std::cout);
        std::cout << "RM(" << r << "," << m << ") " << n << " " << k << "\n";
        std::cout << vit_adds / decode_count << "\t" << vit_comps / decode_count << "\t" << (vit_adds + vit_comps) / decode_count << "\n";
        std::cout << rec_adds / decode_count << "\t" << rec_comps / decode_count << "\t" << (rec_adds + rec_comps) / decode_count << "\n";
        std::cout << rec_adds_2 / decode_count << "\t" << rec_comps_2 / decode_count << "\t" << (rec_adds_2 + rec_comps_2) / decode_count << "\n\n";
    }
    RunRandomTests(4, 100000);
    return 0;
    ReedMullerChecker checker;
    checker.MultiThreadedReedMullerCheck(10);
    checker.PrintDataAsCSV();
    return 0;
}
