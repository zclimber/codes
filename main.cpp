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

int max_runs = 1000;
int report_runs = 100;
std::vector<std::vector<double>> get_bler_quick(int n, int info_size, double design_epsilon, const std::vector<double>& ebno_vec,
    const std::vector<int>& list_size_vec) {

    std::vector<PolarCode> codes;
    for (auto x : list_size_vec) {
        codes.push_back(PolarCode(n, info_size, design_epsilon));
        //codes.back().InitInnerTrellisDecoder();
    }
    int block_size = 1 << n;
    int max_err = 1000;

    std::vector<std::vector<double>> bler(list_size_vec.size(), std::vector<double>(ebno_vec.size(), 0));
    std::vector<std::vector<int>> num_err(list_size_vec.size(), std::vector<int>(ebno_vec.size(), 0));
    std::vector<std::vector<int>> num_run(list_size_vec.size(), std::vector<int>(ebno_vec.size(), 0));

    std::vector<uint8_t> info_bits(info_size);
    std::vector<uint8_t> coded_bits;
    std::vector<double> transitted(block_size, 0);
    std::vector<std::array<double, 2>> probabilities(block_size);
    std::vector<double> llrs(block_size);
    std::vector<double> llrs_calc(block_size);
    std::vector<uint8_t> decoded_info_bits;

    std::mt19937 gen{};

    auto start = std::chrono::high_resolution_clock::now();

    int total_runs = 0;
    std::cout << "Using " << max_runs << " iterations per item" << std::endl;
    for (int ebno_i = 0; ebno_i < ebno_vec.size(); ebno_i++) {
        auto ebno = ebno_vec[ebno_i];
        auto channel = AWGNChannelFromEBN0(ebno, block_size, info_size);
        for (int run = 1; run <= max_runs; run++) {
            total_runs++;
            if ((total_runs % report_runs) == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
                std::cout << "Running iteration " << total_runs << "; time elapsed = " << duration << " seconds" << std::endl;
            }
            for (auto& bit : info_bits)
                bit = gen() % 2;

            codes[0].Encode(info_bits, coded_bits);

            channel.transmit(coded_bits, transitted);
            channel.probability(transitted, probabilities);
            channel.llr(transitted, llrs);
            for (int i = 0; i < block_size; i++) {
                llrs_calc[i] = std::log(probabilities[i][0] / probabilities[i][1]);
            }

            for (int list_i = 0; list_i < list_size_vec.size(); list_i++) {
                auto list_size = list_size_vec[list_i];
#ifdef LLR
                codes[list_i].Decode(llrs, list_size, decoded_info_bits);
#else
                codes[list_i].Decode(probabilities, list_size, decoded_info_bits);
#endif
                num_run[list_i][ebno_i]++;
                if (info_bits != decoded_info_bits)
                    num_err[list_i][ebno_i]++;
            }
        }
    }
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
    std::cout << "Run simulation in " << duration << " seconds" << std::endl;

    for (int list_i = 0; list_i < list_size_vec.size(); list_i++) {
        for (int ebno_i = 0; ebno_i < ebno_vec.size(); ebno_i++) {
            bler[list_i][ebno_i] = (double)num_err[list_i][ebno_i] / num_run[list_i][ebno_i];
        }
    }
    return bler;
}

void run_get_bler() {
    int n = 11;
    int info_length = 1 << (n - 1);

    double design_epsilon = 0.5;

    double ebno_log_min = 1.00;
    double ebno_log_max = 2.01;
    double ebno_log_increment = 0.25;
    std::vector<double> ebno_vec;

    for (double ebno_log = ebno_log_min; ebno_log <= ebno_log_max; ebno_log += ebno_log_increment)
        ebno_vec.push_back(ebno_log);

    std::vector<int> list_size_vec{ 1, 2, 4, 8, 32 };

    auto bler = get_bler_quick(n, info_length, design_epsilon, ebno_vec, list_size_vec);

    std::cout << "Done\n";

    for (int ebno_i = 0; ebno_i < ebno_vec.size(); ebno_i++) {
        std::cout << std::fixed << std::setprecision(3) << ebno_vec[ebno_i] << "\t \t";
        for (int list_i = 0; list_i < list_size_vec.size(); list_i++) {
            std::cout << std::fixed << std::setprecision(6) << bler[list_i][ebno_i] << "\t";
        }
        std::cout << std::endl;
    }
}

int main() {
    //run_get_bler();
    //for (int snr_db_10 = -10; snr_db_10 <= 40; snr_db_10 += 5) {
    //    AWGNChannel channel = AWGNChannelFromSNR(snr_db_10 / 10.);
    //    std::cout << snr_db_10 / 10. << ": " << channel.sigma() << "\n";
    //}
    //return 0;

    int mini_info = 5;
    std::vector<unsigned char> info(8), coded;
    matrix polar_code_base;
    PolarCode code(3, mini_info, .5);
    for (int i = 0; i < mini_info; i++) {
        info.assign(mini_info, 0);
        info[i] = 1;
        code.Encode(info, coded);
        polar_code_base.push_back(coded);
        for (auto b : coded) {
            std::cout << int(b) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    ViterbiSoftDecoder vsd(polar_code_base);
    for (int i = 0; i < mini_info; i++) {
        info.assign(mini_info, 0);
        info[i] = 1;
        vsd.Encode(info, coded);
        for (auto b : coded) {
            std::cout << int(b) << " ";
        }
        std::cout << "\n";
    }

    //for(auto i : code.info_channels_)
    //std::cout << i << " ";
    //std::cout << "\n";

    std::vector<uint8_t> in(mini_info), codeword, out_polar, out_id;
    std::vector<float> transmitted;
    std::vector<double> llrs;
    std::random_device rd{};
    std::mt19937 gen{ rd() };

    for (int iter = 1;; iter++) {
        if (iter % 1000 == 0) {
            std::cout << iter << "\n";
        }
        for (auto& bit : in) {
            bit = 0;
        }
        code.Encode(in, codeword);
        AWGNChannel channel = AWGNChannelFromSNR(1.);
        channel.transmit(codeword, transmitted);
        //channel.probability(transmitted, probabilities);
        channel.llr(transmitted, llrs);
        code.Decode(llrs, 32, out_polar);
        vsd.DecodeInputToCodeword(transmitted, out_id);
        if (out_id == codeword && out_polar != in) {
            std::cerr << "Worse than ML decoder\n";
        }
    }

    return 0;
    // 8, 4, 4
    int n = 10;
    int info_length = 512;

    //std::random_device rd{};
    //std::mt19937 gen{ rd() };
    int iterations = 1000;
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
            std::vector<double> llrs;
            std::vector<std::array<double, 2>> probabilities;
            std::vector<unsigned char> codeword, decoded;
            for (int iter = 0; iter < iterations; iter++) {
                for (unsigned i = 0; i < 512; i++) {
                    input[i] = gen() & 1U;
                }
                code.Encode(input, codeword);
                channel.transmit(codeword, transmitted);
                channel.probability(transmitted, probabilities);
                channel.llr(transmitted, llrs);
                bool prev_decoded = false;
                for (int list_size : {1, 2, 4, 8, 16}) {
                    code.Decode(llrs, list_size, decoded);
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
