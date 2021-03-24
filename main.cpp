#include "testing.h"

#include "base.h"
#include "msf.h"
#include "viterbi.h"
#include "reedmuller.h"

#include <iomanip>
#include <thread>
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

template<typename T>
T pop(std::vector<T>& vector) {
    T ret = vector.back();
    vector.pop_back();
    return ret;
}

struct PolarCode {
    PolarCode(int n, int k, float noise_sigma) : n_pow_(n), n_block_(1 << n), k_(k) {
        BhattacharyyaConstruct(n_block_, k, noise_sigma);
    }

    void BhattacharyyaConstruct(int n, int k, float noise_sigma) {
        double sigma_2 = double(noise_sigma) * noise_sigma * 2;
        double z_awgn = exp(-1 / sigma_2);
        std::vector<std::pair<double, int>> single_pos(n);
        single_pos[0] = { z_awgn, 0 };
        int jump = 1;
        while (jump < n) {
            for (int k = 0; k < jump; k++) {
                double z = single_pos[k].first;
                single_pos[k] = { 2 * z - z * z, k };
                single_pos[k + jump] = { z * z, k + jump };
            }
            jump *= 2;
        }
        std::sort(single_pos.begin(), single_pos.end()); // K first are data bits, N-K last are frozen bits
        is_info_bit_.resize(n, false);
        for (size_t info_bit = 0; info_bit < k; info_bit++) {
            is_info_bit_[single_pos[info_bit].second] = true;
        }
    }

    void Encode(const std::vector<unsigned char>& data, std::vector<unsigned char>& res) {
        res.resize(n_block_);

        const unsigned char* data_ptr = data.data();
        for (size_t bit = 0; bit < n_block_; bit++) {
            if (is_info_bit_[bit]) {
                res[bit] = *(data_ptr)++;
            }
            else {
                res[bit] = 0;
            }
        }

        int jump = n_block_ / 2;
        while (jump >= 1) {
            for (int i = 0; i < n_block_; i += 2 * jump) {
                for (int j = 0; j < jump; j += 1) {
                    res[i + j] = res[i + j] ^ res[i + j + jump];
                }
            }
            jump /= 2;
        }
    }

    void Decode(const std::vector<unsigned char>& sent, std::vector<unsigned char>& res) {

        initializeDataStructures();

        int l = assignInitialPath();

        double* p_0 = getArrayPointer_P(0, l);

        for (uint16_t beta = 0; beta < n_block_; ++beta) {
            p_0[2 * beta] = sent[beta] == 0 ? 0.99 : 0.01;
            p_0[2 * beta + 1] = sent[beta] == 1 ? 0.99 : 0.01;
        }
        decode_scl(res);
    }

    void PolarCode::decode_scl(std::vector<unsigned char>& res) {

        for (uint16_t phi = 0; phi < n_block_; ++phi) {

            recursivelyCalcP(n_pow_, phi);

            if (is_info_bit_[phi])
                continuePaths_UnfrozenBit(phi);
            else {
                for (uint16_t l = 0; l < _list_size; ++l) {
                    if (_activePath.at(l) == 0)
                        continue;
                    int* c_m = getArrayPointer_C(n_pow_, l);
                    c_m[(phi % 2)] = 0; // frozen value assumed to be zero
                    //_arrayPointer_Info.at(l)[phi] = 0;
                }
            }

            if ((phi % 2) == 1)
                recursivelyUpdateC(n_pow_, phi);

        }    
        uint16_t  l_p = 0;

        double p_p1 = 0;
        for (uint16_t l = 0; l < _list_size; ++l) {

            if (_activePath.at(l) == 0)
                continue;

                int* c_m = getArrayPointer_C(n_pow_, l);
                double* p_m = getArrayPointer_P(n_pow_, l);
                if (p_p1 < p_m[c_m[1]]) {
                    l_p = l;
                    p_p1 = p_m[c_m[1]];
                }
        }

        uint8_t* c_0 = _arrayPointer_Info.at(l_p);
        res.resize(k_);
        auto res_ptr = res.begin();
        for (int i = 0; i < n_block_; i++) {
            if (is_info_bit_[i]) {
                *(res_ptr++) = c_0[i];
            }
        }
        //std::vector<uint8_t> deocded_info_bits(_info_length);
        //for (uint16_t beta = 0; beta < _info_length; ++beta)
        //    deocded_info_bits.at(beta) = c_0[_channel_order_descending.at(beta)];

        for (uint16_t s = 0; s < _list_size; ++s) {
            delete[] _arrayPointer_Info.at(s);
            for (uint16_t lambda = 0; lambda < n_pow_ + 1; ++lambda) {

                delete[] _arrayPointer_P.at(lambda).at(s);
                delete[] _arrayPointer_C.at(lambda).at(s);
            }
        }

    }

    void initializeDataStructures() {
        int _n = n_pow_;
        _inactivePathIndices.clear();
        _activePath.resize(_list_size);
        _arrayPointer_P.resize(_n + 1, std::vector<double*>(_list_size));
        _arrayPointer_C.resize(_n + 1, std::vector<int*>(_list_size));
        _arrayPointer_Info.resize(_list_size);
        _pathIndexToArrayIndex.resize(_n + 1, std::vector<int>(_list_size));

        _inactiveArrayIndices.resize(_n + 1);
        for (auto& stack : _inactiveArrayIndices) {
            stack.reserve(_list_size);
            stack.clear();
        }

        _arrayReferenceCount.assign(_n + 1, std::vector<int>(_list_size, 0));

        for (int s = 0; s < _list_size; ++s) {
            _arrayPointer_Info.at(s) = new uint8_t[n_block_]();
            for (uint16_t lambda = 0; lambda < _n + 1; ++lambda) {
                _arrayPointer_P[lambda].at(s) = new double[2 * (1 << (_n - lambda))]();
                _arrayPointer_C[lambda].at(s) = new int[2 * (1 << (_n - lambda))]();
                _inactiveArrayIndices[lambda].push_back(s);
            }
        }

        for (uint16_t l = 0; l < _list_size; ++l) {
            _activePath.at(l) = 0;
            _inactivePathIndices.push_back(l);
        }
    }

    int assignInitialPath() {
        uint16_t  l = _inactivePathIndices.back();
        _inactivePathIndices.pop_back();
        _activePath.at(l) = 1;
        // Associate arrays with path index
        for (int lambda = 0; lambda < n_pow_ + 1; ++lambda) {
            int s = pop(_inactiveArrayIndices[lambda]);
            _pathIndexToArrayIndex.at(lambda).at(l) = s;
            _arrayReferenceCount.at(lambda).at(s) = 1;
        }
        return l;
    }

    int clonePath(int l) {
        int l_p = pop(_inactivePathIndices);
        _activePath.at(l_p) = 1;

        for (int lambda = 0; lambda < n_pow_ + 1; ++lambda) {
            int s = _pathIndexToArrayIndex.at(lambda).at(l);
            _pathIndexToArrayIndex.at(lambda).at(l_p) = s;
            _arrayReferenceCount.at(lambda).at(s)++;
        }
        return l_p;
    }

    void killPath(int l) {
        _activePath.at(l) = 0;
        _inactivePathIndices.push_back(l);

        for (uint16_t lambda = 0; lambda < n_pow_ + 1; ++lambda) {
            uint16_t s = _pathIndexToArrayIndex.at(lambda).at(l);
            _arrayReferenceCount.at(lambda).at(s)--;
            if (_arrayReferenceCount.at(lambda).at(s) == 0) {
                _inactiveArrayIndices.at(lambda).push_back(s);
            }
        }
    }

    double* getArrayPointer_P(int lambda, int  l) {
        int  s = _pathIndexToArrayIndex.at(lambda).at(l);
        int s_p;
        if (_arrayReferenceCount.at(lambda).at(s) == 1) {
            s_p = s;
        }
        else {
            s_p = pop(_inactiveArrayIndices.at(lambda));

            //copy
            std::copy(_arrayPointer_P.at(lambda).at(s), _arrayPointer_P.at(lambda).at(s) + (1 << (n_pow_ - lambda + 1)), _arrayPointer_P.at(lambda).at(s_p));
            std::copy(_arrayPointer_C.at(lambda).at(s), _arrayPointer_C.at(lambda).at(s) + (1 << (n_pow_ - lambda + 1)), _arrayPointer_C.at(lambda).at(s_p));

            _arrayReferenceCount.at(lambda).at(s)--;
            _arrayReferenceCount.at(lambda).at(s_p) = 1;
            _pathIndexToArrayIndex.at(lambda).at(l) = s_p;
        }
        return _arrayPointer_P.at(lambda).at(s_p);
    }

    int* PolarCode::getArrayPointer_C(int lambda, int  l) {
        int  s = _pathIndexToArrayIndex.at(lambda).at(l);
        int s_p;
        if (_arrayReferenceCount.at(lambda).at(s) == 1) {
            s_p = s;
        }
        else {
            s_p = pop(_inactiveArrayIndices.at(lambda));

            std::copy(_arrayPointer_P.at(lambda).at(s), _arrayPointer_P.at(lambda).at(s) + (1 << (n_pow_ - lambda + 1)), _arrayPointer_P.at(lambda).at(s_p));
            std::copy(_arrayPointer_C.at(lambda).at(s), _arrayPointer_C.at(lambda).at(s) + (1 << (n_pow_ - lambda + 1)), _arrayPointer_C.at(lambda).at(s_p));

            _arrayReferenceCount.at(lambda).at(s)--;
            _arrayReferenceCount.at(lambda).at(s_p) = 1;
            _pathIndexToArrayIndex.at(lambda).at(l) = s_p;
        }
        return _arrayPointer_C.at(lambda).at(s_p);
    }

    void recursivelyCalcP(int lambda, int phi) {
        if (lambda == 0)
            return;
        int psi = phi / 2;
        if ((phi % 2) == 0)
            recursivelyCalcP(lambda - 1, psi);

        double sigma = 0.0f;
        for (int l = 0; l < _list_size; ++l) {
            if (_activePath.at(l) == 0)
                continue;
            double* p_lambda = getArrayPointer_P(lambda, l);
            double* p_lambda_1 = getArrayPointer_P(lambda - 1, l);

            int* c_lambda = getArrayPointer_C(lambda, l);
            for (int beta = 0; beta < (1 << (n_pow_ - lambda)); ++beta) {
                if ((phi % 2) == 0) {
                    p_lambda[2 * beta] = 0.5f * (p_lambda_1[2 * (2 * beta)] * p_lambda_1[2 * (2 * beta + 1)]
                        + p_lambda_1[2 * (2 * beta) + 1] * p_lambda_1[2 * (2 * beta + 1) + 1]);
                    p_lambda[2 * beta + 1] = 0.5f * (p_lambda_1[2 * (2 * beta) + 1] * p_lambda_1[2 * (2 * beta + 1)]
                        + p_lambda_1[2 * (2 * beta)] * p_lambda_1[2 * (2 * beta + 1) + 1]);
                }
                else {
                    int u_p = c_lambda[2 * beta];
                    p_lambda[2 * beta] = 0.5f * p_lambda_1[2 * (2 * beta) + (u_p % 2)] * p_lambda_1[2 * (2 * beta + 1)];
                    p_lambda[2 * beta + 1] = 0.5f * p_lambda_1[2 * (2 * beta) + ((u_p + 1) % 2)] * p_lambda_1[2 * (2 * beta + 1) + 1];
                }
                sigma = std::max(sigma, p_lambda[2 * beta]);
                sigma = std::max(sigma, p_lambda[2 * beta + 1]);


            }
        }

        for (int l = 0; l < _list_size; ++l) {
            if (sigma == 0) // Typically happens because of undeflow
                break;
            if (_activePath.at(l) == 0)
                continue;
            double* p_lambda = getArrayPointer_P(lambda, l);
            for (int beta = 0; beta < (1 << (n_pow_ - lambda)); ++beta) {
                p_lambda[2 * beta] = p_lambda[2 * beta] / sigma;
                p_lambda[2 * beta + 1] = p_lambda[2 * beta + 1] / sigma;
            }
        }
    }

    void recursivelyUpdateC(int lambda, int phi) {

        int psi = phi >> 1;
        for (int l = 0; l < _list_size; ++l) {
            if (_activePath.at(l) == 0)
                continue;
            int* c_lambda = getArrayPointer_C(lambda, l);
            int* c_lambda_1 = getArrayPointer_C(lambda - 1, l);
            for (int beta = 0; beta < (1 << (n_pow_ - lambda)); ++beta) {
                c_lambda_1[2 * (2 * beta) + (psi % 2)] = ((c_lambda[2 * beta] + c_lambda[2 * beta + 1]) % 2);
                c_lambda_1[2 * (2 * beta + 1) + (psi % 2)] = c_lambda[2 * beta + 1];
            }
        }
        if ((psi % 2) == 1)
            recursivelyUpdateC(lambda - 1, psi);

    }

    void PolarCode::continuePaths_UnfrozenBit(uint16_t phi) {

        std::vector<double> probForks((unsigned long)(2 * _list_size));
        std::vector<double> probabilities;
        std::vector<int> contForks((unsigned long)(2 * _list_size), 0);

        int i = 0;
        for (unsigned l = 0; l < _list_size; ++l) {
            if (_activePath.at(l) == 0) {
                probForks.at(2 * l) = NAN;
                probForks.at(2 * l + 1) = NAN;
            }
            else {
                double* p_m = getArrayPointer_P(n_pow_, l);
                probForks.at(2 * l) = p_m[0];
                probForks.at(2 * l + 1) = p_m[1];

                probabilities.push_back(probForks.at(2 * l));
                probabilities.push_back(probForks.at(2 * l + 1));

                i++;
            }
        }

        int rho = std::min(2 * i, _list_size);

        std::sort(probabilities.begin(), probabilities.end(), std::greater<double>());

        double threshold = probabilities.at((unsigned long)(rho - 1));
        int num_paths_continued = 0;

        for (int l = 0; l < 2 * _list_size; ++l) {
            if (probForks.at(l) > threshold) {
                contForks.at(l) = 1;
                num_paths_continued++;
            }
            if (num_paths_continued == rho) {
                break;
            }
        }

        if (num_paths_continued < rho) {
            for (int l = 0; l < 2 * _list_size; ++l) {
                if (probForks.at(l) == threshold) {
                    contForks.at(l) = 1;
                    num_paths_continued++;
                }
                if (num_paths_continued == rho) {
                    break;
                }
            }
        }

        for (int l = 0; l < _list_size; ++l) {
            if (_activePath.at(l) == 0)
                continue;
            if (contForks.at(2 * l) == 0 && contForks.at(2 * l + 1) == 0)
                killPath(l);
        }

        for (unsigned l = 0; l < _list_size; ++l) {
            if (contForks.at(2 * l) == 0 && contForks.at(2 * l + 1) == 0)
                continue;
            int* c_m = getArrayPointer_C(n_pow_, l);

            if (contForks.at(2 * l) == 1 && contForks.at(2 * l + 1) == 1) {

                c_m[(phi % 2)] = 0;
                int l_p = clonePath(l);
                c_m = getArrayPointer_C(n_pow_, l_p);
                c_m[(phi % 2)] = 1;

                std::copy(_arrayPointer_Info.at(l), _arrayPointer_Info.at(l) + phi, _arrayPointer_Info.at(l_p));
                _arrayPointer_Info.at(l)[phi] = 0;
                _arrayPointer_Info.at(l_p)[phi] = 1;

            }
            else {
                if (contForks.at(2 * l) == 1) {
                    c_m[(phi % 2)] = 0;
                    _arrayPointer_Info.at(l)[phi] = 0;
                }
                else {
                    c_m[(phi % 2)] = 1;
                    _arrayPointer_Info.at(l)[phi] = 1;
                }
            }
        }

    }

public:
    int n_pow_, n_block_, k_;
    std::vector<bool> is_info_bit_;

    int _list_size = 1;
    std::vector<int> _inactivePathIndices;
    std::vector<int> _activePath;
    std::vector<std::vector<double*>> _arrayPointer_P;
    std::vector<std::vector<int*>> _arrayPointer_C;
    std::vector<uint8_t*> _arrayPointer_Info;
    std::vector<std::vector<int>> _pathIndexToArrayIndex;
    std::vector<std::vector<int>> _inactiveArrayIndices;
    std::vector<std::vector<int>> _arrayReferenceCount;
};

int main() {
    std::random_device rd{};
    std::mt19937 gen{ rd() };

    PolarCode pcode(3, 4, 0.5);
    std::vector<unsigned char> in(pcode.k_), coded, decoded;
    for(auto & x : in){
        x = 1;
    }
    pcode.Encode(in, coded);
    pcode.Decode(coded, decoded);
    if (in != decoded) {
        std::cout << "NOT DECODED";
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
