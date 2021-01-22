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

void SolveLinearSystem(const matrix& coeff, matrix& temp, const std::vector<unsigned char>& results, std::vector<unsigned char>& solution){
    assert(coeff.size() == results.size());
    solution.assign(coeff[0].size(), 2);

    temp.resize(coeff.size());
    for(int i = 0; i < coeff.size(); i++){
        temp[i].resize(coeff[0].size() + 1);
        std::copy(coeff[i].begin(), coeff[i].end(), temp[i].begin());
        temp[i].back() = results[i];
    }
    int fixed_rows = 0;
    for(int i = 0; i  < coeff[0].size(); i++){
        for(int j = fixed_rows; j < coeff.size(); j++){
            if (temp[j][i]){
                swap(temp[fixed_rows], temp[j]);
                for(int k = fixed_rows + 1; k < coeff.size(); k++){
                    if(temp[k][i]){
                        XorVectors(temp[k], temp[fixed_rows]);
                    }
                }
                fixed_rows++;
                break;
            }
        }
    }
    for(int j = fixed_rows; j < coeff.size(); j++){
        assert(temp[j].back() == 0);
    }
    for(int i = (int)coeff[0].size() - 1; i >= 0; i--){
        for(int j = (int)coeff.size() - 1; j >= 0; j--){
            if(temp[j][i]){
                for(int ii = i - 1; ii >= 0; ii--){
                    assert(temp[j][ii] == 0);
                }
                for(int k = j - 1; k >= 0; k--){
                    if(temp[k][i]){
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

    struct CBTData{
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
            if (special_matrix.size() >= 60) {
                predicted_diff_[st][fin] = std::numeric_limits<unsigned long long>::max() / 4;
            }
            else {
                predicted_diff_[st][fin] = (fin - st) * (1ULL << special_matrix.size()) - (1ULL << active_rows[3]);
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
    CompareGenNew(st,  mid, code_gen_matrix);
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

        std::vector<unsigned char> res_l_1, res_l_2, res_r_1, res_r_2;
        MultiplyVectorByMatrix(res_l, special_matrices_[st * 1000 + mid], res_l_1);
        MultiplyVectorByMatrix(mask, eq_full_l, res_l_2);
        MultiplyVectorByMatrix(res_r, special_matrices_[mid * 1000 + fin], res_r_1);
        MultiplyVectorByMatrix(mask, eq_full_r, res_r_2);
        if (res_l_1 != res_l_2) {
            std::cerr << "BAD GAUSS\n";
        }
        if (res_r_1 != res_r_2) {
            std::cerr << "BAD GAUSS\n";
        }

        std::reverse(res_l.begin(), res_l.end());
        std::reverse(res_r.begin(), res_r.end());
        unsigned long long index_l = vector_to_code(res_l.begin(), res_l.begin() + rows_l);
        unsigned long long index_r = vector_to_code(res_r.begin(), res_r.begin() + rows_r);
        TrellisCompoundBranchRule res{ index_l, index_r, w_mask % (1ULL << active_rows[3]) };
        bool inserted = cbt_rules_3.insert(res).second;
        if (!inserted) {
            std::cerr << "Duplicate edges\n";
        }
        assert(inserted);
    }
    new_rec_rules_[st][fin].assign(cbt_rules_3.begin(), cbt_rules_3.end());

    special_matrices_[st * 1000 + fin] = std::move(special_matrix);
    active_rows_[st * 1000 + fin] = active_rows;
}

void CompareGen(int st, int fin, const BlockCodeTrellis& trellis,
                std::vector<std::vector<TrellisEdge>>& edges, LabelCollection& labels_collection,
                const matrix& code_gen_matrix, std::vector<std::vector<unsigned long long>>& min_edges,
                RuleCollection& rules) {
    unsigned label_size = fin - st;
    auto& labels = labels_collection[st][fin];
    // generate all compound branches
    if (label_size == 1) {
        matrix special_matrix;
        std::array<unsigned, 4> active_rows;
        CreateSpecialMatrix(code_gen_matrix, st, fin, fin, active_rows, special_matrix);
        start_rule_parts_[st][fin].assign(1UL << label_size, -1);
        cbt_values_[st][fin].resize(1UL << active_rows[3]);
        std::vector<unsigned char> mask, word;
        for (unsigned long long w_mask = 0; w_mask < (1ULL << special_matrix.size()); w_mask++) {
            code_to_vector(w_mask, special_matrix.size(), mask);
            MultiplyVectorByMatrix(mask, special_matrix, word);
            auto group = w_mask % (1ULL << active_rows[3]);
            start_rule_parts_[st][fin][vector_to_code(word)] = group;
        }
        special_matrices_[st * 1000 + fin] = std::move(special_matrix);
        active_rows_[st * 1000 + fin] = active_rows;

        int fin_size = trellis.layer_end[fin] - trellis.layer_start[fin];
        bool is_size_2 = trellis.data[trellis.layer_start[fin]].prev_cells[0] ==
            trellis.data[trellis.layer_start[fin]].prev_cells[1];
        int st_start = trellis.layer_start[st];
        edges.resize(trellis.layer_end[st] - st_start);
        for (int fin_id = trellis.layer_start[fin]; fin_id < trellis.layer_end[fin]; fin_id++) {
            int fin_edge_id = fin_id - trellis.layer_start[fin];
            int pr_0 = trellis.data[fin_id].prev_cells[0] - st_start;
            int pr_1 = trellis.data[fin_id].prev_cells[1] - st_start;
            if (pr_0 >= 0) {
                edges[pr_0].push_back(TrellisEdge{pr_0, fin_edge_id, 0});
            }
            if (pr_1 != pr_0 && pr_1 >= 0) {
                edges[pr_1].push_back(TrellisEdge{pr_1, fin_edge_id, 1});
            }
        }
        if (!is_size_2) {
            labels.resize(2);
            labels[0] = TrellisEdgeLabel{0, 0.f};
            labels[1] = TrellisEdgeLabel{1, 0.f};
        } else {
            labels.resize(1);
            labels[0] = TrellisEdgeLabel{0, 0.f};
        }
        return;
    }

    std::map<unsigned long long, int> compound_ids;
    std::vector<std::vector<int>> this_edges(min_edges.size());

    std::vector<std::vector<TrellisEdge>> edges_1, edges_2;
    std::set<TrellisCompoundBranchRule> cbt_rules;
    int mid = (st + fin) / 2;
    CompareGen(st, mid, trellis, edges_1, labels_collection, code_gen_matrix, min_edges, rules);
    CompareGen(mid, fin, trellis, edges_2, labels_collection, code_gen_matrix, min_edges, rules);
    edges.resize(edges_1.size());

    std::vector<std::vector<std::tuple<int, unsigned long long, int, int>>> edges_temp(edges_1.size());

    auto& labels_1 = labels_collection[st][mid];
    auto& labels_2 = labels_collection[mid][fin];
    for (auto i = 0; i < edges_1.size(); i++) {
        for (auto& edge_1 : edges_1[i]) {
            auto label_1 = labels_1[edge_1.label_id].current_label;
            for (auto& edge_2 : edges_2[edge_1.to]) {
                auto label_2 = labels_2[edge_2.label_id].current_label;
                unsigned long long probable_label = label_1 + (label_2 << (mid - st));
                edges_temp[i].emplace_back(edge_2.to, probable_label, edge_1.label_id, edge_2.label_id);
            }
        }
        std::sort(edges[i].begin(), edges[i].end());
        edges[i].erase(std::unique(edges[i].begin(), edges[i].end()), edges[i].end());
    }
    std::vector<std::vector<TrellisEdge>> edges_real(edges_1.size());
    std::set<TrellisCompoundBranchRule> cbt_rules_2;
    for (int i = 0; i < edges_temp.size(); i++) {
        std::sort(edges_temp[i].begin(), edges_temp[i].end());
        int last_to = -1;
        unsigned long long last_label = 0;
        int last_id = -1;
        for (auto [to, label, id1, id2] : edges_temp[i]) {
            if (last_to != to) {
                last_to = to;
                last_label = label;
                auto insert = compound_ids.insert({label, (int)compound_ids.size()});
                if (insert.second) {
                    labels.push_back(TrellisEdgeLabel{label, 0.f});
                }
                last_id = insert.first->second;
                edges_real[i].push_back(TrellisEdge{i, to, last_id});
            }
            TrellisCompoundBranchRule res{id1, id2, last_id};
            cbt_rules_2.insert(res);
        }
        edges[i] = std::move(edges_real[i]);
    }
    rules[st][fin].assign(cbt_rules_2.begin(), cbt_rules_2.end());
//    return;

    //start special matrix check
    cbt_rules.clear();
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
    auto& new_rules = labels_collection[st][fin];
    std::set<TrellisCompoundBranchRule> cbt_rules_3;
    for (unsigned long long w_mask = 0; w_mask < (1ULL << w_pow); w_mask++) {
        code_to_vector(w_mask, mask);
        std::reverse(mask.begin(), mask.end());
        MultiplyVectorByMatrix(mask, solutions_l, res_l);
        MultiplyVectorByMatrix(mask, solutions_r, res_r);

        std::vector<unsigned char> res_l_1, res_l_2, res_r_1, res_r_2;
        MultiplyVectorByMatrix(res_l, special_matrices_[st * 1000 + mid], res_l_1);
        MultiplyVectorByMatrix(mask, eq_full_l, res_l_2);
        MultiplyVectorByMatrix(res_r, special_matrices_[mid * 1000 + fin], res_r_1);
        MultiplyVectorByMatrix(mask, eq_full_r, res_r_2);
        if (res_l_1 != res_l_2) {
            std::cerr << "BAD GAUSS\n";
        }
        if (res_r_1 != res_r_2) {
            std::cerr << "BAD GAUSS\n";
        }

        std::reverse(res_l.begin(), res_l.end());
        std::reverse(res_r.begin(), res_r.end());
        unsigned long long index_l = vector_to_code(res_l.begin(), res_l.begin() + rows_l);
        unsigned long long index_r = vector_to_code(res_r.begin(), res_r.begin() + rows_r);
        TrellisCompoundBranchRule res{ index_l, index_r, w_mask % (1ULL << active_rows[3]) };
        cbt_rules_3.insert(res);
    }
    new_rec_rules_[st][fin].assign(cbt_rules_3.begin(), cbt_rules_3.end());

    special_matrices_[st * 1000 + fin] = std::move(special_matrix);
    active_rows_[st * 1000 + fin] = active_rows;
}

void CreateSpecialMatrix(const matrix& code_gen_matrix, int st, int fin, int mid, std::array<unsigned, 4> & parts_sizes, matrix& special_matrix) {

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
        if (i == active_rows) {
            assert(special_rows == active_rows);
        }
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
                set_groups.assign(labels.size(), true);
                assert(starts.back() != -1);
                for (unsigned word = 0; word < starts.size(); word ++) {
                    if (starts[word] == -1)
                        continue;
                    float prob = (word & 1) ? data[st] : -data[st];
                    unsigned ww = word / 2;
                    for (int i = st + 1; i < fin; i++, ww /= 2) {
                        rec_adds_2++;
                        prob += (ww & 1) ? data[i] : -data[i];
                    }
                    if (set_groups[starts[word]]) {
                        set_groups[starts[word]] = false;
                        labels[starts[word]].label = word;
                        labels[starts[word]].value = prob;
                    }
                    else {
                        rec_comps_2++;
                        if (labels[starts[word]].value < prob) {
                            labels[starts[word]].label = word;
                            labels[starts[word]].value = prob;
                        }
                    }
                }
                continue;
                unsigned inverse_mask = starts.size() - 1;
                for (unsigned word = 0; word < starts.size(); word += 2) {
                    if (starts[word] == -1)
                        continue;
                    float prob = -data[st];
                    unsigned ww = word / 2;
                    for (int i = st + 1; i < fin; i++, ww /= 2) {
                        rec_adds_2++;
                        if (ww & 1) {
                            prob += data[i];
                        }
                        else {
                            prob -= data[i];
                        }
                    }
                    if (prob < 0) {

                    }
                    unsigned wcur = word;
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

unsigned long long
Decode(int n, const RuleCollection& rules, LabelCollection& labels_coll, const std::vector<float>& data) {
    for (int st = n - 1; st >= 0; st--) {
        for (int fin = st + 1; fin <= n; fin++) {
            auto& labels = labels_coll[st][fin];
            if (labels_coll[st][fin].empty())
                continue;
            if (fin == st + 1) {
                if (labels.size() == 1) {
                    labels[0].current_value = std::abs(data[st]);
                    labels[0].current_label = (unsigned)(data[st] > 0);
                } else {
                    labels[0].current_value = -data[st];
                    labels[1].current_value = data[st];
                }
            } else {
                int mid = (st + fin) / 2;
                auto& labels_1 = labels_coll[st][mid];
                auto& labels_2 = labels_coll[mid][fin];
                for (auto& label : labels) {
                    label.current_value = std::numeric_limits<float>::min();
                }
                rec_comps--;
                for (const auto& rule : rules[st][fin]) {
                    rec_comps++;
                    rec_adds++;
                    auto link_val = labels_1[rule.first_half].current_value + labels_2[rule.second_half].current_value;
                    if (link_val > labels[rule.result].current_value) {
                        labels[rule.result].current_value = link_val;
                        labels[rule.result].current_label = labels_1[rule.first_half].current_label +
                            (labels_2[rule.second_half].current_label << (mid - st));
                    }
                }
            }
        }
    }
    return labels_coll[0][n][0].current_label;
}

void CheckRecursiveDecoder(std::mt19937& gen, int n, int k, int id, const matrix& code_gen_matrix) {
    //auto trellis = CreateCodeTrellisFromGenMatrix(code_gen_matrix);
    //RuleCollection rules(n, RuleCollection::value_type(n + 1));
    //LabelCollection labels(n, LabelCollection::value_type(n + 1));
    //std::vector<std::vector<TrellisEdge>> edges;
    //std::vector<std::vector<unsigned long long>> edges_buf;
    auto start0 = std::chrono::high_resolution_clock::now();
    //RecursiveGenContext ctx(n);
    //ctx.CompareGen(0, n, trellis, edges, labels, code_gen_matrix, edges_buf, rules);
    RecursiveGenContext ctx2(n);
    ctx2.PredictComputationDifficulty(n, code_gen_matrix);
    std::cout << "Predicting minimal " << ctx2.predicted_diff_[0][n] << " ops\n";

    ctx2.CompareGenNew(0, n, code_gen_matrix);
    auto end0 = std::chrono::high_resolution_clock::now();
    std::cout << "Recurse creation " << std::chrono::duration_cast<std::chrono::duration<double>>(end0 - start0).count() << " s\n";


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
        //auto res = Decode(n, ctx, transmits[i]);
        auto res2 = Decode(n, ctx2, transmits[i]);
        //if (res != codewords[i] && i < decode_count) {
        //    auto res = Decode(n, rules, labels, transmits[i]);
        //    std::cerr << "INCORRECT RECURSIVE DECODE IN " << id << "\n";
        //}
        if (res2 != codewords[i]) {
            std::cerr << "INCORRECT NEW RECURSIVE DECODE IN " << id << "\n";
            auto res3 = Decode(n, ctx2, transmits[i]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Recurse " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << " s\n";
}

void RunRandomTestsSingleThread(std::atomic_int& id_atomic, int max_id) {
    // test minspan
    std::random_device rd{};
    std::mt19937 gen{rd()};
    int n = 24, k = 12;
    matrix code_gen_matrix(k, std::vector<unsigned char>(n));
    for (;;) {
        vit_adds = vit_comps = rec_adds = rec_comps = rec_adds_2 = rec_comps_2 = 0;
        int id = id_atomic.fetch_add(1);
        if (id % 1 == 0)
            std::cout << id << "\n";
        if (id >= max_id)
            break;
        std::set<unsigned long long> before;
        GenerateRandomCode(gen, n, k, id, code_gen_matrix, before);
        CheckVectorToCode(gen, id);
        GenerateMinimalSpanMatrix(code_gen_matrix, n, k);
        CheckCheckMatrix(n, k, code_gen_matrix);
        CheckViterbiDecoder(gen, n, k, id, code_gen_matrix);
        CheckRecursiveDecoder(gen, n, k, id, code_gen_matrix);
        std::cout << vit_adds / decode_count << "\t" << vit_comps / decode_count << "\t" << (vit_adds+ vit_comps) / decode_count << "\n";
        std::cout << rec_adds / decode_count << "\t" << rec_comps / decode_count << "\t" << (rec_adds + rec_comps) / decode_count << "\n";
        std::cout << rec_adds_2 / decode_count << "\t" << rec_comps_2 / decode_count << "\t" << (rec_adds_2 + rec_comps_2) / decode_count << "\n\n";
        CheckSubsets(n, k, id, code_gen_matrix);
    }
}

void RunRandomTests(int threads_count = 1, int tests_count = 100000) {
    if (threads_count == 1) {
        std::atomic_int id_atomic = 0;
        RunRandomTestsSingleThread(id_atomic, tests_count);
    } else {
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

int main() {
    matrix code_gen_matrix;
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::vector<std::pair<int, int>> codes = {{3,1}, {3,2}, {4,2}, {5,2}, {5,3}, {6,2}, { 6, 3 }, {6,4}
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
        CheckViterbiDecoder(gen, n, k, 0, code_gen_matrix);
        CheckRecursiveDecoder(gen, n, k, 0, code_gen_matrix);
        std::cout << "RM(" << r << "," << m << ") " << n << " " << k << "\n";
        std::cout << vit_adds / decode_count << "\t" << vit_comps / decode_count << "\t" << (vit_adds + vit_comps) / decode_count << "\n";
        std::cout << rec_adds / decode_count << "\t" << rec_comps / decode_count << "\t" << (rec_adds + rec_comps) / decode_count << "\n";
        std::cout << rec_adds_2 / decode_count << "\t" << rec_comps_2 / decode_count << "\t" << (rec_adds_2 + rec_comps_2) / decode_count << "\n\n";
    }
    return 0;
    RunRandomTests(1, 100000);
    return 0;
    ReedMullerChecker checker;
    checker.MultiThreadedReedMullerCheck(10);
    checker.PrintDataAsCSV();
    return 0;
}
