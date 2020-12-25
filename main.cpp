#include "testing.h"

#include "base.h"
#include "msf.h"
#include "viterbi.h"
#include "reedmuller.h"

#include <iomanip>
#include <thread>
#include <atomic>

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

void CompareGen(int st, int fin, const BlockCodeTrellis& trellis,
                std::vector<std::vector<TrellisEdge>>& edges, LabelCollection& labels_collection,
                const matrix& code_gen_matrix, std::vector<std::vector<unsigned long long>>& min_edges,
                RuleCollection& rules) {
    unsigned label_size = fin - st;
    auto& labels = labels_collection[st][fin];
    // generate all compound branches
    if (label_size == 1) {
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

    //dig_trellis_pos(st, fin, trellis, min_edges);
    std::map<unsigned long long, int> compound_ids;
    std::vector<std::vector<int>> this_edges(min_edges.size());
    //labels.clear();
    //for (int fin_id = 0; fin_id < min_edges.size(); fin_id++) {
    //    this_edges[fin_id].resize(min_edges[0].size(), -1);
    //    for (int st_id = 0; st_id < min_edges[0].size(); st_id++) {
    //        if (min_edges[fin_id][st_id] != std::numeric_limits<unsigned long long>::max()) {
    //            auto insert = compound_ids.insert({min_edges[fin_id][st_id], (int)compound_ids.size()});
    //            if (insert.second) {
    //                labels.push_back(TrellisEdgeLabel{min_edges[fin_id][st_id], 0.f});
    //            }
    //            this_edges[fin_id][st_id] = insert.first->second;
    //        }
    //    }
    //}

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

                //TrellisCompoundBranchRule res{edge_1.label_id, edge_2.label_id, this_edges[edge_2.to][i]};
                //cbt_rules.insert(res);
                //edges[i].push_back(TrellisEdge{i, edge_2.to, this_edges[edge_2.to][i]});
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
                //auto actual_label = labels[this_edges[to][i]].current_label;
                //if (actual_label != label) {
                //    std::cerr << "PROBABLE LABEL DIFFERENT FROM ACTUAL : " << label << " " << actual_label
                //        << "\n";
                //}
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
        //if (edges_real[i] != edges[i]) {
        //    std::cerr << "EDGES SET DIFFERENT FROM ACTUAL\n";
        //}
    }
    //if (cbt_rules_2 != cbt_rules) {
    //        //std::cerr << "RULES SET DIFFERENT FROM ACTUAL\n";
    //}
    rules[st][fin].assign(cbt_rules_2.begin(), cbt_rules_2.end());
    return;

    //start special matrix check
    cbt_rules.clear();
    matrix special_matrix, special_second_part;
    std::vector<int> row_starts, row_ends;
    int n = code_gen_matrix[0].size();
    for (auto& row : code_gen_matrix) {
        int first_active = std::find(row.begin(), row.end(), 1) - row.begin();
        int last_active = n - (std::find(row.rbegin(), row.rend(), 1) - row.rbegin());
        if (first_active >= st && last_active <= fin) {
            special_matrix.emplace_back(row.begin() + st, row.begin() + fin);
        } else {
            special_second_part.emplace_back(row.begin() + st, row.begin() + fin);
        }
    }
    unsigned active_rows = special_matrix.size();
    for (auto& row : special_second_part) {
        special_matrix.push_back(std::move(row));
    }

    matrix special_copy = special_matrix;
    std::vector<int> cols;
    GetSystematicLikeMatrix(special_copy, cols);
    int special_rows = 0;
    for (int i = 0; i < special_copy.size(); i++) {
        if (std::find(special_copy[i].begin(), special_copy[i].end(), 1) != special_copy[i].end()) {
            special_matrix[special_rows++] = std::move(special_matrix[i]);
        }
    }
    special_matrix.resize(special_rows);

    for (int i = 0; i < special_rows; i++) {
        special_matrix[i].resize(fin - st + special_rows - active_rows, 0);
        if (i >= active_rows) {
            special_matrix[i][fin - st + i - active_rows] = 1;
        }
    }
};

void CompareGen2(int st, int fin, const BlockCodeTrellis& trellis,
                 std::vector<std::vector<TrellisEdge>>& edges, LabelCollection& labels_collection,
                 RuleCollection& rules) {
    unsigned label_size = fin - st;
    auto& labels = labels_collection[st][fin];
    // generate all compound branches
    auto subsets = dig_trellis_pos(st, fin, trellis);
    if (label_size == 1) {
        int mul = 1;
        if (subsets[0][0].size() == 1) {
            labels.resize(2);
            labels[0] = TrellisEdgeLabel{0, 0.};
            labels[1] = TrellisEdgeLabel{1, 0.};
        } else {
            mul = 0;
            labels.resize(1);
            labels[0] = TrellisEdgeLabel{0, 0.};
        }
        edges.resize(subsets[0].size());
        for (int fin_id = 0; fin_id < subsets.size(); fin_id++) {
            for (int st_id = 0; st_id < subsets[0].size(); st_id++) {
                if (!subsets[fin_id][st_id].empty()) {
                    int label = *subsets[fin_id][st_id].begin() * mul;
                    edges[st_id].push_back(TrellisEdge{st_id, fin_id, label});
                }
            }
        }
        return;
    }
    std::map<unsigned long long, unsigned long long> compound_ids;
    std::vector<std::vector<int>> this_edges(subsets.size());
    labels.clear();
    for (int fin_id = 0; fin_id < subsets.size(); fin_id++) {
        this_edges[fin_id].resize(subsets[0].size(), -1);
        for (int st_id = 0; st_id < subsets[0].size(); st_id++) {
            if (subsets[fin_id][st_id].size() > 0) {
                auto insert = compound_ids.insert({*subsets[fin_id][st_id].begin(), compound_ids.size()});
                if (insert.second) {
                    labels.push_back(TrellisEdgeLabel{*subsets[fin_id][st_id].begin(), 0.});
                }
                this_edges[fin_id][st_id] = insert.first->second;
            }
        }
    }

    std::vector<std::vector<TrellisEdge>> edges_1, edges_2;
    std::set<TrellisCompoundBranchRule> cbt_rules;
    CompareGen2(st, (st + fin) / 2, trellis, edges_1, labels_collection, rules);
    CompareGen2((st + fin) / 2, fin, trellis, edges_2, labels_collection, rules);
    edges.resize(edges_1.size());
    for (auto i = 0; i < edges_1.size(); i++) {
        for (auto& edge_1 : edges_1[i]) {
            for (auto& edge_2 : edges_2[edge_1.to]) {
                TrellisCompoundBranchRule res{edge_1.label_id, edge_2.label_id, this_edges[edge_2.to][i]};
                cbt_rules.insert(res);
                edges[i].push_back(TrellisEdge{i, edge_2.to, this_edges[edge_2.to][i]});
            }
        }
    }
    rules[st][fin].assign(cbt_rules.begin(), cbt_rules.end());
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
    auto trellis = CreateCodeTrellisFromGenMatrix(code_gen_matrix);
    RuleCollection rules(n + 1, RuleCollection::value_type(n + 1));
    LabelCollection labels(n + 1, LabelCollection::value_type(n + 1));
    std::vector<std::vector<TrellisEdge>> edges;
    std::vector<std::vector<unsigned long long>> edges_buf;
    CompareGen(0, n, trellis, edges, labels, code_gen_matrix, edges_buf, rules);


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
        auto res = Decode(n, rules, labels, transmits[i]);
        if (res != codewords[i] && i < decode_count) {
            auto res = Decode(n, rules, labels, transmits[i]);
            std::cerr << "INCORRECT RECURSIVE DECODE IN " << id << "\n";
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
        vit_adds = vit_comps = rec_adds = rec_comps = 0;
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
        std::cout << vit_adds / decode_count << "\t" << vit_comps / decode_count << "\n";
        std::cout << rec_adds / decode_count << "\t" << rec_comps / decode_count << "\n\n";
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
    std::vector<std::pair<int, int>> codes = {{5, 3}};
    for (auto [m, r] : codes) {
        vit_adds = vit_comps = rec_adds = rec_comps = 0;
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
        std::cout << vit_adds / decode_count << "\t" << vit_comps / decode_count << "\n";
        std::cout << rec_adds / decode_count << "\t" << rec_comps / decode_count << "\n\n";
    }
    //return 0;
    RunRandomTests(6, 100000);
    return 0;
    ReedMullerChecker checker;
    checker.MultiThreadedReedMullerCheck(10);
    checker.PrintDataAsCSV();
    return 0;
}
