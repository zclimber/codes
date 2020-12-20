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
};

struct TrellisCompoundBranchRule {
    int first_half;
    int second_half;
    int result;

    bool operator<(const TrellisCompoundBranchRule &other) const {
        return std::tie(first_half, second_half, result) < std::tie(other.first_half, other.second_half, other.result);
    }
};

struct TrellisEdgeLabel {
    double current_value;
    unsigned long long current_label;
    unsigned size;
};

using LabelCollection = std::vector<std::vector<std::vector<TrellisEdgeLabel>>>;

using RuleCollection = std::vector<std::vector<std::vector<TrellisCompoundBranchRule>>>;

void CompareGen(int st, int fin, const BlockCodeTrellis &trellis,
                std::vector<std::vector<TrellisEdge>> &edges, LabelCollection &labels_collection,
                RuleCollection &rules) {
    unsigned label_size = fin - st;
    auto &labels = labels_collection[st][fin];
    // generate all compound branches
    auto subsets = dig_trellis_pos(st, fin, trellis);
    if (label_size == 1) {
        int mul = 1;
        if(subsets[0][0].size() == 1){
            labels.resize(2);
            labels[0] = TrellisEdgeLabel{0., 0, label_size};
            labels[1] = TrellisEdgeLabel{0., 1, label_size};
        } else {
            mul = 0;
            labels.resize(1);
            labels[0] = TrellisEdgeLabel{0., 0, label_size};
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
                    labels.push_back(TrellisEdgeLabel{0., *subsets[fin_id][st_id].begin(), label_size});
                }
                this_edges[fin_id][st_id] = insert.first->second;
            }
        }
    }

    std::vector<std::vector<TrellisEdge>> edges_1, edges_2;
    std::set<TrellisCompoundBranchRule> cbt_rules;
    CompareGen(st, (st + fin) / 2, trellis, edges_1, labels_collection, rules);
    CompareGen((st + fin) / 2, fin, trellis, edges_2, labels_collection, rules);
    edges.resize(edges_1.size());
    for (auto i = 0; i < edges_1.size(); i++) {
        for (auto &edge_1 : edges_1[i]) {
            for (auto &edge_2 : edges_2[edge_1.to]) {
                TrellisCompoundBranchRule res{edge_1.label_id, edge_2.label_id, this_edges[edge_2.to][i]};
                cbt_rules.insert(res);
                edges[i].push_back(TrellisEdge{i, edge_2.to, this_edges[edge_2.to][i]});
            }
        }
    }
    rules[st][fin].assign(cbt_rules.begin(), cbt_rules.end());
};

unsigned long long Decode(int n, const RuleCollection &rules, LabelCollection &labels_coll, const std::vector<float> &data) {
    for (int st = n - 1; st >= 0; st--) {
        for (int fin = st + 1; fin <= n; fin++) {
            auto &labels = labels_coll[st][fin];
            if (fin == st + 1) {
                if(labels.size() == 1){
                    labels[0].current_value = std::abs(data[st]);
                    labels[0].current_label = (unsigned)(data[st] > 0);
                } else {
                    labels[0].current_value = -data[st];
                    labels[1].current_value = data[st];
                }
            } else {
                int mid = (st + fin) / 2;
                auto & labels_1 = labels_coll[st][mid];
                auto & labels_2 = labels_coll[mid][fin];
                for (auto &label : labels) {
                    label.current_value = std::numeric_limits<double>::min();
                }
                for (const auto &rule : rules[st][fin]) {
                    auto link_val = labels_1[rule.first_half].current_value + labels_2[rule.second_half].current_value;
                    if (link_val > labels[rule.result].current_value) {
                        labels[rule.result].current_value = link_val;
                        labels[rule.result].current_label = labels_1[rule.first_half].current_label +
                                                            (labels_2[rule.second_half].current_label << (fin - mid));
                    }
                }
            }
        }
    }
    return labels_coll[0][n][0].current_label;
}

void CheckRecursiveDecoder(std::mt19937 &gen, int n, int k, int id, const matrix &code_gen_matrix) {
    auto trellis = CreateCodeTrellisFromGenMatrix(code_gen_matrix);
    RuleCollection rules(n + 1, RuleCollection::value_type(n+1));
    LabelCollection labels(n + 1, LabelCollection::value_type(n + 1));
    std::vector<std::vector<TrellisEdge>> edges;
    CompareGen(0, n, trellis, edges, labels, rules);


    std::vector<unsigned char> input(k, 0), encoded;
    SimpleEncoder enc(code_gen_matrix);
    AWGNChannel channel(0.0001);
    std::vector<std::vector<float>> transmits(decode_count);
    std::vector<unsigned long long> codewords(decode_count);
    for(int i = 0; i < decode_count; i++){
        for(auto & bit : input){
            bit = gen() % 2;
        }
        enc.encode(input, encoded);
        channel.transmit(encoded, transmits[i]);
        codewords[i] = vector_to_code(encoded);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < decode_count; i++){
        auto res = Decode(n, rules, labels, transmits[i]);
        if(res != codewords[i]){
            std::cerr << "INCORRECT RECURSIVE DECODE IN " << id << "\n";
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Recurse " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << " s\n";
}

void RunRandomTestsSingleThread(std::atomic_int &id_atomic, int max_id) {
    // test minspan
    std::random_device rd{};
    std::mt19937 gen{rd()};
    int n = 32, k = 12;
    matrix code_gen_matrix(k, std::vector<unsigned char>(n));
    for (;;) {
        int id = id_atomic.fetch_add(1);
        if (id % 1 == 0)
            std::cout << id << "\n";
        if (id > max_id)
            break;
        CheckVectorToCode(gen, id);
        std::set<unsigned long long> before;
        GenerateRandomCode(gen, n, k, id, code_gen_matrix, before);
        CheckCheckMatrix(n, k, code_gen_matrix);
        CheckViterbiDecoder(gen, n, k, id, code_gen_matrix, before);
        CheckRecursiveDecoder(gen, n, k, id, code_gen_matrix);
//        CheckSubsets(n, k, id, code_gen_matrix);
    }
}

void RunRandomTests(int threads_count = 1, int tests_count = 100000) {
    if(threads_count == 1){
        std::atomic_int id_atomic = 0;
        RunRandomTestsSingleThread(id_atomic, tests_count);
    } else {
        std::vector<std::thread> threads(threads_count);
        std::atomic_int id_atomic = 0;
        for (auto &thr : threads) {
            thr = std::thread([&id_atomic, tests_count]() { RunRandomTestsSingleThread(id_atomic, 100000); });
        }
        for (auto &thr : threads) {
            thr.join();
        }
    }
}

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

int main() {
    RunRandomTests();
    return 0;
    ReedMullerChecker checker;
    checker.MultiThreadedReedMullerCheck(10);
    checker.PrintDataAsCSV();
    return 0;
}
