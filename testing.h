//
// Created by ME on 20.12.2020.
//

#pragma once

#include "base.h"
#include "msf.h"
#include "viterbi.h"
#include "reedmuller.h"

#include <chrono>
#include <map>
#include <mutex>

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

int decode_count = 1000000;

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

void CheckViterbiDecoder(std::mt19937 &gen, int n, int k, int id, const matrix &code_gen_matrix) {
    std::vector<unsigned char> input(k, 0);
    SimpleEncoder enc(code_gen_matrix);
    AWGNChannel channel(0.0001);
    std::vector<std::vector<float>> transmits(decode_count);
    std::vector<std::vector<unsigned char>> codewords(decode_count);
    for(int i = 0; i < decode_count; i++){
        for(auto & bit : input){
            bit = gen() % 2;
        }
        enc.encode(input, codewords[i]);
        channel.transmit(codewords[i], transmits[i]);
    }

    ViterbiSoftDecoder dec(code_gen_matrix);
    std::vector<unsigned char> restored(n);
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < decode_count; i++){
        auto prob_log = dec.DecodeInputToCodeword(transmits[i], restored);
        if(restored != codewords[i]){
            std::cerr << "INCORRECT RECURSIVE DECODE IN " << id << "\n";
            exit(1);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Viterbi " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << " s\n";
}

void CheckSubsets(int n, int k, int id, const matrix &code_gen_matrix) {
    std::vector<int> row_starts, row_ends;
    for (auto row : code_gen_matrix) {
        row_starts.push_back(std::find(row.begin(), row.end(), 1) - row.begin());
        int one_from_end = std::find(row.rbegin(), row.rend(), 1) - row.rbegin();
        row_ends.push_back(n - one_from_end);
    }
    auto trellis = CreateCodeTrellisFromGenMatrix(code_gen_matrix);
    for (int st = 0; st < n; st++) {
        for (int fin = st + 1; fin <= n; fin++) {
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
            for (int i = 0; i < check_matrix.size(); i++) {
                if (std::find(check_matrix[i].begin(), check_matrix[i].end(), 1) != check_matrix[i].end()) {
                    check_matrix[i] = check_copy[i];
                }
            }
            matrix set_matrix(check_matrix.begin(), check_matrix.begin() + active_rows);
            matrix coset_matrix(check_matrix.begin() + active_rows, check_matrix.end());

            auto words_set = gen_all_codewords(set_matrix);
            auto words_coset = gen_all_codewords(coset_matrix);
            // check words_set are set base and words_coset are cosets base
            if (cosets.size() != words_coset.size()) {
                std::cerr << "WORDS COSET SIZE NOT EQUAL TO COSETS COUNT\n";
            }
            for (auto coset_word : words_coset) {
                std::set<unsigned long long> modified_words;
                for (auto set_word : words_set) {
                    modified_words.insert(set_word ^ coset_word);
                }
                if (cosets.count(modified_words) == 0) {
                    std::cerr << "DEDUCED SET NOT FOUND IN SETS\n";
                }
            }
        }
    }
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
