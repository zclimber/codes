//
// Created by ME on 20.12.2020.
//

#pragma once

#define _USE_MATH_DEFINES

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <vector>
#include <set>
#include <sstream>

using matrix = std::vector<std::vector<unsigned char>>;

std::string PrintVector(const std::vector<unsigned char>& data) {
    std::string res;
    res.resize((data.size() * 2), '0');
    for (int i = 0; i < data.size(); i++) {
        res[i * 2] += data[i];
        res[i * 2 + 1] = ' ';
    }
    res.pop_back();
    return res;
}

std::string PrintMatrix(const matrix& data) {
    std::stringstream ss;
    for (auto& row : data) {
        ss << PrintVector(row) << "\n";
    }
    return ss.str();
}

void XorVectors(std::vector<unsigned char>& vec1, const std::vector<unsigned char>& vec2) {
    for (int j = 0; j < vec2.size(); j++) {
        vec1[j] ^= vec2[j];
    }
}

void AndVectors(std::vector<unsigned char>& vec1, const std::vector<unsigned char>& vec2) {
    for (int j = 0; j < vec2.size(); j++) {
        vec1[j] &= vec2[j];
    }
}

void MultiplyVectorByMatrix(const std::vector<unsigned char>& data, const matrix& gen_matrix,
    std::vector<unsigned char>& res) {
    res.assign(gen_matrix.front().size(), 0U);
    for (int row = 0; row < gen_matrix.size(); row++) {
        if (data[row] == 1) {
            XorVectors(res, gen_matrix[row]);
        }
    }
}

unsigned long long vector_to_code(const std::vector<unsigned char>& vec) {
    unsigned long long res = 0;
    for (unsigned i = 0; i < vec.size(); i++) {
        res |= static_cast<unsigned long long>(vec[i]) << i;
    }
    return res;
}

template<typename Iterator>
unsigned long long vector_to_code(Iterator begin, Iterator end) {
    unsigned long long res = 0;
    auto size = end - begin;
    for (unsigned i = 0; i < size; i++) {
        res |= static_cast<unsigned long long>(begin[i]) << i;
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

void code_to_vector(unsigned long long code, int size, std::vector<unsigned char>& res) {
    res.resize(size);
    for (int i = 0; i < size; i++) {
        res[i] = code & 1U;
        code >>= 1U;
    }
}

void code_to_vector(unsigned long long code, std::vector<unsigned char>& res) {
    auto size = res.size();
    for (unsigned i = 0; i < size; i++) {
        res[i] = code & 1U;
        code >>= 1U;
    }
}

std::set<unsigned long long> gen_all_codewords(const matrix& gen_matrix) {
    std::vector<unsigned long long> base;
    base.reserve(gen_matrix.size());
    for (const auto& vec : gen_matrix) {
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

void SwapColumns(matrix& m, int col1, int col2) {
    if (col1 == col2)
        return;
    for (int i = 0; i < m.size(); i++) {
        std::swap(m[i][col1], m[i][col2]);
    }
}

void GetSystematicLikeMatrix(matrix& temp, std::vector<int>& columns) {
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

matrix RevertColumnSwappedMatrix(const matrix& temp, const std::vector<int>& columns) {
    matrix res(temp.size());
    for (int i = 0; i < temp.size(); i++) {
        res[i].resize(columns.size());
        for (int j = 0; j < columns.size(); j++) {
            res[i][columns[j]] = temp[i][j];
        }
    }
    return res;
}

matrix TransposeMatrix(const matrix& matrix1) {
    matrix res(matrix1[0].size());
    for (unsigned i = 0; i < res.size(); i++) {
        res[i].resize(matrix1.size());
        for (unsigned j = 0; j < matrix1.size(); j++) {
            res[i][j] = matrix1[j][i];
        }
    }
    return res;
}

matrix GenerateTransposedCheckMatrix(const matrix& gen_matrix, int n, int k) {
    matrix temp = gen_matrix;
    std::vector<int> columns;
    GetSystematicLikeMatrix(temp, columns);

    int r = n - k;
    matrix res(n);
    for (int i = 0; i < n; i++) {
        if (i >= k) {
            res[i].resize(r);
            res[i][i - k] = 1;
        }
        else {
            res[i] = { temp[i].begin() + k, temp[i].end() };
        }
    }
    matrix resres(n);
    for (int i = 0; i < n; i++) {
        resres[columns[i]] = std::move(res[i]);
    }
    return resres;
}

// Es = 1., N0 = sigma * sigma * 2;
class AWGNChannel {
public:
    explicit AWGNChannel(float noise_sigma) : gen_(std::random_device{}()), noise_(0., noise_sigma),
        sqrt_2pi_N0_inv_(1. / (std::sqrt(2 * M_PI) * noise_sigma)), N0_(noise_sigma* noise_sigma * 2), two_on_sq_sigma_(2 / (noise_sigma * noise_sigma)) {
    }

    template<typename Float>
    void transmit(const std::vector<unsigned char>& data, std::vector<Float>& result) {
        result.resize(data.size());
        for (auto i = 0U; i < data.size(); i++) {
            result[i] = noise_(gen_) + (data[i] == 1 ? 1.f : -1.f);
        }
    }

    double llr(double signal) {
        return signal * two_on_sq_sigma_;
    }

    void llr(const std::vector<double>& transmitted, std::vector<double>& llrs) {
        llrs.resize(transmitted.size());
        for (int i = 0; i < transmitted.size(); i++) {
            llrs[i] = llr(transmitted[i]);
        }
    }

    double probability(unsigned char bit, double signal) {
        if (bit == 0) {
            signal += 1;
        }
        else {
            signal -= 1;
        }
        return sqrt_2pi_N0_inv_ * std::exp((signal * signal) * -1 / N0_);
    }

    void probability(double signal, std::array<double, 2> & prob) {
        prob[0] = probability(0, signal);
        prob[1] = probability(1, signal);
    }

    void probability(const std::vector<double>& transmitted, std::vector<std::array<double, 2>> & prob) {
        prob.resize(transmitted.size());
        for (int i = 0; i < transmitted.size(); i++) {
            probability(transmitted[i], prob[i]);
        }
    }

    void probability(unsigned char bit, const std::vector<double>& transmitted, std::vector<double>& prob) {
        prob.resize(transmitted.size());
        for (int i = 0; i < transmitted.size(); i++) {
            prob[i] = probability(bit, transmitted[i]);
        }
    }

    friend AWGNChannel AWGNChannelFromSNR(double snr_db);

private:
    std::mt19937 gen_;
    std::normal_distribution<float> noise_;
    double sqrt_2pi_N0_inv_;
    double N0_;
    double two_on_sq_sigma_;
};

//snr == ES/N0
inline AWGNChannel AWGNChannelFromSNR(double snr_db) {
    double sigma = std::sqrt(1. / (2. * std::pow(10., (snr_db / 10))));
    return AWGNChannel(sigma);
}
inline AWGNChannel AWGNChannelFromEBN0(double ebn0_db, int code_n, int code_k) {
    double snr_db = ebn0_db + 10. * std::log10(double(code_k) / code_n);
    return AWGNChannelFromSNR(snr_db);
}
inline AWGNChannel AWGNChannelFromSigma(double sigma) {
    return AWGNChannel(sigma);
}

class SimpleEncoder {
public:
    explicit SimpleEncoder(matrix gen_matrix) : n(gen_matrix.front().size()), k(gen_matrix.size()),
        gen_matrix(std::move(gen_matrix)) {
    }

    void encode(const std::vector<unsigned char>& data, std::vector<unsigned char>& res) {
        MultiplyVectorByMatrix(data, gen_matrix, res);
    }

private:
    int n = 0, k = 0;
    matrix gen_matrix;
};
