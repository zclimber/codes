#pragma once

#include "base.h"

#include "viterbi.h"

#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstdint>
#include <vector>
#include <array>
#include <optional>

#define LLR

class PolarCode {
public:
    PolarCode(size_t num_layers, size_t info_length, double epsilon) :
        n_pow_(num_layers), info_size_(info_length), design_epsilon_(epsilon)
    {
        block_size_ = 1 << n_pow_;
        is_frozen_bit_.resize(block_size_);
        bit_rev_sorted_ids_.resize(block_size_);
        channels_ordered_by_rel_.resize(block_size_);
        InitBitReversedIds();
        InitPolarCodeBhatt();
    }

    void InitInnerTrellisDecoder();

    void Encode(const std::vector<uint8_t>& info_bits, std::vector<uint8_t>& codeword);
    void Decode(const std::vector<double>& llr, size_t list_size, std::vector<uint8_t>& info_bits);
    void Decode(const std::vector<double>& p1, const std::vector<double>& p0, size_t list_size, std::vector<uint8_t>& info_bits);
    void Decode(const std::vector<std::array<double, 2>>& probabilities, size_t list_size, std::vector<uint8_t>& info_bits);

    void InitPolarCodeGaussApprox(double noise_sigma);

    void SetInfoBits(const std::vector<int>& bits);

    size_t GetCodeInfoSize() const {
        return info_size_;
    }

    size_t GetCodeBlockSize() const {
        return block_size_;
    }

public:

    size_t n_pow_;
    size_t info_size_;
    size_t block_size_;
    int trellis_n_ = 2;
    bool inner_decoder_ = false;

    double design_epsilon_;

    std::vector<bool> is_frozen_bit_;
    std::vector<int> channels_ordered_by_rel_;
    std::vector<int> info_channels_;
    std::vector<int> bit_rev_sorted_ids_;

    std::vector<uint8_t> codeword_, codeword_buf_;
    std::vector<uint8_t> bits_padded_;
    std::vector<uint8_t> info_bits_;

    void InitPolarCodeBhatt();
    void InitBitReversedIds();

    void DecodeSCL(std::vector<uint8_t>& codeword);
    void GetInfoBitsFromCodeword(const std::vector<uint8_t>& codeword, std::vector<uint8_t>& info_bits);

    size_t list_size_;

    std::vector<int> inactive_path_indices_;
    std::vector<int> active_path_;

    using PArrayType = std::vector<std::array<double, 2>>;
    using CArrayType = std::vector<std::array<uint8_t, 2>>;

    std::vector<double> path_metric_;
    std::vector<std::vector<PArrayType>> arrays_P_;
    std::vector<std::vector<CArrayType>> arrays_C_;
    std::vector<std::vector<std::vector<double>>> arrays_LLR_;
    std::vector<std::vector<uint8_t>> deduced_padded_bits_;
    std::vector<std::vector<int>> pathIndex_to_arrayIndex_;
    std::vector<std::vector<int>> inactive_array_indexes_;
    std::vector<std::vector<int>> array_references_count_;

    std::vector<std::array<bool, 2>> contForks;
    std::vector<std::pair<double, int>> probabilities;
    std::vector<std::pair<double, int>> probabilities_LLR_;

    std::vector<std::optional<ViterbiSoftDecoder>> trellis_decoders_;

    void initializeDataStructures();
    size_t assignInitialPath();
    size_t clonePath(size_t);
    void killPath(size_t l);

    size_t GetArrayPointerIndex(size_t lambda, size_t l);
    PArrayType& getArrayPointer_P(size_t lambda, size_t  l);
    CArrayType& getArrayPointer_C(size_t lambda, size_t  l);
    std::vector<double>& getArrayPointer_LLR(size_t lambda, size_t l);

    void recursivelyCalcP(size_t lambda, size_t phi);
    void recursivelyCalcLLR(size_t lambda, size_t phi);
    void recursivelyUpdateC(size_t lambda, size_t phi);

    void continuePaths_FrozenBit(size_t phi);
    void PopulateContForks();
    void PopulateContForksLLR();
    void continuePaths_UnfrozenBit(size_t phi);

    void continuePaths_Trellis(size_t phi);

    size_t findMostProbablePath();
};

template<typename T>
T stack_pop(std::vector<T>& stack) {
    if (stack.empty()) {
        throw std::exception("Stack is empty");
    }
    T res = stack.back();
    stack.pop_back();
    return res;
}

double CalcMetric(double value, uint8_t bit) {
    bool value_sign = value < 0;
    if (value_sign == bit) {
        return 0;
    }
    else {
        return value_sign ? value : -value;
    }
}

thread_local std::vector<uint8_t> bit_reverse_temp_;
void BitReverseArray(std::vector<uint8_t>& vector_to_reverse, const std::vector<int>& bits_reversed_order) {
    bit_reverse_temp_.resize(bits_reversed_order.size());
    for (int i = 0; i < bits_reversed_order.size(); i++) {
        bit_reverse_temp_[i] = vector_to_reverse[bits_reversed_order[i]];
    }
    swap(bit_reverse_temp_, vector_to_reverse);
}

void PolarCode::SetInfoBits(const std::vector<int>& bits) {
    is_frozen_bit_.assign(block_size_, true);
    for (auto bit_id : bits) {
        is_frozen_bit_[bit_id] = false;
    }
}

void PolarCode::InitPolarCodeBhatt() {
    std::vector<double> channel_vec(block_size_);

    for (int i = 0; i < block_size_; i++) {
        channel_vec[i] = design_epsilon_;
    }
    for (int iteration = 0; iteration < n_pow_; iteration++) {
        int increment = 1 << iteration;
        for (int j = 0; j < increment; j += 1) {
            for (int i = 0; i < block_size_; i += 2 * increment) {
                double c1 = channel_vec[i + j];
                double c2 = channel_vec[i + j + increment];
                channel_vec[i + j] = c1 + c2 - c1 * c2;
                channel_vec[i + j + increment] = c1 * c2;
            }
        }
    }

    std::iota(channels_ordered_by_rel_.begin(), channels_ordered_by_rel_.end(), 0);
    std::sort(channels_ordered_by_rel_.begin(),
        channels_ordered_by_rel_.end(),
        [&](size_t i1, size_t i2) { return channel_vec[bit_rev_sorted_ids_[i1]] < channel_vec[bit_rev_sorted_ids_[i2]]; });

    info_channels_.assign(channels_ordered_by_rel_.begin(), channels_ordered_by_rel_.begin() + info_size_);
    std::sort(info_channels_.begin(), info_channels_.end());
    SetInfoBits(info_channels_);
}

double XiFuncFast(double x) {
    if (x > 12) {
        return 0.9861 * x - 2.3152;
    }
    else if (x > 3.5) {
        return x * (9.005e-3 * x + 0.7694) - 0.9507;
    }
    else if (x > 1) {
        return x * (0.062883 * x + 0.3678) - 0.1627;
    }
    else {
        return x * (0.2202 * x + 0.06448);
    }
}

void PolarCode::InitPolarCodeGaussApprox(double noise_sigma) {
    std::vector<double> channel_rel_log(block_size_);

    double initial_rel_log = 2.0 / (noise_sigma * noise_sigma);
    channel_rel_log[0] = initial_rel_log;

    for (int lambda = 1; lambda <= n_pow_; lambda++) {
        for (int i = (1 << lambda) / 2 - 1; i >= 0; i--) {
            channel_rel_log[i * 2 + 1] = channel_rel_log[i] * 2;
            channel_rel_log[i * 2] = XiFuncFast(channel_rel_log[i]);
        }
    }

    std::iota(channels_ordered_by_rel_.begin(), channels_ordered_by_rel_.end(), 0);
    std::sort(channels_ordered_by_rel_.begin(),
        channels_ordered_by_rel_.end(),
        [&](size_t i1, size_t i2) { return channel_rel_log[bit_rev_sorted_ids_[i1]] > channel_rel_log[bit_rev_sorted_ids_[i2]]; });

    info_channels_.assign(channels_ordered_by_rel_.begin(), channels_ordered_by_rel_.begin() + info_size_);
    std::sort(info_channels_.begin(), info_channels_.end());
    SetInfoBits(info_channels_);
}

void PolarCode::InitInnerTrellisDecoder() {
#ifndef LLR
    return;
#endif // LLR

    inner_decoder_ = true;
    int trellis_block_size_ = 1 << trellis_n_;
    auto info_it = info_channels_.begin();
    std::vector<PolarCode> codes;
    for (int i = 0; i <= trellis_block_size_; i++)
        codes.push_back(PolarCode(trellis_n_, i, design_epsilon_));
    trellis_decoders_.reserve(block_size_ / trellis_block_size_);

    for (int index = 0; index < block_size_; index += trellis_block_size_) {
        auto info_end = info_it;
        while (info_end != info_channels_.end() && *info_end < index + trellis_block_size_)
            info_end++;
        std::vector<int> trellis_info_bits(info_it, info_end);
        info_it = info_end;

        int info_size = trellis_info_bits.size();
        if (info_size == 0) {
            trellis_decoders_.push_back(std::nullopt);
            continue;
        }
        for (auto& channel_id : trellis_info_bits) {
            channel_id -= index;
        }
        codes[info_size].SetInfoBits(trellis_info_bits);
        std::vector<uint8_t> matrix_gen_bits;
        matrix temp_matrix(info_size);
        for (int i = 0; i < info_size; i++) {
            matrix_gen_bits.assign(info_size, 0);
            matrix_gen_bits[i] = 1;
            codes[info_size].Encode(matrix_gen_bits, temp_matrix[i]);
        }
        trellis_decoders_.push_back(ViterbiSoftDecoder(temp_matrix));
    }
}

void PolarCode::Encode(const std::vector<uint8_t>& info_bits, std::vector<uint8_t>& codeword) {
    codeword.assign(block_size_, 0);

    for (int i = 0; i < info_size_; i++) {
        codeword[info_channels_[i]] = info_bits[i];
    }

    info_bits_ = std::move(info_bits);
    bits_padded_ = codeword;

    for (size_t iteration = 0; iteration < n_pow_; iteration++) {
        size_t increment = 1 << iteration;
        for (size_t j = 0; j < increment; j += 1) {
            for (size_t i = 0; i < block_size_; i += 2 * increment) {
                codeword[i + j] = codeword[i + j] ^ codeword[i + j + increment];
            }
        }
    }
    BitReverseArray(codeword, bit_rev_sorted_ids_);
    codeword_ = codeword;

}

void PolarCode::Decode(const std::vector<double>& p1, const std::vector<double>& p0, size_t list_size, std::vector<uint8_t>& info_bits) {
    list_size_ = list_size;

    initializeDataStructures();

    int l = assignInitialPath();

    auto& p_0 = getArrayPointer_P(0, l);

    for (int beta = 0; beta < block_size_; beta++) {
        p_0[beta][0] = p0[beta];
        p_0[beta][1] = p1[beta];
    }

    std::vector<uint8_t> codeword;
    DecodeSCL(codeword);
    GetInfoBitsFromCodeword(codeword, info_bits);
}

void PolarCode::Decode(const std::vector<std::array<double, 2>>& probabilities, size_t list_size, std::vector<uint8_t>& info_bits) {
    list_size_ = list_size;

    initializeDataStructures();
    int l = assignInitialPath();
    auto& p_0 = getArrayPointer_P(0, l);

    if (probabilities.size() != block_size_) {
        throw std::exception("Decoding wrong block size");
    }
    p_0 = probabilities;

    DecodeSCL(codeword_buf_);
    GetInfoBitsFromCodeword(codeword_buf_, info_bits);
}

void PolarCode::Decode(const std::vector<double>& llrs, size_t list_size, std::vector<uint8_t>& info_bits) {
    list_size_ = list_size;

    initializeDataStructures();
    int l = assignInitialPath();
    auto& llr_0 = getArrayPointer_LLR(0, l);

    if (llrs.size() != block_size_) {
        throw std::exception("Decoding wrong block size");
    }
    llr_0 = llrs;

    DecodeSCL(codeword_buf_);
    GetInfoBitsFromCodeword(codeword_buf_, info_bits);
}

void PolarCode::DecodeSCL(std::vector<uint8_t>& codeword) {
    for (size_t phi = 0; phi < block_size_; phi++) {
#ifdef LLR
        recursivelyCalcLLR(n_pow_, phi);
#else
        recursivelyCalcP(n_pow_, phi);
#endif
        if (is_frozen_bit_[phi] == 1)
            continuePaths_FrozenBit(phi);
        else
            continuePaths_UnfrozenBit(phi);

        if ((phi % 2) == 1)
            recursivelyUpdateC(n_pow_, phi);
    }
    auto l = findMostProbablePath();

    auto& probable_padded_bits = deduced_padded_bits_[l];
    auto& most_probable_path_array = getArrayPointer_C(0, l);

    codeword.resize(block_size_);
    for (int i = 0; i < block_size_; i++) {
        codeword[i] = most_probable_path_array[i][0];
    }
    std::vector<uint8_t> c0_selected(codeword);
    for (int iteration = n_pow_ - 1; iteration >= 0; iteration--) {
        size_t increment = 1 << iteration;
        for (int j = 0; j < increment; j += 1) {
            for (int i = 0; i < block_size_; i += 2 * increment) {
                c0_selected[i + j] = c0_selected[i + j] ^ c0_selected[i + j + increment];
            }
        }
    }
    BitReverseArray(c0_selected, bit_rev_sorted_ids_);

    if (c0_selected != probable_padded_bits) {
        std::cout << "Calculated padded bits not equal to real\n";
    }

    std::vector<uint8_t> decoded_info_bits(info_size_);
    for (int beta = 0; beta < info_size_; beta++) {
        decoded_info_bits[beta] = probable_padded_bits[info_channels_[beta]];
    }

    if (probable_padded_bits != bits_padded_ && info_bits_ == decoded_info_bits) {
        std::cout << "Info bits OK but padded bits differ\n";
    }

    if (codeword != codeword_ && info_bits_ == decoded_info_bits) {
        std::cout << "Codeword is wrong\n";
    }

    if (codeword == codeword_ && info_bits_ != decoded_info_bits) {
        std::cout << "Codeword is right but decoding is wrong\n";
    }
}

void PolarCode::GetInfoBitsFromCodeword(const std::vector<uint8_t>& codeword, std::vector<uint8_t>& info_bits) {
    std::vector<uint8_t> c0_selected(codeword);
    for (int iteration = n_pow_ - 1; iteration >= 0; iteration--) {
        size_t increment = 1 << iteration;
        for (int j = 0; j < increment; j += 1) {
            for (int i = 0; i < block_size_; i += 2 * increment) {
                c0_selected[i + j] = c0_selected[i + j] ^ c0_selected[i + j + increment];
            }
        }
    }
    BitReverseArray(c0_selected, bit_rev_sorted_ids_);
    info_bits.resize(info_size_);
    for (int beta = 0; beta < info_size_; beta++)
        info_bits[beta] = c0_selected[info_channels_[beta]];
}

void PolarCode::initializeDataStructures() {

    inactive_path_indices_.resize(list_size_);
    std::iota(inactive_path_indices_.begin(), inactive_path_indices_.end(), 0);
    active_path_.assign(list_size_, 0);
    path_metric_.assign(list_size_, 0);

    arrays_P_.resize(n_pow_ + 1);
    arrays_C_.resize(n_pow_ + 1);
    arrays_LLR_.resize(n_pow_ + 1);
    pathIndex_to_arrayIndex_.resize(n_pow_ + 1);
    array_references_count_.resize(n_pow_ + 1);
    inactive_array_indexes_.resize(n_pow_ + 1);

    for (int i = 0; i < n_pow_ + 1; i++) {
        pathIndex_to_arrayIndex_[i].resize(list_size_);
        array_references_count_[i].resize(list_size_);
        inactive_array_indexes_[i].resize(list_size_);
        std::iota(inactive_array_indexes_[i].begin(), inactive_array_indexes_[i].end(), 0);

        arrays_P_[i].resize(list_size_);
        arrays_C_[i].resize(list_size_);
        arrays_LLR_[i].resize(list_size_);
        for (int l = 0; l < list_size_; l++) {
            arrays_P_[i][l].resize(1 << (n_pow_ - i));
            arrays_C_[i][l].resize(1 << (n_pow_ - i));
            arrays_LLR_[i][l].resize(1 << (n_pow_ - i));
        }
    }

    deduced_padded_bits_.resize(list_size_, std::vector<unsigned char>(block_size_));
}

size_t PolarCode::assignInitialPath() {
    int l = stack_pop(inactive_path_indices_);
    active_path_[l] = true;
    for (int lambda = 0; lambda < n_pow_ + 1; lambda++) {
        int s = stack_pop(inactive_array_indexes_[lambda]);
        pathIndex_to_arrayIndex_[lambda][l] = s;
        array_references_count_[lambda][s] = 1;
    }
    return l;
}

size_t PolarCode::clonePath(size_t l) {
    int l_p = stack_pop(inactive_path_indices_);
    active_path_[l_p] = true;
    path_metric_[l_p] = path_metric_[l];

    for (int lambda = 0; lambda < n_pow_ + 1; lambda++) {
        int s = pathIndex_to_arrayIndex_[lambda][l];
        pathIndex_to_arrayIndex_[lambda][l_p] = s;
        array_references_count_[lambda][s]++;
    }
    return l_p;
}

void PolarCode::killPath(size_t l) {
    if (!active_path_[l]) {
        std::cerr << "Inactive path is being killed\n";
    }
    active_path_[l] = false;
    inactive_path_indices_.push_back(l);
    path_metric_[l] = 0;

    for (int lambda = 0; lambda < n_pow_ + 1; lambda++) {
        size_t s = pathIndex_to_arrayIndex_[lambda][l];
        array_references_count_[lambda][s]--;
        if (array_references_count_[lambda][s] == 0) {
            inactive_array_indexes_[lambda].push_back(s);
        }
    }
}

size_t PolarCode::GetArrayPointerIndex(size_t lambda, size_t l) {
    size_t s = pathIndex_to_arrayIndex_[lambda][l];
    if (array_references_count_[lambda][s] == 1) {
        return s;
    }
    else {
        size_t s_p = stack_pop(inactive_array_indexes_[lambda]);

#ifdef LLR
        arrays_LLR_[lambda][s_p] = arrays_LLR_[lambda][s];
#else
        arrays_P_[lambda][s_p] = arrays_P_[lambda][s];
#endif
        arrays_C_[lambda][s_p] = arrays_C_[lambda][s];

        array_references_count_[lambda][s]--;
        array_references_count_[lambda][s_p] = 1;
        pathIndex_to_arrayIndex_[lambda][l] = s_p;
        return s_p;
    }
}

PolarCode::PArrayType& PolarCode::getArrayPointer_P(size_t lambda, size_t l) {
    return arrays_P_[lambda][GetArrayPointerIndex(lambda, l)];
}

PolarCode::CArrayType& PolarCode::getArrayPointer_C(size_t lambda, size_t l) {
    return arrays_C_[lambda][GetArrayPointerIndex(lambda, l)];
}

std::vector<double>& PolarCode::getArrayPointer_LLR(size_t lambda, size_t l) {
    return arrays_LLR_[lambda][GetArrayPointerIndex(lambda, l)];
}

void PolarCode::recursivelyCalcP(size_t lambda, size_t phi) {
    if (lambda == 0) {
        return;
    }
    if (phi % 2 == 0) {
        recursivelyCalcP(lambda - 1, phi / 2);
    }

    size_t psi = phi / 2;
    double sigma = 0;
    for (int l = 0; l < list_size_; l++) {
        if (!active_path_[l])
            continue;
        auto& p_lambda = getArrayPointer_P(lambda, l);
        auto& p_lambda_1 = getArrayPointer_P(lambda - 1, l);
        for (int beta = 0; beta < (1 << (n_pow_ - lambda)); beta++) {
            if (phi % 2 == 0) {
                p_lambda[beta][0] = 0.5 * (p_lambda_1[2 * beta][0] * p_lambda_1[2 * beta + 1][0]
                    + p_lambda_1[2 * beta][1] * p_lambda_1[2 * beta + 1][1]);
                p_lambda[beta][1] = 0.5 * (p_lambda_1[2 * beta][1] * p_lambda_1[2 * beta + 1][0]
                    + p_lambda_1[2 * beta][0] * p_lambda_1[2 * beta + 1][1]);
            }
            else {
                auto& c_lambda = getArrayPointer_C(lambda, l);
                int u_p = c_lambda[beta][0];
                p_lambda[beta][0] = 0.5 * p_lambda_1[2 * beta][u_p] * p_lambda_1[2 * beta + 1][0];
                p_lambda[beta][1] = 0.5 * p_lambda_1[2 * beta][u_p ^ 1] * p_lambda_1[2 * beta + 1][1];
            }
            sigma = std::max(sigma, p_lambda[beta][0]);
            sigma = std::max(sigma, p_lambda[beta][1]);
        }
    }

    for (int l = 0; l < list_size_; l++) {
        if (!active_path_[l])
            continue;
        auto& p_lambda = getArrayPointer_P(lambda, l);
        for (int beta = 0; beta < (1 << (n_pow_ - lambda)); beta++) {
            p_lambda[beta][0] /= sigma;
            p_lambda[beta][1] /= sigma;
        }
    }
}

void PolarCode::recursivelyCalcLLR(size_t lambda, size_t phi) {
    if (lambda == 0) {
        return;
    }
    if (phi % 2 == 0) {
        recursivelyCalcLLR(lambda - 1, phi / 2);
    }

    if (inner_decoder_ && lambda < trellis_n_) {
        return;
    }

    /// llr как log(p[0] / p[1])

    size_t psi = phi / 2;
    for (int l = 0; l < list_size_; l++) {
        if (!active_path_[l])
            continue;
        auto& llr_lambda = getArrayPointer_LLR(lambda, l);
        auto& llr_lambda_1 = getArrayPointer_LLR(lambda - 1, l);

        for (int beta = 0; beta < (1 << (n_pow_ - lambda)); beta++) {
            if (phi % 2 == 0) {
                //double upper = std::exp(llr_lambda_1[2 * beta] + llr_lambda_1[2 * beta + 1]) + 1;
                //double lower = std::exp(llr_lambda_1[2 * beta]) + std::exp(llr_lambda_1[2 * beta + 1]);
                //llr_lambda[beta] = std::log(upper / lower);
                int sign_lambda_a_b = (llr_lambda_1[2 * beta] < 0) == (llr_lambda_1[2 * beta + 1] < 0) ? 1 : -1;
                llr_lambda[beta] = sign_lambda_a_b * std::min(std::abs(llr_lambda_1[2 * beta]), std::abs(llr_lambda_1[2 * beta + 1]));
            }
            else {
                auto& c_lambda = getArrayPointer_C(lambda, l);
                auto u_p = c_lambda[beta][0];
                llr_lambda[beta] = (1 - 2 * u_p) * llr_lambda_1[2 * beta] + llr_lambda_1[2 * beta + 1];
            }

        }
    }
}

void PolarCode::recursivelyUpdateC(size_t lambda, size_t phi) {
    size_t psi = phi / 2;
    for (int l = 0; l < list_size_; l++) {
        if (!active_path_[l])
            continue;
        auto& c_lambda = getArrayPointer_C(lambda, l);
        auto& c_lambda_1 = getArrayPointer_C(lambda - 1, l);
        for (int beta = 0; beta < (1 << (n_pow_ - lambda)); beta++) {
            c_lambda_1[2 * beta][psi % 2] = c_lambda[beta][0] ^ c_lambda[beta][1];
            c_lambda_1[2 * beta + 1][psi % 2] = c_lambda[beta][1];
        }
    }
    if (psi % 2 == 1) {
        recursivelyUpdateC(lambda - 1, psi);
    }
}

void PolarCode::continuePaths_FrozenBit(size_t phi) {
    for (int l = 0; l < list_size_; l++) {
        if (!active_path_[l])
            continue;
        auto& c_m = getArrayPointer_C(n_pow_, l);
        c_m[0][phi % 2] = 0;
        deduced_padded_bits_[l][phi] = 0;
#ifdef LLR
        auto& llr_p = getArrayPointer_LLR(n_pow_, l);
        path_metric_[l] += std::log(1 + std::exp(-llr_p[0]));
        //path_metric_[l] += CalcMetric(llr_p[0], 0);
#endif
    }
}

void PolarCode::PopulateContForks() {
    size_t rho = std::min(list_size_, probabilities.size());

    if (rho != probabilities.size()) {
        std::nth_element(probabilities.begin(), probabilities.begin() + rho, probabilities.end(), std::greater<>());
    }

    for (int i = 0; i < rho; i++) {
        int fork_id = probabilities[i].second;
        contForks[fork_id / 2][fork_id % 2] = true;
    }
}

void PolarCode::PopulateContForksLLR() {
    size_t rho = std::min(list_size_, probabilities_LLR_.size());

    if (rho != probabilities_LLR_.size()) {
        std::nth_element(probabilities_LLR_.begin(), probabilities_LLR_.begin() + rho, probabilities_LLR_.end());
    }

    for (int i = 0; i < rho; i++) {
        int fork_id = probabilities_LLR_[i].second;
        contForks[fork_id / 2][fork_id % 2] = true;
    }
}

void PolarCode::continuePaths_UnfrozenBit(size_t phi) {
    contForks.assign(list_size_, { false, false });
    probabilities.clear();
    probabilities_LLR_.clear();

    for (int l = 0; l < list_size_; l++) {
        if (active_path_[l]) {
#ifdef LLR
            auto& llr_p = getArrayPointer_LLR(n_pow_, l);
            probabilities_LLR_.emplace_back(path_metric_[l] + std::log(1 + std::exp(-llr_p[0])), l * 2);
            probabilities_LLR_.emplace_back(path_metric_[l] + std::log(1 + std::exp(+llr_p[0])), l * 2 + 1);
#else
            auto& p_m = getArrayPointer_P(n_pow_, l);
            probabilities.emplace_back(p_m[0][0], l * 2);
            probabilities.emplace_back(p_m[0][1], l * 2 + 1);
#endif
        }
    }

#ifdef LLR
    PopulateContForksLLR();
#else
    PopulateContForks();
#endif

    for (int l = 0; l < list_size_; l++) {
        if (!active_path_[l])
            continue;

        if (!contForks[l][0] && !contForks[l][1]) {
            killPath(l);
        }
    }

    for (int l = 0; l < list_size_; l++) {
        if (!active_path_[l])
            continue;

        auto& c_m = getArrayPointer_C(n_pow_, l);
        if (contForks[l][0] == 1 && contForks[l][1] == 1) {
            c_m[0][phi % 2] = 0;
            size_t l_p = clonePath(l);
            auto& c_m_second = getArrayPointer_C(n_pow_, l_p);
            c_m_second[0][phi % 2] = 1;

#ifdef LLR
            auto& llr_p = getArrayPointer_LLR(n_pow_, l);
            //path_metric_[l] += CalcMetric(llr_p[0], 0);
            //path_metric_[l_p] += CalcMetric(llr_p[0], 1);
            path_metric_[l] += std::log(1 + std::exp(-llr_p[0]));
            path_metric_[l_p] += std::log(1 + std::exp(llr_p[0]));
#endif

            deduced_padded_bits_[l_p] = deduced_padded_bits_[l];
            deduced_padded_bits_[l][phi] = 0;
            deduced_padded_bits_[l_p][phi] = 1;
        }
        else {
            if (contForks[l][0] == 1) {
                c_m[0][phi % 2] = 0;
                deduced_padded_bits_[l][phi] = 0;
#ifdef LLR
                auto& llr_p = getArrayPointer_LLR(n_pow_, l);
                //path_metric_[l] += CalcMetric(llr_p[0], 0);
                path_metric_[l] += std::log(1 + std::exp(-llr_p[0]));
#endif
            }
            else {
                c_m[0][phi % 2] = 1;
                deduced_padded_bits_[l][phi] = 1;
#ifdef LLR
                auto& llr_p = getArrayPointer_LLR(n_pow_, l);
                //path_metric_[l] += CalcMetric(llr_p[0], 1);
                path_metric_[l] += std::log(1 + std::exp(llr_p[0]));
#endif
            }
        }
    }

}

void PolarCode::continuePaths_Trellis(size_t phi) {
    auto& current_trellis = trellis_decoders_[phi >> trellis_n_];
    std::vector<float> calc_words(1 << trellis_n_);
    std::vector<uint8_t> deduced_items;
    for (int l = 0; l < list_size_; l++) {
        auto& llrs = getArrayPointer_LLR(n_pow_ - trellis_n_, l);
        for (int i = 0; i < (1 << trellis_n_); i++) {
            calc_words[i] = -llrs[i];
        }
        current_trellis->DecodeInputToCodeword(calc_words, deduced_items);
        
        auto start = (phi >> trellis_n_) << trellis_n_;
        for (int i = start; i <= phi; i++) {
            deduced_padded_bits_[l][i] = deduced_items[i - start];
        }
    }
}

size_t PolarCode::findMostProbablePath() {
    int l_p = 0;
    double p_p1 = 0;
    double p_llr = std::numeric_limits<double>::max();
    for (int l = 0; l < list_size_; l++) {
        if (active_path_[l]) {
#ifdef LLR
            if (path_metric_[l] < p_llr) {
                p_llr = path_metric_[l];
                l_p = l;
            }
#else
            auto& c_m = getArrayPointer_C(n_pow_, l);
            auto& p_m = getArrayPointer_P(n_pow_, l);
            if (p_p1 > p_m[0][c_m[0][1]]) {
                l_p = l;
                p_p1 = p_m[0][c_m[0][1]];
            }
#endif
        }
    }
    return l_p;
}

void PolarCode::InitBitReversedIds() {
    for (int i = 0; i < block_size_; i++) {
        size_t to_be_reversed = i;
        bit_rev_sorted_ids_[i] = (to_be_reversed & 1) << (n_pow_ - 1);
        for (size_t j = n_pow_ - 1; j > 0; j--) {
            to_be_reversed >>= 1;
            bit_rev_sorted_ids_[i] += (to_be_reversed & 1) << (j - 1);
        }
    }
}