/*******************************************************************************
* Copyright 2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#include <complex>
#include <iostream>
#include <vector>

#include "test_spsm.hpp"

extern std::vector<sycl::device*> devices;

namespace {

template <typename fpType, typename intType>
int test_spsm(sycl::device* dev, sycl::property_list queue_properties,
              sparse_matrix_format_t format,
              intType m,
              intType cols,
              double density_A_matrix,
              oneapi::mkl::index_base index,
              oneapi::mkl::layout dense_matrix_layout,
              oneapi::mkl::transpose transpose_A,
              oneapi::mkl::transpose transpose_X,
              fpType alpha,
              oneapi::mkl::sparse::spsm_alg alg, oneapi::mkl::sparse::matrix_view A_view,
              const std::set<oneapi::mkl::sparse::matrix_property>& matrix_properties,
              bool reset_data, bool test_scalar_on_device) {
    if (test_scalar_on_device) {
        // Scalars on the device is not planned to be supported with the buffer API
        return 1;
    }

    sycl::queue main_queue(*dev, exception_handler_t(), queue_properties);

    intType ldx = cols;
    intType ldy = cols;

    intType indexing = (index == oneapi::mkl::index_base::zero) ? 0 : 1;
    const std::size_t mu = static_cast<std::size_t>(m * cols);
    const bool is_symmetric =
        matrix_properties.find(oneapi::mkl::sparse::matrix_property::symmetric) !=
        matrix_properties.cend();

    // Use a fixed seed for operations very sensitive to the input data
    std::srand(1);

    // Input matrix
    std::vector<intType> ia_host, ja_host;
    std::vector<fpType> a_host;
    // Set non-zero values to the diagonal, except if the matrix is viewed as a unit matrix.
    const bool require_diagonal =
        !(A_view.type_view == oneapi::mkl::sparse::matrix_descr::diagonal &&
          A_view.diag_view == oneapi::mkl::diag::unit);
    intType nnz =
        generate_random_matrix<fpType, intType>(format, m, m, density_A_matrix, indexing, ia_host,
                                                ja_host, a_host, is_symmetric, require_diagonal);

    // Input dense vector.
    // The input `x` is initialized to random values on host and device.
    std::vector<fpType> x_host(mu, 1);
    //rand_vector(x_host, mu);

    // Output and reference dense vectors.
    // They are both initialized with a dummy value to catch more errors.
    std::vector<fpType> y_host(mu, -2.0f);
    std::vector<fpType> y_ref_host(y_host);

    // Shuffle ordering of column indices/values to test sortedness
    shuffle_sparse_matrix_if_needed(format, matrix_properties, indexing, ia_host.data(),
                                    ja_host.data(), a_host.data(), nnz, static_cast<std::size_t>(m));

#if 1
    auto ia_buf = make_buffer(ia_host);
    auto ja_buf = make_buffer(ja_host);
    auto a_buf = make_buffer(a_host);
    auto x_buf = make_buffer(x_host);
    auto y_buf = make_buffer(y_host);

    oneapi::mkl::sparse::matrix_handle_t A_handle = nullptr;
    oneapi::mkl::sparse::dense_matrix_handle_t X_handle = nullptr;
    oneapi::mkl::sparse::dense_matrix_handle_t Y_handle = nullptr;
    oneapi::mkl::sparse::spsm_descr_t descr = nullptr;
    try {
        init_sparse_matrix(main_queue, format, &A_handle, m, m, nnz, index, ia_buf, ja_buf, a_buf);
        for (auto property : matrix_properties) {
            CALL_RT_OR_CT(oneapi::mkl::sparse::set_matrix_property, main_queue, A_handle, property);
        }
        CALL_RT_OR_CT(oneapi::mkl::sparse::init_dense_matrix, main_queue, &X_handle, m,
                      cols, ldx, dense_matrix_layout, x_buf);
        CALL_RT_OR_CT(oneapi::mkl::sparse::init_dense_matrix, main_queue, &Y_handle,
                      static_cast<std::int64_t>(m), cols, ldy, dense_matrix_layout,
                      y_buf);

        CALL_RT_OR_CT(oneapi::mkl::sparse::init_spsm_descr, main_queue, &descr);

        std::size_t workspace_size = 0;
        CALL_RT_OR_CT(oneapi::mkl::sparse::spsm_buffer_size, main_queue, transpose_A, transpose_X, &alpha,
                      A_view, A_handle, X_handle, Y_handle, alg, descr, workspace_size);
        sycl::buffer<std::uint8_t, 1> workspace_buf((sycl::range<1>(workspace_size)));

        CALL_RT_OR_CT(oneapi::mkl::sparse::spsm_optimize, main_queue, transpose_A, transpose_X, &alpha,
                      A_view, A_handle, X_handle, Y_handle, alg, descr, workspace_buf);

        CALL_RT_OR_CT(oneapi::mkl::sparse::spsm, main_queue, transpose_A, transpose_X, &alpha,
                      A_view, A_handle, X_handle, Y_handle, alg, descr);
 
        if (reset_data) {
            intType reset_nnz = generate_random_matrix<fpType, intType>(
                format, m, m, density_A_matrix, indexing, ia_host, ja_host, a_host, is_symmetric,
                require_diagonal);
            shuffle_sparse_matrix_if_needed(format, matrix_properties, indexing, ia_host.data(),
                                            ja_host.data(), a_host.data(), reset_nnz, static_cast<std::size_t>(m));
            if (reset_nnz > nnz) {
                ia_buf = make_buffer(ia_host);
                ja_buf = make_buffer(ja_host);
                a_buf = make_buffer(a_host);
            }
            else {
                copy_host_to_buffer(main_queue, ia_host, ia_buf);
                copy_host_to_buffer(main_queue, ja_host, ja_buf);
                copy_host_to_buffer(main_queue, a_host, a_buf);
            }
            copy_host_to_buffer(main_queue, y_ref_host, y_buf);
            nnz = reset_nnz;
            set_matrix_data(main_queue, format, A_handle, m, m, nnz, index, ia_buf, ja_buf, a_buf);

            std::size_t workspace_size_2 = 0;
            CALL_RT_OR_CT(oneapi::mkl::sparse::spsm_buffer_size, main_queue, transpose_A, transpose_X, &alpha,
                      A_view, A_handle, X_handle, Y_handle, alg, descr, workspace_size);
            if (workspace_size_2 > workspace_size) {
                workspace_buf = sycl::buffer<std::uint8_t, 1>((sycl::range<1>(workspace_size_2)));
            }

            std::cerr<<__LINE__<<std::endl;
            CALL_RT_OR_CT(oneapi::mkl::sparse::spsm_optimize, main_queue, transpose_A, transpose_X, &alpha,
                      A_view, A_handle, X_handle, Y_handle, alg, descr, workspace_buf);
 
            CALL_RT_OR_CT(oneapi::mkl::sparse::spsm, main_queue, transpose_A, transpose_X, &alpha,
                      A_view, A_handle, X_handle, Y_handle, alg, descr);
        }
    }
    catch (const sycl::exception& e) {
        std::cout << "Caught synchronous SYCL exception during sparse SPSV:\n"
                  << e.what() << std::endl;
        print_error_code(e);
        return 0;
    }
    catch (const oneapi::mkl::unimplemented& e) {
        wait_and_free_handles(main_queue, A_handle, X_handle, Y_handle);
        if (descr) {
            sycl::event ev_release_descr;
            CALL_RT_OR_CT(ev_release_descr = oneapi::mkl::sparse::release_spsm_descr, main_queue,
                          descr);
            ev_release_descr.wait();
        }
        return test_skipped;
    }
    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of sparse SPSV:\n" << error.what() << std::endl;
        return 0;
    }
    CALL_RT_OR_CT(oneapi::mkl::sparse::release_spsm_descr, main_queue, descr);
    free_handles(main_queue, A_handle, X_handle, Y_handle);

    // Compute reference.
//    prepare_reference_spsm_data(format, ia_host.data(), ja_host.data(), a_host.data(), m, nnz,
//                                indexing, transpose_A, x_host.data(), alpha, A_view,
//                                y_ref_host.data());

    // Compare the results of reference implementation and DPC++ implementation.
    auto y_acc = y_buf.get_host_access(sycl::read_only);
    bool valid = check_equal_vector(y_acc, /*y_ref*/x_host);

    return static_cast<int>(valid);
#endif
    return 1;
}

class parseSpsmBufferTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(parseSpsmBufferTests, RealSinglePrecision) {
    using fpType = float;
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spsm<fpType, int32_t>, test_spsm<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::nontrans, num_passed, num_skipped);
    test_helper<fpType>(test_spsm<fpType, int32_t>, test_spsm<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::trans, num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

TEST_P(parseSpsmBufferTests, RealDoublePrecision) {
    using fpType = double;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spsm<fpType, int32_t>, test_spsm<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::nontrans, num_passed, num_skipped);
    test_helper<fpType>(test_spsm<fpType, int32_t>, test_spsm<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::trans, num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

TEST_P(parseSpsmBufferTests, ComplexSinglePrecision) {
    using fpType = std::complex<float>;
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spsm<fpType, int32_t>, test_spsm<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::nontrans, num_passed, num_skipped);
    test_helper<fpType>(test_spsm<fpType, int32_t>, test_spsm<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::trans, num_passed, num_skipped);
    test_helper<fpType>(test_spsm<fpType, int32_t>, test_spsm<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::conjtrans, num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

TEST_P(parseSpsmBufferTests, ComplexDoublePrecision) {
    using fpType = std::complex<double>;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spsm<fpType, int32_t>, test_spsm<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::nontrans, num_passed, num_skipped);
    test_helper<fpType>(test_spsm<fpType, int32_t>, test_spsm<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::trans, num_passed, num_skipped);
    test_helper<fpType>(test_spsm<fpType, int32_t>, test_spsm<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::conjtrans, num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

INSTANTIATE_TEST_SUITE_P(SparseSpsvBufferTestSuite, parseSpsmBufferTests,
                         testing::ValuesIn(devices), ::DeviceNamePrint());

} // anonymous namespace
