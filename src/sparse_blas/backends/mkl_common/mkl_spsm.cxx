/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

// In this file functions and types using the namespace oneapi::mkl::sparse:: refer to the backend's namespace for better readability.

namespace oneapi::mkl::sparse {

struct spsm_descr {
    bool buffer_size_called = false;
    bool optimized_called = false;
    oneapi::mkl::transpose last_optimized_opA;
    matrix_view last_optimized_A_view;
    matrix_handle_t last_optimized_A_handle;
    dense_vector_handle_t last_optimized_x_handle;
    dense_vector_handle_t last_optimized_y_handle;
    spsm_alg last_optimized_alg;
};

} // namespace oneapi::mkl::sparse

namespace oneapi::mkl::sparse::BACKEND {

void init_spsm_descr(sycl::queue& /*queue*/, spsm_descr_t* p_spsm_descr) {
    *p_spsm_descr = new spsm_descr();
}

sycl::event release_spsm_descr(sycl::queue& queue, spsm_descr_t spsm_descr,
                               const std::vector<sycl::event>& dependencies) {
    return detail::submit_release(queue, spsm_descr, dependencies);
}

void check_valid_spsm(const std::string& function_name, oneapi::mkl::transpose opA,
                      matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                      dense_vector_handle_t y_handle, bool is_alpha_host_accessible, spsm_alg alg) {
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    detail::check_valid_spsm_common(function_name, A_view, internal_A_handle, x_handle, y_handle,
                                    is_alpha_host_accessible);

    if (alg == spsm_alg::no_optimize_alg &&
        !internal_A_handle->has_matrix_property(matrix_property::sorted)) {
        throw mkl::unimplemented(
            "sparse_blas", function_name,
            "The backend does not support `no_optimize_alg` unless A_handle has the property `matrix_property::sorted`.");
    }

#if BACKEND == gpu
    detail::data_type data_type = internal_A_handle->get_value_type();
    if ((data_type == detail::data_type::complex_fp32 ||
         data_type == detail::data_type::complex_fp64) &&
        opA == oneapi::mkl::transpose::conjtrans) {
        throw mkl::unimplemented("sparse_blas", function_name,
                                 "The backend does not support spsm using conjtrans.");
    }
#else
    (void)opA;
#endif // BACKEND
}

void spsm_buffer_size(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                      matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                      dense_vector_handle_t y_handle, spsm_alg alg, spsm_descr_t spsm_descr,
                      std::size_t& temp_buffer_size) {
    // TODO: Add support for external workspace once the close-source oneMKL backend supports it.
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    check_valid_spsm(__func__, opA, A_view, A_handle, x_handle, y_handle, is_alpha_host_accessible,
                     alg);
    temp_buffer_size = 0;
    spsm_descr->buffer_size_called = true;
}

inline void common_spsm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                                 matrix_view A_view, matrix_handle_t A_handle,
                                 dense_vector_handle_t x_handle, dense_vector_handle_t y_handle,
                                 spsm_alg alg, spsm_descr_t spsm_descr) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    check_valid_spsm("spsm_optimize", opA, A_view, A_handle, x_handle, y_handle,
                     is_alpha_host_accessible, alg);
    if (!spsm_descr->buffer_size_called) {
        throw mkl::uninitialized("sparse_blas", "spsm_optimize",
                                 "spsm_buffer_size must be called before spsm_optimize.");
    }
    spsm_descr->optimized_called = true;
    spsm_descr->last_optimized_opA = opA;
    spsm_descr->last_optimized_A_view = A_view;
    spsm_descr->last_optimized_A_handle = A_handle;
    spsm_descr->last_optimized_x_handle = x_handle;
    spsm_descr->last_optimized_y_handle = y_handle;
    spsm_descr->last_optimized_alg = alg;
}

void spsm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                   matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                   dense_vector_handle_t y_handle, spsm_alg alg, spsm_descr_t spsm_descr,
                   sycl::buffer<std::uint8_t, 1> /*workspace*/) {
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    if (!internal_A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    common_spsm_optimize(queue, opA, alpha, A_view, A_handle, x_handle, y_handle, alg, spsm_descr);
    if (alg == spsm_alg::no_optimize_alg) {
        return;
    }
    internal_A_handle->can_be_reset = false;
    oneapi::mkl::sparse::optimize_trsv(queue, A_view.uplo_view, opA, A_view.diag_view,
                                       internal_A_handle->backend_handle);
}

sycl::event spsm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                          matrix_view A_view, matrix_handle_t A_handle,
                          dense_vector_handle_t x_handle, dense_vector_handle_t y_handle,
                          spsm_alg alg, spsm_descr_t spsm_descr, void* /*workspace*/,
                          const std::vector<sycl::event>& dependencies) {
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    if (internal_A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    common_spsm_optimize(queue, opA, alpha, A_view, A_handle, x_handle, y_handle, alg, spsm_descr);
    if (alg == spsm_alg::no_optimize_alg) {
        return detail::collapse_dependencies(queue, dependencies);
    }
    internal_A_handle->can_be_reset = false;
    return oneapi::mkl::sparse::optimize_trsv(queue, A_view.uplo_view, opA, A_view.diag_view,
                                              internal_A_handle->backend_handle, dependencies);
}

template <typename T>
sycl::event internal_spsm(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                          matrix_view A_view, matrix_handle_t A_handle,
                          dense_vector_handle_t x_handle, dense_vector_handle_t y_handle,
                          spsm_alg /*alg*/, spsm_descr_t /*spsm_descr*/,
                          const std::vector<sycl::event>& dependencies,
                          bool is_alpha_host_accessible) {
    T host_alpha =
        detail::get_scalar_on_host(queue, static_cast<const T*>(alpha), is_alpha_host_accessible);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    internal_A_handle->can_be_reset = false;
    if (internal_A_handle->all_use_buffer()) {
        oneapi::mkl::sparse::trsv(queue, A_view.uplo_view, opA, A_view.diag_view, host_alpha,
                                  internal_A_handle->backend_handle, x_handle->get_buffer<T>(),
                                  y_handle->get_buffer<T>());
        // Dependencies are not used for buffers
        return {};
    }
    else {
        return oneapi::mkl::sparse::trsv(queue, A_view.uplo_view, opA, A_view.diag_view, host_alpha,
                                         internal_A_handle->backend_handle,
                                         x_handle->get_usm_ptr<T>(), y_handle->get_usm_ptr<T>(),
                                         dependencies);
    }
}

sycl::event spsm(sycl::queue& queue, oneapi::mkl::transpose opA, const void* alpha,
                 matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                 dense_vector_handle_t y_handle, spsm_alg alg, spsm_descr_t spsm_descr,
                 const std::vector<sycl::event>& dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    check_valid_spsm(__func__, opA, A_view, A_handle, x_handle, y_handle, is_alpha_host_accessible,
                     alg);

    if (!spsm_descr->optimized_called) {
        throw mkl::uninitialized("sparse_blas", __func__,
                                 "spsm_optimize must be called before spsm.");
    }
    CHECK_DESCR_MATCH(spsm_descr, opA, "spsm_optimize");
    CHECK_DESCR_MATCH(spsm_descr, A_view, "spsm_optimize");
    CHECK_DESCR_MATCH(spsm_descr, A_handle, "spsm_optimize");
    CHECK_DESCR_MATCH(spsm_descr, x_handle, "spsm_optimize");
    CHECK_DESCR_MATCH(spsm_descr, y_handle, "spsm_optimize");
    CHECK_DESCR_MATCH(spsm_descr, alg, "spsm_optimize");

    auto value_type = detail::get_internal_handle(A_handle)->get_value_type();
    DISPATCH_MKL_OPERATION("spsm", value_type, internal_spsm, queue, opA, alpha, A_view, A_handle,
                           x_handle, y_handle, alg, spsm_descr, dependencies,
                           is_alpha_host_accessible);
}

} // namespace oneapi::mkl::sparse::BACKEND
