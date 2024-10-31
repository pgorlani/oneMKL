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

#include "sparse_blas/backends/mkl_common/mkl_dispatch.hpp"
#include "sparse_blas/backends/mkl_common/mkl_handles.hpp"
#include "sparse_blas/common_op_verification.hpp"
#include "sparse_blas/macros.hpp"
#include "sparse_blas/matrix_view_comparison.hpp"
#include "sparse_blas/sycl_helper.hpp"

#include "oneapi/mkl/sparse_blas/detail/mklgpu/onemkl_sparse_blas_mklgpu.hpp"

#define BACKEND mklgpu

#include "sparse_blas/backends/mkl_common/mkl_spmm.cxx"
#include "sparse_blas/backends/mkl_common/mkl_spmv.cxx"
#include "sparse_blas/backends/mkl_common/mkl_spsv.cxx"
#include "sparse_blas/backends/mkl_common/mkl_spsm.cxx"

#undef BACKEND
