.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_potrf_scratchpad_size:

potrf_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemkl_lapack_potrf` function.

.. container:: section

  .. rubric:: Description
         
``potrf_scratchpad_size`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double`` 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_potrf` function should be able to hold.
Calls to this routine must specify the template parameter explicitly.

potrf_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      template <typename T>
      std::int64_t potrf_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t lda) 
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   Device queue where calculations by :ref:`onemkl_lapack_potrf` function will be performed.

upper_lower
   Indicates whether the upper or lower triangular part of :math:`A` is
   stored and how :math:`A` is factored:

   If ``upper_lower = oneapi::mkl::uplo::upper``, the array ``a`` stores the
   upper triangular part of the matrix :math:`A`, and the strictly lower
   triangular part of the matrix is not referenced.

   If ``upper_lower = oneapi::mkl::uplo::lower``, the array ``a`` stores the
   lower triangular part of the matrix :math:`A`, and the strictly upper
   triangular part of the matrix is not referenced.

n
   Specifies the order of the matrix :math:`A` (:math:`0 \le n`).

lda
   The leading dimension of ``a``.

.. container:: section

  .. rubric:: Return Value

The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_potrf` function should be able to hold.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines`


