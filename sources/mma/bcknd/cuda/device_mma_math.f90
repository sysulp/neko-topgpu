! Copyright (c) 2021-2024, The Neko Authors
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!
!   * Redistributions of source code must retain the above copyright
!     notice, this list of conditions and the following disclaimer.
!
!   * Redistributions in binary form must reproduce the above
!     copyright notice, this list of conditions and the following
!     disclaimer in the documentation and/or other materials provided
!     with the distribution.
!
!   * Neither the name of the authors nor the names of its
!     contributors may be used to endorse or promote products derived
!     from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
! FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
! COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
! INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
! LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
! ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.
!
module device_mma_math
  use, intrinsic :: iso_c_binding, only: c_ptr, c_int
  use num_types, only: rp, c_rp
  use utils, only: neko_error
  use comm, only: NEKO_COMM, pe_size, MPI_REAL_PRECISION
  use mpi_f08, only: MPI_SUM, MPI_IN_PLACE, MPI_Allreduce
  use neko_config
  use cuda_mma_math

  implicit none
  private

  public :: device_mma_gensub1, device_mma_gensub2, device_mma_gensub3, device_mma_gensub4

  contains

  subroutine device_mma_gensub1(low_d, upp_d,x_d, xmin_d, xmax_d, asyinit, n)
    type(c_ptr) :: low_d, upp_d,x_d, xmin_d, xmax_d
    real(c_rp) :: asyinit
    integer :: n
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call mma_gensub1_cuda(low_d, upp_d,x_d, xmin_d, xmax_d, asyinit, n)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_mma_gensub1


subroutine device_mma_gensub2(low_d, upp_d, x_d, xold1_d, xold2_d,xmin_d, xmax_d, asydecr, asyincr, n)
    type(c_ptr) :: low_d, upp_d, x_d, xold1_d, xold2_d,xmin_d, xmax_d
    real(c_rp) :: asydecr, asyincr
    integer :: n
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call mma_gensub2_cuda(low_d, upp_d, x_d, xold1_d, xold2_d,xmin_d, xmax_d, asydecr, asyincr, n)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_mma_gensub2

subroutine device_mma_gensub3(x_d, df0dx_d, dfdx_d,low_d, upp_d, min_d, max_d,alpha_d, beta_d, p0j_d, q0j_d, pij_d, qij_d, n, m)
    type(c_ptr) :: x_d, df0dx_d, dfdx_d,low_d, upp_d, min_d, max_d,alpha_d, beta_d, p0j_d, q0j_d, pij_d, qij_d
    integer :: n, m
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call mma_gensub3_cuda(x_d, df0dx_d, dfdx_d,low_d, upp_d, min_d,max_d,alpha_d, beta_d, p0j_d, q0j_d, pij_d, qij_d, n, m)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_mma_gensub3 

subroutine device_mma_gensub4(x_d, low_d, upp_d, pij_d, qij_d, n, m, bi_d)
    type(c_ptr) :: x_d, low_d, upp_d, pij_d, qij_d, bi_d
    integer :: n, m
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call mma_gensub4_cuda(x_d, low_d, upp_d, pij_d, qij_d, n, m, bi_d)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_mma_gensub4




end module device_mma_math