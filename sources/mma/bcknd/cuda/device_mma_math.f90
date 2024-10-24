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

  public :: device_mma_gensub1, device_mma_gensub2, device_mma_gensub3, device_mma_gensub4, device_mma_max, device_max2, &
  device_rex, device_lcsc2, device_relambda, device_sub2cons2, device_maxval, device_norm, device_delx, device_add2inv2,&
  device_GG, device_diagx, device_bb, device_updatebb, device_AA, device_updateAA, device_dx, device_dy, device_deta,&
  device_dxsi, device_maxval2, device_maxval3, device_kkt_rex

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

subroutine device_mma_max(xsi_d,x_d,alpha_d,n)
    type(c_ptr) :: xsi_d,x_d,alpha_d
    integer :: n
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_mma_max(xsi_d,x_d,alpha_d,n)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_mma_max

subroutine device_max2(a_d, b, c_d, d, n)
    type(c_ptr) :: a_d,c_d
    real(c_rp) :: b, d
    integer :: n
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_max2(a_d, b, c_d, d, n)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_max2



subroutine device_rex(rex_d,  x_d,  low_d, upp_d,  pij_d, p0j_d,qij_d, q0j_d, lambda_d, xsi_d, eta_d, n, m)
    type(c_ptr) :: rex_d,  x_d,  low_d, upp_d,  pij_d, p0j_d,qij_d, q0j_d, lambda_d, xsi_d, eta_d
    integer(c_int) :: n,m
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_rex(rex_d,  x_d,  low_d, upp_d,  pij_d, p0j_d,qij_d, q0j_d, lambda_d, xsi_d, eta_d, n, m)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_rex

function device_lcsc2(a_d, b_d, n) result(res)
    type(c_ptr) :: a_d, b_d
    integer(c_int) :: n
    real(kind=rp) :: res

    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    res= cuda_lcsc2(a_d, b_d, n)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end function device_lcsc2

subroutine device_relambda(relambda_d, x_d,  upp_d, low_d, pij_d, qij_d,  n, m)
    type(c_ptr) :: relambda_d, x_d,  upp_d, low_d, pij_d, qij_d
    integer(c_int) :: n, m
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_relambda(relambda_d, x_d,  upp_d, low_d, pij_d, qij_d,  n, m)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_relambda

subroutine device_sub2cons2(rexsi_d, xsi_d, x_d, alpha_d, epsi,n) 
    type(c_ptr):: rexsi_d,xsi_d,x_d,alpha_d
    real(kind=rp) :: epsi
    integer(c_int) :: n
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_sub2cons2(rexsi_d, xsi_d, x_d, alpha_d, epsi, n) 
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_sub2cons2

function device_maxval(rex_d,n) result(res)
    type(c_ptr):: rex_d
    real(kind=rp) :: res
    integer(c_int) :: n
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    res=cuda_maxval(rex_d,n)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end function device_maxval

function device_norm(rex_d,n) result(res)
    type(c_ptr):: rex_d
    real(kind=rp) :: res
    integer(c_int) :: n
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    res=cuda_norm(rex_d,n)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end function device_norm

subroutine device_delx(delx_d, x_d, low_d, upp_d,  pij_d,  qij_d,  p0j_d, q0j_d, alpha_d,  beta_d, lambda_d, epsi, n, m)
    type(c_ptr):: delx_d, x_d, low_d, upp_d,  pij_d,  qij_d,  p0j_d, q0j_d, alpha_d,  beta_d, lambda_d
    real(kind=rp) :: epsi
    integer(c_int) ::  n, m
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_delx(delx_d, x_d, low_d, upp_d,  pij_d,  qij_d,  p0j_d, q0j_d, alpha_d,  beta_d, lambda_d, epsi, n, m)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_delx

subroutine device_add2inv2(a_d, b_d, c, n)
    type(c_ptr):: a_d, b_d
    real(kind=rp) :: c
    integer(c_int) ::  n
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_add2inv2(a_d, b_d, c, n)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_add2inv2


subroutine device_GG(GG_d,  x_d,  low_d,  upp_d, pij_d, qij_d, n, m)
    type(c_ptr):: GG_d,  x_d,  low_d,  upp_d, pij_d, qij_d
    integer(c_int) ::  n, m
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_GG(GG_d,  x_d,  low_d,  upp_d, pij_d, qij_d, n, m)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_GG

subroutine device_diagx(diagx_d, x_d, xsi_d, low_d, upp_d, p0j_d, q0j_d,  pij_d, qij_d, &
 alpha_d, beta_d,  eta_d, lambda_d, n, m)
type(c_ptr):: diagx_d, x_d, xsi_d, low_d, upp_d, p0j_d, q0j_d,  pij_d, qij_d,  alpha_d, &
beta_d,  eta_d, lambda_d
integer(c_int) ::  n, m
! #if HAVE_HIP
!     call neko_error('no device backend configured')
! #elif HAVE_CUDA
call cuda_diagx(diagx_d, x_d, xsi_d, low_d, upp_d, p0j_d, q0j_d,  pij_d, qij_d, &
 alpha_d, beta_d,  eta_d, lambda_d, n, m)
! #elif HAVE_OPENCL
!     call neko_error('no device backend configured')
! #else
!     call neko_error('no device backend configured')
! #endif
end subroutine device_diagx


subroutine device_bb(bb_d, GG_d, delx_d,diagx_d,n,m) 
    type(c_ptr):: bb_d, GG_d, delx_d,diagx_d
    integer(c_int) ::  n, m
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_bb(bb_d, GG_d, delx_d,diagx_d,n,m)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_bb

subroutine device_updatebb(bb_d, dellambda_d, dely_d, d_d,mu_d, y_d, delz, m)
    type(c_ptr):: bb_d, dellambda_d, dely_d, d_d,mu_d, y_d
    integer(c_int) ::   m
    real(c_rp) :: delz
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_updatebb(bb_d, dellambda_d, dely_d, d_d,mu_d, y_d, delz, m)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_updatebb

subroutine device_AA(AA_d, GG_d,  diagx_d, n, m)
    type(c_ptr):: AA_d, GG_d,  diagx_d
    integer(c_int) ::   n, m
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_AA(AA_d, GG_d,  diagx_d, n, m) 
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_AA

subroutine device_updateAA(AA_d, globaltmp_mm_d, s_d, lambda_d, d_d, mu_d, y_d,a_d, zeta,z, m)
    type(c_ptr):: AA_d, globaltmp_mm_d, s_d, lambda_d, d_d, mu_d, y_d,a_d
    integer(c_int) ::   m
    real(c_rp) ::  zeta,z
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_updateAA(AA_d, globaltmp_mm_d, s_d, lambda_d, d_d, mu_d, y_d,a_d, zeta,z, m)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_updateAA

subroutine device_dx(dx_d, delx_d, diagx_d, GG_d, dlambda_d, n, m)
    type(c_ptr):: dx_d, delx_d, diagx_d, GG_d, dlambda_d
    integer(c_int) ::   n,m
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_dx(dx_d, delx_d, diagx_d, GG_d, dlambda_d, n, m)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_dx
subroutine device_dy(dy_d, dely_d, dlambda_d, d_d, mu_d, y_d,  n)
    type(c_ptr):: dy_d, dely_d, dlambda_d, d_d, mu_d, y_d
    integer(c_int) ::   n
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_dy(dy_d, dely_d, dlambda_d, d_d, mu_d, y_d, n) 
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_dy

subroutine device_dxsi(dxsi_d, xsi_d, dx_d,x_d,alpha_d, epsi, n) 
    type(c_ptr):: dxsi_d, xsi_d, dx_d,x_d,alpha_d
    integer(c_int) ::   n
    real(c_rp) :: epsi
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_dxsi(dxsi_d, xsi_d, dx_d,x_d,alpha_d, epsi, n)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_dxsi
subroutine device_deta(deta_d, eta_d, dx_d,  x_d, beta_d, epsi,n) 
    type(c_ptr):: deta_d, eta_d, dx_d,  x_d, beta_d
    integer(c_int) ::   n
    real(c_rp) :: epsi
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_deta(deta_d, eta_d, dx_d,  x_d, beta_d, epsi,n)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_deta

function device_maxval2(dxx_d, xx_d, cons, n) result(res)
    type(c_ptr):: dxx_d, xx_d
    integer :: n
    real(kind=rp), intent(in) :: cons
    real(kind=rp) :: res

    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    res = cuda_maxval2(dxx_d, xx_d, cons, n)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end function device_maxval2


function device_maxval3(dx_d, x_d, alpha_d,cons, n) result(res)
    type(c_ptr):: dx_d, x_d, alpha_d
    real(kind=rp), intent(in) :: cons
    real(kind=rp) :: res
    integer(c_int) :: n
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    res=cuda_maxval3(dx_d, x_d, alpha_d,cons, n)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end function device_maxval3



subroutine device_kkt_rex(rex_d,  df0dx_d,  dfdx_d, xsi_d, eta_d, lambda_d, n, m)
    type(c_ptr):: rex_d,  df0dx_d,  dfdx_d, xsi_d, eta_d, lambda_d
    integer(c_int) ::   n, m
    ! #if HAVE_HIP
    !     call neko_error('no device backend configured')
    ! #elif HAVE_CUDA
    call cuda_kkt_rex(rex_d,  df0dx_d,  dfdx_d, xsi_d, eta_d, lambda_d, n, m)
    ! #elif HAVE_OPENCL
    !     call neko_error('no device backend configured')
    ! #else
    !     call neko_error('no device backend configured')
    ! #endif
end subroutine device_kkt_rex



end module device_mma_math