submodule (mma) mma_gpu
use mpi_f08, only: MPI_INTEGER, MPI_REAL, mpi_sum, mpi_min, mpi_max, &
MPI_Allreduce
use utils, only: neko_error
use comm, only: neko_comm, mpi_real_precision
use device_math
use device_mma_math
use vector, only: vector_t
use matrix, only: matrix_t
use device

contains
subroutine mma_gensub_gpu(this, iter, x, df0dx, fval, dfdx)
  ! ----------------------------------------------------- !
  ! Generate the approximation sub problem by computing   !
  ! the lower and upper asymtotes and the other necessary !
  ! parameters (alpha, beta, p0j, q0j, pij, qij, ...).    !
  ! ----------------------------------------------------- !
  class(mma_t), intent(inout) :: this
  type(vector_t), intent(in) :: x
  type(vector_t), intent(in) :: df0dx
  type(vector_t), intent(in) :: fval
  type(matrix_t), intent(in) :: dfdx

  integer, intent(in) :: iter
  integer :: i, j, ierr
  type(vector_t) :: globaltmp_m
  real(kind=rp) :: cons


  call globaltmp_m%init(this%m)
  if (iter .lt. 3) then
    call device_add3s2(this%low%x_d,this%xmax%x_d,this%xmin%x_d,-this%asyinit,this%asyinit,this%n)
    call device_add2(this%low%x_d,x%x_d,this%n)

    call device_add3s2( this%upp%x_d,this%xmax%x_d,this%xmin%x_d,this%asyinit,- this%asyinit,this%n)
    call device_add2(this%upp%x_d,x%x_d,this%n)
    
    !!!!!equal!!!!
    !call device_mma_gensub1(this%low%x_d, this%upp%x_d,x%x_d, this%xmin%x_d, this%xmax%x_d, this%asyinit, this%n)
    !!!!

  else
   call device_mma_gensub2(this%low%x_d, this%upp%x_d, x%x_d, this%xold1%x_d, this%xold2%x_d,this%xmin%x_d, this%xmax%x_d, &
     this%asydecr, this%asyincr, this%n)
 end if
 call device_mma_gensub3(x%x_d, df0dx%x_d, dfdx%x_d,this%low%x_d, this%upp%x_d, this%xmin%x_d, this%xmax%x_d,this%alpha%x_d, &
   this%beta%x_d, this%p0j%x_d, this%q0j%x_d, this%pij%x_d, this%qij%x_d, this%n, this%m) 
 call device_mma_gensub4(x%x_d, this%low%x_d, this%upp%x_d, this%pij%x_d, this%qij%x_d, this%n, this%m, this%bi%x_d)

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !!!!!cpu gpu transfer part
 globaltmp_m%x=0.0_rp
 call device_memcpy_r1(this%bi%x, this%bi%x_d, this%m, DEVICE_TO_HOST, sync=.false.)
 call MPI_Allreduce(this%bi%x, globaltmp_m%x, this%m, &
   mpi_real_precision, mpi_sum, neko_comm, ierr)
 call device_memcpy_r1(globaltmp_m%x, globaltmp_m%x_d, this%m, HOST_TO_DEVICE, sync=.false.)
 call device_sub3(this%bi%x_d,globaltmp_m%x_d, fval%x_d, this%m)
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 print *, "I am in mma_gpu.f90"


 ! call cuda_mpisum(this%bi%x_d, this%m)
 ! call device_sub2(this%bi%x_d, fval%x_d, this%m)
 call globaltmp_m%free()
end subroutine mma_gensub_gpu


subroutine mma_subsolve_dpip_gpu(this, designx)
  class(mma_t), intent(inout) :: this
  type(vector_t), intent(in) :: designx
  integer :: i, j, k, iter, ggdumiter, itto, ierr
  real(kind=rp) :: epsi, residumax, residunorm, &
  z, zeta, rez, rezeta, &
  delz, dz, dzeta, &
  steg, dummy_one, zold, zetaold, newresidu
  type(vector_t) :: y, lambda, s, mu, &   !!!m
  rey, relambda, remu, res, &
  dely, dellambda, &
  dy, dlambda, ds, dmu, &
  yold, lambdaold, sold, muold
  type(vector_t) :: globaltmp_m

  type(vector_t) :: x, xsi, eta, & !!!!!n
  rex, rexsi, reeta, &
  delx, diagx, dx, dxsi, deta, &
  xold, xsiold, etaold
  real(kind=rp) :: residu
  real(kind=rp) :: residu_small
  real(kind=rp) :: ratio_xx_dxx
  type(vector_t) :: bb
  type(matrix_t) :: GG
  type(matrix_t) :: AA
  type(matrix_t) :: globaltmp_mm

  integer :: info
  integer, dimension(this%m+1) :: ipiv
  real(kind=rp) :: re_xstuff_squ_global

  integer :: nglobal

  real(kind=rp) :: cons


  call globaltmp_m%init(this%m)
  call globaltmp_mm%init(this%m, this%m)

  call y%init(this%m)
  call lambda%init(this%m)
  call s%init(this%m)
  call mu%init(this%m)
  call rey%init(this%m)
  call relambda%init(this%m)
  call remu%init(this%m)
  call res%init(this%m)
  call dely%init(this%m)
  call dellambda%init(this%m)
  call dy%init(this%m)
  call dlambda%init(this%m)
  call ds%init(this%m)
  call dmu%init(this%m)
  call yold%init(this%m)
  call lambdaold%init(this%m)
  call sold%init(this%m)
  call muold%init(this%m)
  call x%init(this%n)
  call xsi%init(this%n)
  call eta%init(this%n)
  call rex%init(this%n)
  call rexsi%init(this%n)
  call reeta%init(this%n)
  call delx%init(this%n)
  call diagx%init(this%n)
  call dx%init(this%n)
  call dxsi%init(this%n)
  call deta%init(this%n)
  call xold%init(this%n)
  call xsiold%init(this%n)
  call etaold%init(this%n)
  call bb%init(this%m+1)

  call GG%init(this%m, this%n)
  call AA%init(this%m+1, this%m+1)
  dummy_one = 1
  epsi = 1 !100
  cons=0.5
  call device_add3s2(x%x_d,this%alpha%x_d,this%beta%x_d,cons,cons,this%n)
  cons=1
  call device_cfill(y%x_d,cons,this%m)
  z = 1
  zeta = 1
  call device_cfill(lambda%x_d,cons,this%m)
  call device_cfill(s%x_d,cons,this%m)





  print *, "I am in mma_subsolve_dpip_gpu"





end subroutine mma_subsolve_dpip_gpu

subroutine mma_KKT_gpu(this, x, df0dx, fval, dfdx)
  class(mma_t), intent(inout) :: this
  type(vector_t), intent(in) :: x
  type(vector_t), intent(in) :: fval
  type(vector_t), intent(in) :: df0dx
  type(matrix_t), intent(in) :: dfdx
  real(kind=rp) :: rez, rezeta
  type(vector_t) :: rey, relambda, remu, res
  type(vector_t) :: rex, rexsi, reeta
  real(kind=rp) ::residu_val !!!(3*this%n+4*this%m+2)
  real(kind=rp), dimension(4*this%m+2) ::residu_small !!!(4*this%m+2)
  integer :: ierr
  real(kind=rp) :: re_xstuff_squ_global
  real(kind=rp) :: cons
  real(kind=rp) :: globaltemp_norm
end subroutine mma_KKT_gpu


end submodule mma_gpu
