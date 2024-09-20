!===========================================================================!
!                       Method of Moving Asymptotes                         !
! This implementation is based on the following documents:                  !
!        1-https://people.kth.se/~krille/mmagcmma.pdf                       !
!        2-https://people.kth.se/~krille/originalmma.pdf                    !
!        2-https://comsolyar.com/wp-content/uploads/2020/03/gcmma.pdf       !
! ------------------------------------------------------------------------- !
!                                                                           !
! This module solves the following original optimization problem:           !
!                                                                           !
!      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )          !
!    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m                    !
!                xmin_j <= x_j <= xmax_j,    j = 1,...,n                    !
!                z >= 0,   y_i >= 0,         i = 1,...,m                    !
!                                                                           !
! by first creating the following convex approximation of the original      !
! problem:                                                                  !
!                                                                           !
!      Minimize sum_{j = 1,...,n} (p0j / (upp_j-x_j) + q0j / (x_j-low_j)) + !
!                        a0*z + sum_i = 1,...,m(c_i*y_i + 0.5*d_i*y_i^2)    !
!    subject to sum_{j = 1,...,n} (pij / (upp_j-x_j) + qij / (x_j-low_j)) + !
!                    a_i*z + y_i <= b_i,                       i = 1,...,m  !
!               xmin_j <= alpha_j <= x_j <= beta_j <= xmax_j   j = 1,...,n  !
!               y_i>=0                                         i = 1,...,m  !
!               z>=0                                                        !
!                                                                           !
! note that based on eq(3.5) there should be r0 in the approximated problem !
! however since it is just a constant added to a minimization problem, it   !
! is ignored.                                                               !
! A primal-dual algorithm is then employed to solve the aproximated problem !
! using interior point method.                                              !
!===========================================================================!

module mma

  ! Inclusions from Neko
  use num_types, only: rp
  use comm, only: neko_comm, mpi_real_precision, pe_rank, pe_size
  use neko_config, only: NEKO_BCKND_DEVICE
  use vector, only: vector_t
  use matrix, only: matrix_t

  ! Inclusions from external dependencies and standard libraries
  use, intrinsic :: iso_fortran_env, only: stderr => error_unit
  use mpi_f08, only: mpi_sum, MPI_Allreduce, mpi_max, mpi_min, mpi_sum, &
       mpi_integer
  implicit none
  private

  type, public :: mma_t
     private

     real(kind=rp) :: a0, f0val, asyinit, asyincr, asydecr, epsimin, &
          residumax, residunorm
     integer :: n, m, max_iter
     type(vector_t) :: xold1, xold2, low, upp, alpha, beta, a, c, d, xmax, xmin

     logical :: is_initialized = .false.
     logical :: is_updated = .false.

     ! Internal dummy variables for MMA
     type(vector_t) :: p0j, q0j
     type(matrix_t) :: pij, qij
     type(vector_t) :: bi

     !---nesessary for KKT check after updating df0dx, fval, dfdx --------
     real(kind=rp) :: z, zeta
     type(vector_t) :: y, lambda, s, mu
     type(vector_t) :: xsi, eta

   contains
     procedure, public, pass(this) :: init => mma_init
     procedure, public, pass(this) :: free => mma_free
     procedure, public, pass(this) :: update => mma_update
     procedure, public, pass(this) :: KKT => mma_KKT

     ! Getters for the MMA object
     procedure, public, pass(this) :: get_n => mma_get_n
     procedure, public, pass(this) :: get_m => mma_get_m
     procedure, public, pass(this) :: get_residumax => mma_get_residumax
     procedure, public, pass(this) :: get_residunorm => mma_get_residunorm

     !Generates the sub problem--the MMA convex approximation
     procedure, pass(this) :: mma_gensub
     !Solve the dual with an interior point method
     procedure, pass(this) :: mma_subsolve_dpip

  end type mma_t

  ! ========================================================================== !
  ! Interface for the CPU backend

  interface
     !> Generate the approximation sub problem on the CPU.
     module subroutine mma_gensub_cpu(this, iter, x, df0dx, fval, dfdx)
       class(mma_t), intent(inout) :: this
       real(kind=rp), dimension(this%n), intent(in) :: x
       real(kind=rp), dimension(this%n), intent(in) :: df0dx
       real(kind=rp), dimension(this%m), intent(in) :: fval
       real(kind=rp), dimension(this%m, this%n), intent(in) :: dfdx
       integer, intent(in) :: iter
     end subroutine mma_gensub_cpu

     !> Solve the dual with an interior point method on the CPU.
     module subroutine mma_subsolve_dpip_cpu(this, designx)
       class(mma_t), intent(inout) :: this
       real(kind=rp), dimension(this%n), intent(inout) :: designx
     end subroutine mma_subsolve_dpip_cpu

     !> Compute the KKT condition for a given design x on the CPU.
     module subroutine mma_KKT_cpu(this, x, df0dx, fval, dfdx)
       class(mma_t), intent(inout) :: this
       real(kind=rp), dimension(this%n), intent(in) :: x
       real(kind=rp), dimension(this%m), intent(in) :: fval
       real(kind=rp), dimension(this%n), intent(in) :: df0dx
       real(kind=rp), dimension(this%m, this%n), intent(in) :: dfdx
     end subroutine mma_KKT_cpu
  end interface

contains

  subroutine mma_init(this, x, n, m, a0, a, c, d, xmin, xmax)
    ! ----------------------------------------------------- !
    ! Initializing the mma object and all the parameters    !
    ! required for MMA method. (a_i, c_i, d_i, ...)         !
    ! x: the design varaibles(DV), n: number of DV,         !
    ! m: number of constraints                              !
    !                                                       !
    ! Note that residumax & residunorm of the KKT conditions!
    ! are initialized with 10^5. This is done to avoid      !
    ! unnecessary extera computation of KKT norms for the   !
    ! initial design.                                       !
    ! ----------------------------------------------------- !

    class(mma_t), intent(inout) :: this
    integer, intent(in) :: n, m
    real(kind=rp), intent(in), dimension(n) :: x
    ! -------------------------------------------------------------------!
    !      Internal parameters for MMA                                   !
    !      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )   !
    !    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m             !
    !                xmin_j <= x_j <= xmax_j,    j = 1,...,n             !
    !                z >= 0,   y_i >= 0,         i = 1,...,m             !
    ! -------------------------------------------------------------------!
    real(kind=rp), dimension(n) :: xmax, xmin
    real(kind=rp), dimension(m) :: a, c, d
    real(kind=rp) :: a0

    call this%free()

    this%n = n
    this%m = m

    ! allocate(this%x(n))
    ! this%x = x
    call this%xold1%init(n)
    call this%xold2%init(n)
    this%xold1%x = x
    this%xold2%x = x

    call this%alpha%init(n)
    call this%beta%init(n)

    call this%a%init(m)
    call this%c%init(m)
    call this%d%init(m)
    call this%low%init(n)
    call this%upp%init(n)
    call this%xmax%init(n)
    call this%xmin%init(n)

    !internal dummy variables for MMA
    call this%p0j%init(n)
    call this%q0j%init(n)
    call this%pij%init(m,n)
    call this%qij%init(m,n)
    call this%bi%init(m)

    !---nesessary for KKT check after updating df0dx, fval, dfdx --------
    call this%y%init(m)
    call this%lambda%init(m)
    call this%s%init(m)
    call this%mu%init(m)
    call this%xsi%init(n)
    call this%eta%init(n)


    ! this%epsimin =  1.0e-10_rp
    ! based on the Cpp Code by Neils
    this%epsimin = 1.0e-9_rp * sqrt(1.0*m + 1.0*n)

    this%max_iter = 100

    this%a0 = a0
    this%a%x = a
    this%c%x = c
    this%d%x = d
    !setting the bounds for the design variable based on the problem
    this%xmax%x = xmax
    this%xmin%x = xmin




    this%low%x(:) = minval(x)
    this%upp%x(:) = maxval(x)

    !following parameters are set based on eq.3.8:--------
    this%asyinit = 0.5_rp !
    this%asyincr = 1.2_rp ! 1.1
    this%asydecr = 0.7_rp !0.65

    !setting KKT norms to a large number for the initial design
    this%residumax = 10**5_rp
    this%residunorm = 10**5_rp

    !the object is correctly initialized
    this%is_initialized = .true.
  end subroutine mma_init

  subroutine mma_update(this, iter, x, df0dx, fval, dfdx)
    ! ----------------------------------------------------- !
    ! Update the design variable x by solving the convex    !
    ! approximation of the problem.                         !
    !                                                       !
    ! This subroutine is called in each iteration of the    !
    ! optimization loop                                     !
    !                                                       !
    ! Todo: This should be overloaded for different input   !
    ! ----------------------------------------------------- !
    class(mma_t), intent(inout) :: this
    integer, intent(in) :: iter
    real(kind=rp), dimension(this%n), intent(inout) :: x
    real(kind=rp), dimension(this%n), intent(in) :: df0dx
    real(kind=rp), dimension(this%m), intent(in) :: fval
    real(kind=rp), dimension(this%m, this%n), intent(in) :: dfdx

    if (.not. this%is_initialized) then
       write(stderr, *) "The MMA object is not initialized."
       error stop
    end if

    ! generate a convex approximation of the problem
    call this%mma_gensub(iter, x, df0dx, fval, dfdx)

    !solve the approximation problem using interior point method
    call this%mma_subsolve_dpip(x)

    this%is_updated = .true.
  end subroutine mma_update


  subroutine mma_KKT(this, x, df0dx, fval, dfdx)
    ! ----------------------------------------------------- !
    ! Compute the KKT condition right hand side for a given !
    ! design x and set the max and norm values of the       !
    ! residue of KKT system to this%residumax and           !
    ! this%residunorm.                                      !
    !                                                       !
    ! The left hand sides of the KKT conditions are computed!
    ! for the following nonlinear programming problem:      !
    ! Minimize  f_0(x) + a_0*z +                            !
    !                       sum( c_i*y_i + 0.5*d_i*(y_i)^2 )!
    !   subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m !
    !         xmax_j <= x_j <= xmin_j,    j = 1,...,n       !
    !        z >= 0,   y_i >= 0,         i = 1,...,m        !
    !                                                       !
    !                                                       !
    ! Note that before calling this function, the function  !
    ! values (f0val, fval, dfdx, ...) should be updated     !
    ! using the new x values.                               !
    ! ----------------------------------------------------- !
    class(mma_t), intent(inout) :: this
    real(kind=rp), dimension(this%n), intent(in) :: x
    real(kind=rp), dimension(this%m), intent(in) :: fval
    real(kind=rp), dimension(this%n), intent(in) :: df0dx
    real(kind=rp), dimension(this%m, this%n), intent(in) :: dfdx

    if (.not. this%is_initialized) then
       write(stderr, *) "The MMA object is not initialized."
       error stop
    end if

    if (NEKO_BCKND_DEVICE .eq. 0) then
       call mma_KKT_cpu(this, x, df0dx, fval, dfdx)
    else
       write(stderr, *) "Device not supported for MMA."
       error stop
    end if

  end subroutine mma_KKT

  !> Deallocate the MMA object.
  subroutine mma_free(this)

    class(mma_t), intent(inout) :: this

    ! Deallocate the internal vectors
    call this%xold1%free()
    call this%xold2%free()
    call this%alpha%free()
    call this%beta%free()
    call this%a%free()
    call this%c%free()
    call this%d%free()
    call this%low%free()
    call this%upp%free()
    call this%xmax%free()
    call this%xmin%free()
    call this%p0j%free()
    call this%q0j%free()
    call this%bi%free()
    call this%y%free()
    call this%lambda%free()
    call this%s%free()
    call this%mu%free()
    call this%xsi%free()
    call this%eta%free()

    ! Deallocate the internal dummy matrices
    call this%pij%free()
    call this%qij%free()

    this%is_initialized = .false.
    this%is_updated = .false.

  end subroutine mma_free

  ! ========================================================================== !
  ! Private subroutines

  subroutine mma_gensub(this, iter, x, df0dx, fval, dfdx)
    ! ----------------------------------------------------- !
    ! Generate the approximation sub problem by computing   !
    ! the lower and upper asymtotes and the other necessary !
    ! parameters (alpha, beta, p0j, q0j, pij, qij, ...).    !
    ! ----------------------------------------------------- !
    class(mma_t), intent(inout) :: this
    real(kind=rp), dimension(this%n), intent(in) :: x
    real(kind=rp), dimension(this%n), intent(in) :: df0dx
    real(kind=rp), dimension(this%m), intent(in) :: fval
    real(kind=rp), dimension(this%m, this%n), intent(in) :: dfdx
    integer, intent(in) :: iter

    if (NEKO_BCKND_DEVICE .eq. 0) then
       call mma_gensub_cpu(this, iter, x, df0dx, fval, dfdx)
    else
       write(stderr, *) "Device not supported for MMA."
       error stop
    end if

  end subroutine mma_gensub

  subroutine mma_subsolve_dpip(this, designx)
    ! ------------------------------------------------------- !
    ! Dual-primal interior point method using Newton's step   !
    ! to solve MMA sub problem.                               !
    ! A Backtracking Line Search approach is used to compute  !
    ! the step size; starting with the full Newton's step     !
    ! (delta = 1) and deviding by 2 until we have a step size !
    ! that leads to a feasible point while ensuring a         !
    ! decrease in the residue.                                !
    ! ------------------------------------------------------- !
    class(mma_t), intent(inout) :: this
    real(kind=rp), dimension(this%n), intent(inout) :: designx

    if (NEKO_BCKND_DEVICE .eq. 0) then
       call mma_subsolve_dpip_cpu(this, designx)
    else
       write(stderr, *) "Device not supported for MMA."
       error stop
    end if

  end subroutine mma_subsolve_dpip

  ! ========================================================================== !
  ! Getters and setters

  pure function mma_get_n(this) result(n)
    class(mma_t), intent(in) :: this
    integer :: n
    n = this%n
  end function mma_get_n

  pure function mma_get_m(this) result(m)
    class(mma_t), intent(in) :: this
    integer :: m
    m = this%m
  end function mma_get_m

  pure function mma_get_residumax(this) result(residumax)
    class(mma_t), intent(in) :: this
    real(kind=rp) :: residumax
    residumax = this%residumax
  end function mma_get_residumax

  pure function mma_get_residunorm(this) result(residunorm)
    class(mma_t), intent(in) :: this
    real(kind=rp) :: residunorm
    residunorm = this%residunorm
  end function mma_get_residunorm

end module mma

