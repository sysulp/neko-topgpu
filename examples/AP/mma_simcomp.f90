! Copyright (c) 2024, The Neko Authors
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

 ! Implements the `mma_comp_t` type.
module mma_simcomp
  use neko_config
  use num_types, only: rp
  use case, only: case_t
  use json_module, only: json_file
  use json_utils, only: json_get_or_default
  use simulation_component, only: simulation_component_t
  use field, only: field_t
  use logger, only: neko_log
  use mma, only: mma_t
  use comm
    
    
  use neko_config, only: NEKO_BCKND_DEVICE
  use vector, only: vector_t
  use matrix, only: matrix_t
  use device
  
  implicit none
  private

  ! An empty user defined simulation component.
  ! This is a simple example of a user-defined simulation component.
  type, public, extends(simulation_component_t) :: mma_comp_t

     real(kind=rp) :: tol !< Just some dummy variable to show it working.
     type(field_t) :: tmp !< Just some dummy field to show it working.


     type(field_t) :: designx !< Just some dummy field to show it working.
     type(field_t) :: xmax !< Just some dummy field to show it working.
     type(field_t) :: xmin !< Just some dummy field to show it working.

     integer :: m !< Just some dummy variable to show it working.
     real(kind=rp) :: a0_const !< Just some dummy variable to show it working.
     real(kind=rp) :: a_const !< Just some dummy variable to show it working.
     real(kind=rp) :: c_const !< Just some dummy variable to show it working.
     real(kind=rp) :: d_const !< Just some dummy variable to show it working.

     type(mma_t) :: mma !< The actual MMA simulation component.

   contains
     ! Constructor from json, wrapping the actual constructor.
     procedure, pass(this) :: init => simcomp_test_init_from_json
     ! Actual constructor.
     procedure, pass(this) :: init_from_attributes => &
          simcomp_test_init_from_attributes
     ! Destructor.
     procedure, pass(this) :: free => simcomp_test_free
     ! Compute the simcomp_test field.
     procedure, pass(this) :: compute_ => simcomp_test_compute
  end type mma_comp_t

contains
  ! Constructor from json.
  subroutine simcomp_test_init_from_json(this, json, case)
    class(mma_comp_t), intent(inout) :: this
    type(json_file), intent(inout) :: json
    class(case_t), intent(inout), target :: case

    call this%init_base(json, case)

    call this%tmp%init(case%msh, case%fluid%Xh, "tmp")
    call this%designx%init(case%msh, case%fluid%Xh, "designx")
    call this%xmax%init(case%msh, case%fluid%Xh, "xmax")
    call this%xmin%init(case%msh, case%fluid%Xh, "xmin")

    ! Read the tolerance
    call json_get_or_default(json, "tol", this%tol, 1.0e-6_rp)
    call json_get_or_default(json, "m", this%m, 2)
    call json_get_or_default(json, "a0_const", this%a0_const, 1.0_rp)
    call json_get_or_default(json, "a_const", this%a_const, 0.0_rp)
    call json_get_or_default(json, "c_const", this%c_const, 1000.0_rp)
    call json_get_or_default(json, "d_const", this%d_const, 1.0_rp)

    call this%init_from_attributes()
  end subroutine simcomp_test_init_from_json

  ! Actual constructor.
  subroutine simcomp_test_init_from_attributes(this)
    class(mma_comp_t), intent(inout) :: this

    real(kind=rp), allocatable ::a(:), c(:), d(:)
    real(kind=rp) :: a0
    integer :: nloc, nglobal, ierr, rank


    allocate(a(this%m))
    allocate(c(this%m))
    allocate(d(this%m))


    a0= this%a0_const
    a= this%a_const
    c= this%c_const
    d= this%d_const
    ! a0= 1.0
    ! a= 0.0
    ! c= 100000.0
    ! d= 1000.0
    nloc = this%designx%dof%size()
    ! print *, "nloc=", nloc
    ! call MPI_Allreduce(nloc, nglobal, 1, &
    !     MPI_INTEGER, mpi_sum, neko_comm, ierr)
    ! print *, "nglobal=", nglobal

    ! initial design
    this%designx%x=1.0
    this%xmax%x = 10.0_rp
    this%xmin%x = 0.0_rp
    call this%mma%init(reshape(this%designx%x, [nloc]), &
         nloc, this%m, a0, a, c, d, this%xmin%x, this%xmax%x)

    ! Get the rank of the current process
    ! call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    ! print *, "************************"
    ! print *, 'Processor ID (Rank): ', rank, "Max_x= ", maxval(this%designx%dof%x)
    ! print *, 'Processor ID (Rank): ', rank, "min_x= ", minval(this%designx%dof%x)
    ! print *, 'Processor ID (Rank): ', rank, "Max_y= ", maxval(this%designx%dof%y)
    ! print *, 'Processor ID (Rank): ', rank, "min_y= ", minval(this%designx%dof%y)
    ! print *, 'Processor ID (Rank): ', rank, "Max_z= ", maxval(this%designx%dof%z)
    ! print *, 'Processor ID (Rank): ', rank, "min_z= ", minval(this%designx%dof%z)
    ! print *, "************************"

  end subroutine simcomp_test_init_from_attributes

  ! Destructor.
  subroutine simcomp_test_free(this)
    class(mma_comp_t), intent(inout) :: this

    call this%tmp%free()
    call this%mma%free()

    call this%free_base()
  end subroutine simcomp_test_free

  ! Computations.
  subroutine simcomp_test_compute(this, t, tstep)
    implicit none
    class(mma_comp_t), intent(inout) :: this
    real(kind=rp), intent(in) :: t
    integer, intent(in) :: tstep

    real(kind=rp) :: L

    integer :: iter, i, j, k, e, counter, ierr, rank, size, nglobal
    integer, allocatable :: recv_counts(:), displs(:)
    real(kind=rp), dimension(2,4) :: teststuff

    real(kind=rp) :: start_time, end_time
    real(kind=rp), dimension(this%mma%get_n()) :: x

    real(kind=rp), dimension(this%mma%get_m()) :: fval, fvalglobal
    real(kind=rp), dimension(this%mma%get_m(),this%mma%get_n()) :: dfdx
    real(kind=rp) :: f0val, f0valeps, f0valglobal
    real(kind=rp), dimension(this%mma%get_n()) :: df0dx
    ! character(len=50) :: filename
    real(kind=rp), dimension(this%mma%get_n(),4) :: stuff
    ! real(kind=rp), dimension(4320,4) :: all_stuff
    real(kind=rp), allocatable :: all_stuff(:,:)
    integer, allocatable :: nloc_all(:)
    
    
    !gpu stuffffffffffffff
    type(vector_t) :: Vx
    type(vector_t) :: Vdf0dx
    type(vector_t) :: Vfval
    type(matrix_t) :: Vdfdx
    

    character(len=80) :: iFileName ! Filename to save the VTK data

    L=0_rp

    x= reshape(this%designx%x, [this%mma%get_n()])
    call func1 (this, this%mma%get_n(), this%mma%get_m(), L, f0val, df0dx, fval , dfdx)
    print *, 'iter=', 0,&
         '-------,f0val= ', f0val, ',   fval= ', fval
    
    call Vx%init(this%mma%get_n())
    call Vdf0dx%init(this%mma%get_n())
    call Vfval%init(this%mma%get_m())
    call Vdfdx%init(this%mma%get_m(),this%mma%get_n())
    
    call device_memcpy_r1(x, Vx%x_d,this%mma%get_n(), HOST_TO_DEVICE, sync=.false.)
    call device_memcpy_r1(df0dx, Vdf0dx%x_d,this%mma%get_n(), HOST_TO_DEVICE, sync=.false.)
    call device_memcpy_r1(fval, Vfval%x_d,this%mma%get_m(), HOST_TO_DEVICE, sync=.false.)
    call device_memcpy_r2(dfdx, Vdfdx%x_d,this%mma%get_n()*this%mma%get_m(), HOST_TO_DEVICE, sync=.false.)
    
    
    iter =0
    call this%mma%mma_gensub_gpu(iter, Vx, Vdf0dx, Vfval, Vdfdx)
    

  end subroutine simcomp_test_compute

  subroutine write_stuff_vtk(stuff, n, filename)
    ! ----------------------------------------------------------- !
    !  This subroutine writes a nx4 array into a vtk file.        !
    !  The array is called stuff(n,4) holding the coordinates of  !
    !  all points and thier corresponding scalar value            !
    !  For a given point n, the array is defined as follows:      !
    !  x =stuff(n,1), y=stuff(n,2), z=stuff(n,3)                  !
    !  scalar field value= stuff(n,4)                             !
    ! ----------------------------------------------------------- !
    implicit none
    integer, intent(in) :: n ! Number of design variables
    real(kind=rp), dimension(n,4), intent(in) :: stuff ! Array containing x, y, z, and T values
    character(len=*), intent(in) :: filename ! Filename to save the VTK data

    integer :: i ! Loop variable
    integer :: unit ! File unit number

    ! Open the VTK file for writing
    open(newunit=unit, file=filename, status="replace", action="write")

    ! Write VTK header
    write(unit, '(A)') '# vtk DataFile Version 3.0'
    write(unit, '(A)') 'Stuff data'
    write(unit, '(A)') 'ASCII'
    write(unit, '(A)') 'DATASET POLYDATA'

    ! Write the points (x, y, z coordinates)
    write(unit, '(A, I8)') 'POINTS ', n, ' double'
    do i = 1, n
       write(unit, '(3(ES15.8,1X))') stuff(i, 1), stuff(i, 2), stuff(i, 3)
    end do

    ! Write the temperature data associated with the points
    write(unit, '(A, I8)') 'POINT_DATA ', n
    write(unit, '(A)') 'SCALARS Temperature double 1'
    write(unit, '(A)') 'LOOKUP_TABLE default'
    do i = 1, n
       write(unit, '(ES15.8)') stuff(i, 4)
    end do

    ! Close the file
    close(unit)

  end subroutine write_stuff_vtk

  subroutine func1 (this, n, m, L, f0val, df0dx, fval , dfdx)
    ! ----------------------------------------------------------- !
    !  This subroutine calculates function values and gradients   !
    !  for "toy problem 3":                                       !
    !                                                             !
    !    minimize sum_(j=1,..,n) xj/n                             !
    !  subject to sum_(j=1,..,n) {(xj - Xj_GLL)^2}=0              !
    ! ----------------------------------------------------------- !
    implicit none
    class(mma_comp_t), intent(inout) :: this

    integer, intent(in) :: n, m
    real(kind=rp), intent(in) :: L
    real(kind=rp), intent(inout) :: f0val
    real(kind=rp), dimension(n), intent(inout) :: df0dx
    real(kind=rp), dimension(m), intent(inout) :: fval
    real(kind=rp), dimension(m,n), intent(inout) :: dfdx

    real(kind=rp), dimension(n) :: x, coordx, coordy, coordz
    integer :: i,j,k,e, counter, ierr, nglobal
    real(kind=rp) :: Globalf0val


    call MPI_Allreduce(n, nglobal, 1, &
         MPI_INTEGER, mpi_sum, neko_comm, ierr)


    x= reshape(this%designx%x, [n])
    coordx= reshape(this%designx%dof%x, [n])

    f0val=sum(x)/nglobal
    df0dx=1.0_rp/nglobal
    Globalf0val=0_rp
    call MPI_Allreduce(f0val, Globalf0val, 1, &
         mpi_real_precision, mpi_sum, neko_comm, ierr)
    f0val=Globalf0val
    ! f0val=0
    ! df0dx=0
    fval(1)=sum((x-coordx)**2)
    ! fval(1)=sum(this%designx%x) - this%mma%get_n()*L
    Globalf0val=0_rp
    call MPI_Allreduce(fval(1), Globalf0val, 1, &
         mpi_real_precision, mpi_sum, neko_comm, ierr)
    fval(1)=Globalf0val

    dfdx(1,:)=2*(x-coordx)
    fval(2) = -fval(1)
    dfdx(2,:) = - dfdx(1,:)


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! f0val=sum(this%designx%x)/nglobal
    ! df0dx=1.0_rp/nglobal
    ! Globalf0val=0_rp
    ! call MPI_Allreduce(f0val, Globalf0val, 1, &
    !   mpi_real_precision, mpi_sum, neko_comm, ierr)
    ! f0val=Globalf0val

    ! ! f0val=0
    ! ! df0dx=0

    ! fval(1)=sum(this%designx%x)/nglobal
    ! ! fval(1)=sum(this%designx%x) - this%mma%get_n()*L
    ! Globalf0val=0_rp
    ! call MPI_Allreduce(fval(1), Globalf0val, 1, &
    !   mpi_real_precision, mpi_sum, neko_comm, ierr)
    ! fval(1)=Globalf0val

    ! dfdx(1,:)=1.0_rp/nglobal
    ! fval(2) = -fval(1)
    ! dfdx(2,:) = - dfdx(1,:)
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  end subroutine func1

end module mma_simcomp

