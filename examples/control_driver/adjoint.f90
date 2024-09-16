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

 ! Implements the `adjoint_t` type.
module adjoint_mod
  use num_types, only: rp, dp
  use json_module, only: json_file
  use json_utils, only: json_get, json_get_or_default
  use simulation_component, only: simulation_component_t
  use case, only: case_t
  use field, only: field_t
  use coefs, only: coef_t
  use field_registry, only: neko_field_registry
  use scratch_registry, only: neko_scratch_registry
  use adjoint_pnpn, only: adjoint_pnpn_t
  use adjoint_output, only: adjoint_output_t
  use neko_config, only: NEKO_BCKND_DEVICE
  use field_math, only: field_cfill, field_sub2, field_copy, field_glsc2, field_glsc3
  use field_math, only: field_add2
  use math, only: glsc2, glsc3
  use device_math, only: device_glsc2
  use adv_lin_no_dealias, only: adv_lin_no_dealias_t
  use logger, only: neko_log, LOG_SIZE
  use adjoint_scheme, only: adjoint_scheme_t
  use adjoint_fctry, only: adjoint_scheme_factory
  use time_step_controller, only: time_step_controller_t
  use time_scheme_controller, only: time_scheme_controller_t
  use mpi_f08, only: MPI_WTIME
  use jobctrl, only: jobctrl_time_limit
  use profiler, only: profiler_start, profiler_stop, profiler_start_region, &
       profiler_end_region
  use file, only: file_t
  use num_types, only : rp, sp, dp
  use fluid_scheme, only : fluid_scheme_factory
  use fluid_pnpn, only : fluid_pnpn_t
  use fluid_scheme, only : fluid_scheme_t
  use fluid_output, only : fluid_output_t
  use chkp_output, only : chkp_output_t
  use mean_sqr_flow_output, only : mean_sqr_flow_output_t
  use mean_flow_output, only : mean_flow_output_t
  use fluid_stats_output, only : fluid_stats_output_t
  use mpi_f08
  use mesh_field, only : mesh_fld_t, mesh_field_init, mesh_field_free
  use parmetis, only : parmetis_partmeshkway
  use redist, only : redist_mesh
  use sampler, only : sampler_t
  use flow_ic, only : set_flow_ic
  use scalar_ic, only : set_scalar_ic
  use field, only : field_t
  use field_registry, only : neko_field_registry
  use stats, only : stats_t
  use file, only : file_t
  use utils, only : neko_error
  use mesh, only : mesh_t
  use comm
  use time_scheme_controller, only : time_scheme_controller_t
  use logger, only : neko_log, NEKO_LOG_QUIET, LOG_SIZE
  use jobctrl, only : jobctrl_set_time_limit
  use user_intf, only : user_t
  use scalar_pnpn, only : scalar_pnpn_t
  use json_module, only : json_file, json_core, json_value
  use json_utils, only : json_get, json_get_or_default
  use scratch_registry, only : scratch_registry_t, neko_scratch_registry
  use point_zone_registry, only: neko_point_zone_registry
  use material_properties, only : material_properties_t
  use adjoint_ic, only : set_adjoint_ic
  use json_utils, only : json_extract_item
  use json_utils_ext, only: json_key_fallback, json_get_subdict
  implicit none
  private

  ! An empty user defined simulation component.
  ! This is a simple example of a user-defined simulation component.
  type, public :: adjoint_obj

     class(adjoint_scheme_t), allocatable :: scheme
     type(case_t), pointer :: case

     ! Fields
     real(kind=rp) :: tol
     type(adjoint_output_t) :: f_out
     type(sampler_t) :: s

     logical :: have_scalar = .false.
     logical :: converged = .false.
     logical :: computed = .false.

   contains
     ! Constructor from json, wrapping the actual constructor.
     procedure, pass(this) :: init => adjoint_init_from_json
     ! Actual constructor.
     procedure, pass(this) :: init_from_attributes => &
          adjoint_init_from_attributes
     ! Destructor.
     procedure, pass(this) :: free => adjoint_free
  end type adjoint_obj

contains

  ! Constructor from json.
  subroutine adjoint_init_from_json(this, case)
    class(adjoint_obj), intent(inout) :: this
    type(case_t), intent(inout), target :: case

    ! Read the tolerance
    call json_get_or_default(case%params, "tol", this%tol, 1.0e-6_rp)

    ! Check if the scalar field is allocated
    if (allocated(case%scalar)) then
       this%have_scalar = .true.
    end if

    call this%init_from_attributes(case, this%tol)

  end subroutine adjoint_init_from_json

  ! Constructor from attributes
  subroutine adjoint_init_from_attributes(this, case, tol)
    class(adjoint_obj), intent(inout) :: this
    class(case_t), intent(inout), target :: case
    real(kind=rp), intent(in), optional :: tol

    this%case => case
    this%tol = tol
    call adjoint_case_init_common(this, case)

  end subroutine adjoint_init_from_attributes

  !> Initialize a case from its (loaded) params object
  subroutine adjoint_case_init_common(this, C)
    class(adjoint_obj), intent(inout) :: this
    type(case_t), intent(inout) :: C
    character(len=:), allocatable :: output_directory
    integer :: lx = 0
    logical :: scalar = .false.
    type(file_t) :: msh_file, bdry_file, part_file
    logical :: found, logical_val
    integer :: integer_val
    real(kind=rp) :: real_val
    character(len=:), allocatable :: string_val
    real(kind=rp) :: stats_start_time, stats_output_val
    integer :: stats_sampling_interval
    integer :: output_dir_len
    integer :: precision

    ! extra things for json
    type(json_file) :: ic_json
    character(len=:), allocatable :: json_key

    !
    ! Setup fluid scheme
    !
    ! HARRY
    ! keep the schemes the same for SURE
    call json_get(C%params, 'case.fluid.scheme', string_val)
    call adjoint_scheme_factory(this%scheme, trim(string_val))

    ! HARRY
    ! same with polynomial order
    call json_get(C%params, 'case.numerics.polynomial_order', lx)
    lx = lx + 1 ! add 1 to get number of gll points
    call this%scheme%init(C%msh, lx, C%params, C%usr, C%material_properties)
    ! this%scheme%chkp%tlag => C%tlag
    ! this%scheme%chkp%dtlag => C%dtlag
    select type (f => this%scheme)
      type is (adjoint_pnpn_t)
       !  f%chkp%abx1 => f%abx1
       !  f%chkp%abx2 => f%abx2
       !  f%chkp%aby1 => f%aby1
       !  f%chkp%aby2 => f%aby2
       !  f%chkp%abz1 => f%abz1
       !  f%chkp%abz2 => f%abz2
    end select

    ! !
    ! ! Setup scalar scheme
    ! !
    ! ! @todo no scalar factory for now, probably not needed
    ! if (C%params%valid_path('case.scalar')) then
    !    call json_get_or_default(C%params, 'case.scalar.enabled', scalar,&
    !                             .true.)
    ! end if

    ! if (scalar) then
    !    allocate(C%scalar)
    !    call C%scalar%init(C%msh, this%scheme%c_Xh, this%scheme%gs_Xh, C%params, C%usr,&
    !                       C%material_properties)
    !    call this%scheme%chkp%add_scalar(C%scalar%s)
    !    this%scheme%chkp%abs1 => C%scalar%abx1
    !    this%scheme%chkp%abs2 => C%scalar%abx2
    !    this%scheme%chkp%slag => C%scalar%slag
    ! end if

    !
    ! Setup user defined conditions
    !
    ! json_key = json_key_fallback(C%params, 'case.adjoint.boundary_types', &
    !      'case.fluid.boundary_types')

    ! call json_get(C%params, 'case.fluid.boundary_types', string_val)
    ! write(*,*) 'string_val: ', string_val

    ! if (trim(string_val) .eq. 'user') then
    !    call this%scheme%set_usr_inflow(C%usr%fluid_user_if)
    ! end if


    ! ! Setup user boundary conditions for the scalar.
    ! if (scalar) then
    !    call C%scalar%set_user_bc(C%usr%scalar_user_bc)
    ! end if

    !
    ! Setup initial conditions
    !
    ! This should be unique from forward solution
    ! HARRY
    ! ------------------------------------------------------------
    ! I want to give a subdictionary to IC's so we can different ICs
    ! for different solvers,
    ! (not hardcoded to fluid)
    !
    json_key = json_key_fallback(C%params, 'case.adjoint.initial_condition', &
         'case.fluid.initial_condition')

    call json_get(C%params, json_key//'.type', string_val)
    call json_get_subdict(C%params, json_key, ic_json)

    if (trim(string_val) .ne. 'user') then
       call set_adjoint_ic( &
            this%scheme%u_adj, this%scheme%v_adj, this%scheme%w_adj, &
            this%scheme%p_adj, this%scheme%c_Xh, this%scheme%gs_Xh, &
            string_val, ic_json)
    else
       call set_adjoint_ic( &
            this%scheme%u_adj, this%scheme%v_adj, this%scheme%w_adj, &
            this%scheme%p_adj, this%scheme%c_Xh, this%scheme%gs_Xh, &
            C%usr%fluid_user_ic, ic_json)
    end if

    ! if (scalar) then
    !    call json_get(C%params, 'case.scalar.initial_condition.type', string_val)
    !    if (trim(string_val) .ne. 'user') then
    !       call set_scalar_ic(C%scalar%s, &
    !         C%scalar%c_Xh, C%scalar%gs_Xh, string_val, C%params)
    !    else
    !       call set_scalar_ic(C%scalar%s, &
    !         C%scalar%c_Xh, C%scalar%gs_Xh, C%usr%scalar_user_ic, C%params)
    !    end if
    ! end if

    ! Add initial conditions to BDF scheme (if present)
    select type (f => this%scheme)
      type is (adjoint_pnpn_t)
       call f%ulag%set(f%u_adj)
       call f%vlag%set(f%v_adj)
       call f%wlag%set(f%w_adj)

       ! baseflow is solution to forward problem
       !  u_b => neko_field_registry%get_field('u')
       !  v_b => neko_field_registry%get_field('v')
       !  w_b => neko_field_registry%get_field('w')

       !!
       !! Setup initial baseflow
       !!
       !call json_get(C%params, 'case.fluid.baseflow.type', string_val)

       !if (trim(string_val) .ne. 'user') then
       !   call set_baseflow(u_b, v_b, w_b, this%scheme%c_Xh, this%scheme%gs_Xh, &
       !        string_val, C%params)
       !else
       !   call set_baseflow(u_b, v_b, w_b, this%scheme%c_Xh, this%scheme%gs_Xh, &
       !        C%usr%baseflow_user, C%params)
       !end if



       !  ! Tim what is this for?
       !  call field_cfill(f%u_b, 0.0_rp)
       !  call field_cfill(f%v_b, 0.0_rp)
       !  call field_cfill(f%w_b, 0.0_rp)


    end select

    !
    ! Validate that the case is properly setup for time-stepping
    !
    call this%scheme%validate

    ! if (scalar) then
    !    call C%scalar%slag%set(C%scalar%s)
    !    call C%scalar%validate
    ! end if

    !
    ! Setup output precision of the field files
    !
    call json_get_or_default(C%params, 'case.output_precision', string_val,&
         'single')

    if (trim(string_val) .eq. 'double') then
       precision = dp
    else
       precision = sp
    end if

    !
    ! Setup sampler
    !
    call this%s%init(C%end_time)
    if (scalar) then
       this%f_out = adjoint_output_t(precision, this%scheme, C%scalar, &
            path = trim(output_directory))
    else
       this%f_out = adjoint_output_t(precision, this%scheme, &
            path = trim(output_directory))
    end if

    ! HARRY
    ! fuck the sampler we're changing this anyway
    call json_get_or_default(C%params, 'case.fluid.output_control',&
         string_val, 'org')

    if (trim(string_val) .eq. 'org') then
       ! yes, it should be real_val below for type compatibility
       call json_get(C%params, 'case.nsamples', real_val)
       call this%s%add(this%f_out, real_val, 'nsamples')
    else if (trim(string_val) .eq. 'never') then
       ! Fix a dummy 0.0 output_value
       call json_get_or_default(C%params, 'case.fluid.output_value', real_val, &
            0.0_rp)
       call this%s%add(this%f_out, 0.0_rp, string_val)
    else
       call json_get(C%params, 'case.fluid.output_value', real_val)
       call this%s%add(this%f_out, real_val, string_val)
    end if

    ! !
    ! ! Save checkpoints (if nothing specified, default to saving at end of sim)
    ! !
    ! call json_get_or_default(C%params, 'case.output_checkpoints',&
    !      logical_val, .true.)
    ! if (logical_val) then
    !    call json_get_or_default(C%params, 'case.checkpoint_format', &
    !         string_val, "chkp")
    !   !   C%f_chkp = chkp_output_t(this%scheme%chkp, path = output_directory, &
    !         ! fmt = trim(string_val))
    !    call json_get_or_default(C%params, 'case.checkpoint_control', &
    !         string_val, "simulationtime")
    !    call json_get_or_default(C%params, 'case.checkpoint_value', real_val,&
    !         1e10_rp)
    !   !  call this%s%add(C%f_chkp, real_val, string_val)
    ! end if

  end subroutine adjoint_case_init_common

  ! Destructor.
  subroutine adjoint_free(this)
    class(adjoint_obj), intent(inout) :: this

    call this%scheme%free()
    call this%s%free()

  end subroutine adjoint_free

end module adjoint_mod

