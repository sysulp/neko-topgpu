! Copyright (c) 2023, The Neko Authors
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
!> Implements the `minimum_dissipation_objective_function_t` type.
!
!
! If the objective function \int |\nabla u|^2,
! the corresponding adjoint forcing is \int \nabla v \cdot \nabla u
module minimum_dissipation_objective_function
  use num_types, only : rp
  use field_list, only : field_list_t
  use json_module, only : json_file
  use json_utils, only: json_get, json_get_or_default
  use source_term, only : source_term_t
  use coefs, only : coef_t
  use neko_config, only : NEKO_BCKND_DEVICE
  use utils, only : neko_error
  use field, only: field_t
  use new_design, only: new_design_t
  use field_math, only: field_col3, field_addcol3
  use user_intf, only: user_t, simulation_component_user_settings
  use json_module, only: json_file
  use steady_simcomp, only: steady_simcomp_t
  use simcomp_executor, only: neko_simcomps
  use fluid_user_source_term, only: fluid_user_source_term_t
  use num_types, only : rp
  use field, only : field_t
  use field_registry, only : neko_field_registry
  use math, only : rzero, copy, chsign
  use device_math, only: device_copy, device_cmult
  use neko_config, only: NEKO_BCKND_DEVICE
  use operators, only: curl, grad
  use scratch_registry, only : neko_scratch_registry
  use adjoint_minimum_dissipation_source_term, only : adjoint_minimum_dissipation_source_term_t
  use objective_function, only : objective_function_t
  use fluid_scheme, only : fluid_scheme_t
  use adjoint_scheme, only : adjoint_scheme_t
  use fluid_source_term, only: fluid_source_term_t
  use math, only : glsc2
  implicit none
  private

  !> A constant source term.
  !! The strength is specified with the `values` keyword, which should be an
  !! array, with a value for each component of the source.
  type, public, extends(objective_function_t) :: minimum_dissipation_objective_function_t
   
   contains
     !> The common constructor using a JSON object.
     procedure, pass(this) :: init => minimum_dissipation_objective_function_init
     !> Destructor.
     procedure, pass(this) :: free => minimum_dissipation_objective_function_free
     !> Computes the source term and adds the result to `fields`.
     procedure, pass(this) :: compute => minimum_dissipation_objective_function_compute
     !> Computes the source term and adds the result to `fields`.
     procedure, pass(this) :: compute_sensitivity => minimum_dissipation_objective_function_compute_sensitivity
  end type minimum_dissipation_objective_function_t

contains
  !> The common constructor using a JSON object.
  !! @param json The JSON object for the source.
  !! @param fields A list of fields for adding the source values.
  !! @param coef The SEM coeffs.
  subroutine minimum_dissipation_objective_function_init(this, fluid, adjoint)
    class(minimum_dissipation_objective_function_t), intent(inout) :: this
    class(fluid_scheme_t), intent(inout) :: fluid
    class(adjoint_scheme_t), intent(inout) :: adjoint
    ! TODO
    ! I'm actually a bit confused here..
    ! either we do something like this:
    ! initialize the adjoint source term
    ! allocate(adjoint_minimum_dissipation_source_term_t :: this%adjoint_forcing)
    !
    ! Or it seems we can just create a specific adjoint forcing and append it...
    ! this way we can init from components
    ! so maybe we don't need to store the adjoint forcing in the objective function type
    type(adjoint_minimum_dissipation_source_term_t) :: adjoint_forcing

    call this%init_base(fluid%dm_Xh)

    ! you will need to init this!
    ! append a source term based on objective function
  	 ! init the adjoint forcing term for the adjoint
    call adjoint_forcing%init_from_components(adjoint%f_adj_x, adjoint%f_adj_y, adjoint%f_adj_z, &
                                                fluid%u, fluid%v, fluid%w, &
                                                adjoint%c_Xh)
  ! append adjoint forcing term based on objective function
  call adjoint%source_term%add_source_term(adjoint_forcing)



  end subroutine minimum_dissipation_objective_function_init


  !> Destructor.
  subroutine minimum_dissipation_objective_function_free(this)
    class(minimum_dissipation_objective_function_t), intent(inout) :: this
    ! TODO
    ! you probably need to deallocate the source term!

    call this%free_base()
  end subroutine minimum_dissipation_objective_function_free

  subroutine minimum_dissipation_objective_function_compute(this, fluid)
    class(minimum_dissipation_objective_function_t), intent(inout) :: this
    class(fluid_scheme_t), intent(in) :: fluid
    integer :: i
    type(field_t), pointer :: wo1, wo2, wo3
    type(field_t), pointer :: objective_field 
    integer :: temp_indices(4)
    integer n



    call neko_scratch_registry%request_field(wo1, temp_indices(1))
    call neko_scratch_registry%request_field(wo2, temp_indices(2))
    call neko_scratch_registry%request_field(wo3, temp_indices(3))
    call neko_scratch_registry%request_field(objective_field, temp_indices(4))

	 ! compute the objective function.
	 ! TODO
	 ! we should be using masks etc

	 call grad(wo1%x, wo2%x, wo3%x, fluid%u%x, fluid%C_Xh) 
    call field_col3(objective_field,wo1,wo1)
    call field_addcol3(objective_field,wo2,wo2)
    call field_addcol3(objective_field,wo3,wo3)

	 call grad(wo1%x, wo2%x, wo3%x, fluid%v%x, fluid%C_Xh) 
    call field_addcol3(objective_field,wo1,wo1)
    call field_addcol3(objective_field,wo2,wo2)
    call field_addcol3(objective_field,wo3,wo3)

	 call grad(wo1%x, wo2%x, wo3%x, fluid%w%x, fluid%C_Xh) 
    call field_addcol3(objective_field,wo1,wo1)
    call field_addcol3(objective_field,wo2,wo2)
    call field_addcol3(objective_field,wo3,wo3)

    ! integrate the field
    n = wo1%size()
    this%objective_function_value = glsc2(objective_field%x,fluid%C_Xh%b, n)

    !TODO
    ! GPUS

    call neko_scratch_registry%relinquish_field(temp_indices)

  end subroutine minimum_dissipation_objective_function_compute

  subroutine minimum_dissipation_objective_function_compute_sensitivity(this, fluid, adjoint)
    class(minimum_dissipation_objective_function_t), intent(inout) :: this
    class(fluid_scheme_t), intent(in) :: fluid
    class(adjoint_scheme_t), intent(in) :: adjoint


    ! here it should just be an inner product between the forward and adjoint
    call field_col3(this%sensitivity_to_coefficient, fluid%u, adjoint%u_adj)
    call field_addcol3(this%sensitivity_to_coefficient, fluid%v, adjoint%v_adj)
    call field_addcol3(this%sensitivity_to_coefficient, fluid%w, adjoint%w_adj)

  end subroutine minimum_dissipation_objective_function_compute_sensitivity

end module minimum_dissipation_objective_function
