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

 ! Implements the `objective_function_t` type.
module objective_function
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
  use dofmap, only : dofmap_t
  use filters, only: permeability_field
  implicit none
  private

  type :: objective_function_t
   ! > objective function value
   real(kind=rp), public :: objective_function_value
   ! it may also be nice to have a list of the objective function value to append to instead?
   !real(kind=rp), public :: objective_function_value(:)

   !> A mask for where the objective function is evaluated
   ! TODO
   ! note for TIM
   ! a field is a bit excessive here...
   ! we could either use point zone or define an array of logicals the same size as the field
   ! I think the second option is better
   ! for now we'll use the whole field
  	type(field_t), public :: objective_location
  	!> the forcing term entering the adjoint equation
  	class(source_term_t), public :: adjoint_forcing

	contains:
	!> init (will make this legit at some point)
	procedure, pass(this) :: init_base => objective_function_init_base
	procedure, pass(this) :: free_base => objective_function_free_base


	!> this will compute the objective function
	! note for TIM,
	! this will REALLY need to be modified in the future...
	! in a steady case, we just need to compute it on the last step
	! in an unsteady case this will be a time integral
	!
	! TODO
	! this will need to be deffered in some way 
	! the init reads the JSON (or maybe we pass what objective function we have externally)
	! based on the objective we have, we  
	procedure(objective_function_compute), pass(this), deferred :: compute  
	! TODO
	! maybe it would have been smarter to have a "coeficient" type, which is just a scalar field
	! and set of mappings going from design_indicator -> coeficient and their corresponding chain rules
	! maybe also some information about what equation they live in...

	!> Destructor 
	procedure(objective_function_free), pass(this), deferred :: free 
	! TODO
	end type objective_function_t

	abstract interface
		subroutine objective_function_compute 

	public :: objective_function_t

contains

	subroutine objective_function_init(this, dm_Xh)
	class(objective_function_t), target, intent(inout) :: this
	type(dofmap_t) :: dm_Xh    !< Dofmap associated with \f$ X_h \f$
	integer :: n, i
	! init the fields
	call this%design_indicator%init(dm_Xh, "design_indicator")
   call this%brinkman_amplitude%init(dm_Xh, "brinkman_amplitude")

   ! TODO
   ! this is where we steal basically everything in brinkman_source_term regarding loading initial fields
   ! for now, make it a cylinder by hand
   this%design_indicator = 0.0_rp
   this%brinkman_amplitude = 0.0_rp

   n = this%design_indicator%dof%size()
   do i = 1, n
     if(((this%design_indicator%dof%x(i,1,1,1) - 1.0_rp)**2 + (this%design_indicator%dof%y(i,1,1,1) - 0.5_rp)**2).lt.0.1_rp) then
     	 this%design_indicator%x(i,1,1,1) = 1.0_rp
     endif
   enddo

	! TODO
	! we would also need to make a mapping type that reads in parameters etc about filtering and mapping
	! ie, 
	! call mapper%init(this woud be from JSON)

	! and then we would map for the first one
	call this%map_forward()


	endsubroutine objective_function_init
	

	subroutine objective_function_map_forward(this)
	class(objective_function_t), target, intent(inout) :: this

	! TODO
	! this should be somehow deffered so we can pick different mappings!!!
	! so this would be:
	! call mapper%forward(fld_out, fld_in)
	call permeability_field(this%brinkman_amplitude, this%design_indicator, &
         & 0.0_rp, 10.0_rp, 1.0_rp)


	endsubroutine objective_function_map_forward

	subroutine objective_function_map_backward(this)
	class(objective_function_t), target, intent(inout) :: this
	! TODO
	! again..
	! so this would be:
	! call mapper%backward(fld_out, fld_in)

	endsubroutine objective_function_map_backward

	subroutine objective_function_free(this)
	class(objective_function_t), target, intent(inout) :: this
	call this%brinkman_amplitude%free()
	call this%design_indicator%free()

	endsubroutine objective_function_free
end module objective_function

