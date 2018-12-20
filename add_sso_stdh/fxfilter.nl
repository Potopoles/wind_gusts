&RunSpecification
  soft_memory_limit = 0
  verbosity = "silent"
  diagnostic_length = 0
  strict_usage = .false.
/

&GlobalResource
  dictionary = "/users/oprusers/owm/opr/resources/dictionary_cosmo.txt","/users/oprusers/owm/opr/resources/dictionary_default.txt","/users/oprusers/owm/opr/resources/dictionary_ifs.txt","/users/oprusers/owm/opr/resources/dictionary_observation.txt"
  grib_definition_path = "/users/oprusers/owm/bin/support/grib_api_definitions_1","/users/oprusers/owm/bin/support/grib_api_definitions_2"
  grib2_sample = "/users/oprusers/owm/modules/kesch/libgrib-api-cosmo-resources/1.20.0.3/samples/COSMO_GRIB2_default.tmpl"
/

&GlobalSettings
  default_dictionary = "cosmo",
  default_model_name = "unknown"
  location_to_gridpoint = "sn" 
  default_out_type_stdlongitude =.false.
! <GlobalSettings:originating_center>
! <GlobalSettings:production_status>
! <GlobalSettings:experiment_tag>
/ 

&ModelSpecification
  model_name = "unknown"
/

&Process 
  in_file = "/scratch-shared/meteoswiss/scratch/heimc/cache/add_sso_stdh/laf_template"
  in_regrid_target = "__AUTO__"
  in_dictionary = "__AUTO__"
  out_file = "./SSO_STDH"
  out_type = "GRIB1"
  out_type_largenc = .false.
  out_type_autoncname = .false.
  out_type_noundef = .true.
  out_type_nousetag = .true.
  out_duplicate_infield = .true.
  out_size_field = 663
  out_size_vdate = 1
  selection_mode  = "INCLUDE_ONLY"
/
&Process 
  in_field = "SSO_STDH", set_reference_date=201806031200, level_class="all", levlist=-1,
/
