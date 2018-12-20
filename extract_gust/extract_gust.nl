


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
  default_model_name = "cosmo-1"
  location_to_gridpoint = "sn" 
  default_out_type_stdlongitude =.false.
! <GlobalSettings:originating_center>
! <GlobalSettings:production_status>
! <GlobalSettings:experiment_tag>
/ 

&ModelSpecification
 model_name         = "cosmo-1"
 earth_axis_large   = 6371229.
 earth_axis_small   = 6371229.
 hydrometeor        = "QR", "QG", "QS"
 precip_all         = "RAIN_GSP", "SNOW_GSP", "GRAU_GSP"
 precip_snow        = "SNOW_GSP", "GRAU_GSP"
 precip_rain        = "RAIN_GSP"
 precip_convective  =
 precip_gridscale   = "RAIN_GSP", "SNOW_GSP", "GRAU_GSP"
/



# Input for verification archive
#--------------------------------
&Process
  in_file  = "lfff00000000c",
  !in_regrid_target  = "GRID"
  in_regrid_target  = "__AUTO__" 
  out_file = "c1ffsurf000", out_type = "GRIB1",
  out_size_field = 2649
  out_cost = 1
/
&Process in_field = "HSURF" /
&Process in_field = "FR_LAND" /
&Process in_field = "SOILTYP" /

&Process
  in_file  =  "lfff<DDHH>0000"
  !in_regrid_target  = "GRID" 
  in_regrid_target  = "__AUTO__" 
  in_dictionary  = "__AUTO__" 
  out_file = "c1ffsurf<HHH>", out_type = "GRIB1",
  !out_size_field = 663
  out_size_field = 2649
  out_cost = 1
  tstart   = 0, tstop = 24, tincr = 1 /
&Process in_field = "U_10M" /
&Process in_field = "V_10M" /
&Process in_field = "VMAX_10M" /
