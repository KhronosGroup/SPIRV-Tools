find_program(NOSETESTS_EXECUTABLE NAMES nosetests PATHS $ENV{PYTHON_PACKAGE_PATH})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nosetests REQUIRED_VARS NOSETESTS_EXECUTABLE
  FOUND_VAR nosetests_FOUND
  FAIL_MESSAGE "nosetests was not found - python support code will not be tested")

if(nosetests_FOUND AND NOT TARGET nosetests::nosetests)
  add_executable(nosetests::nosetests IMPORTED)
  set_target_properties(nosetests::nosetests PROPERTIES
    IMPORTED_LOCATION ${NOSETESTS_EXECUTABLE})

  mark_as_advanced(NOSETESTS_EXECUTABLE)
endif()

# Run nosetests on file ${PREFIX}_nosetest.py. Nosetests will look for classes
# and functions whose names start with "nosetest". The test name will be
# ${PREFIX}_nosetests.
function(spirv_add_nosetests PREFIX)
  if(NOT "${SPIRV_SKIP_TESTS}" AND NOSETESTS_EXE)
    add_test(
      NAME ${PREFIX}_nosetests
      COMMAND nosetests::nosetests -m "^[Nn]ose[Tt]est" -v
        ${CMAKE_CURRENT_SOURCE_DIR}/${PREFIX}_nosetest.py)
  endif()
endfunction()
