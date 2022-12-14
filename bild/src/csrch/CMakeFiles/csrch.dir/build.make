# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/exp/DGSOL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/exp/DGSOL/bild

# Include any dependencies generated for this target.
include src/csrch/CMakeFiles/csrch.dir/depend.make

# Include the progress variables for this target.
include src/csrch/CMakeFiles/csrch.dir/progress.make

# Include the compile flags for this target's objects.
include src/csrch/CMakeFiles/csrch.dir/flags.make

src/csrch/CMakeFiles/csrch.dir/dcsrch.f.o: src/csrch/CMakeFiles/csrch.dir/flags.make
src/csrch/CMakeFiles/csrch.dir/dcsrch.f.o: ../src/csrch/dcsrch.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building Fortran object src/csrch/CMakeFiles/csrch.dir/dcsrch.f.o"
	cd /data/exp/DGSOL/bild/src/csrch && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/csrch/dcsrch.f -o CMakeFiles/csrch.dir/dcsrch.f.o

src/csrch/CMakeFiles/csrch.dir/dcsrch.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/csrch.dir/dcsrch.f.i"
	cd /data/exp/DGSOL/bild/src/csrch && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/csrch/dcsrch.f > CMakeFiles/csrch.dir/dcsrch.f.i

src/csrch/CMakeFiles/csrch.dir/dcsrch.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/csrch.dir/dcsrch.f.s"
	cd /data/exp/DGSOL/bild/src/csrch && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/csrch/dcsrch.f -o CMakeFiles/csrch.dir/dcsrch.f.s

src/csrch/CMakeFiles/csrch.dir/dcstep.f.o: src/csrch/CMakeFiles/csrch.dir/flags.make
src/csrch/CMakeFiles/csrch.dir/dcstep.f.o: ../src/csrch/dcstep.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building Fortran object src/csrch/CMakeFiles/csrch.dir/dcstep.f.o"
	cd /data/exp/DGSOL/bild/src/csrch && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/csrch/dcstep.f -o CMakeFiles/csrch.dir/dcstep.f.o

src/csrch/CMakeFiles/csrch.dir/dcstep.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/csrch.dir/dcstep.f.i"
	cd /data/exp/DGSOL/bild/src/csrch && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/csrch/dcstep.f > CMakeFiles/csrch.dir/dcstep.f.i

src/csrch/CMakeFiles/csrch.dir/dcstep.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/csrch.dir/dcstep.f.s"
	cd /data/exp/DGSOL/bild/src/csrch && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/csrch/dcstep.f -o CMakeFiles/csrch.dir/dcstep.f.s

# Object files for target csrch
csrch_OBJECTS = \
"CMakeFiles/csrch.dir/dcsrch.f.o" \
"CMakeFiles/csrch.dir/dcstep.f.o"

# External object files for target csrch
csrch_EXTERNAL_OBJECTS =

src/csrch/libcsrch.a: src/csrch/CMakeFiles/csrch.dir/dcsrch.f.o
src/csrch/libcsrch.a: src/csrch/CMakeFiles/csrch.dir/dcstep.f.o
src/csrch/libcsrch.a: src/csrch/CMakeFiles/csrch.dir/build.make
src/csrch/libcsrch.a: src/csrch/CMakeFiles/csrch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking Fortran static library libcsrch.a"
	cd /data/exp/DGSOL/bild/src/csrch && $(CMAKE_COMMAND) -P CMakeFiles/csrch.dir/cmake_clean_target.cmake
	cd /data/exp/DGSOL/bild/src/csrch && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/csrch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/csrch/CMakeFiles/csrch.dir/build: src/csrch/libcsrch.a

.PHONY : src/csrch/CMakeFiles/csrch.dir/build

src/csrch/CMakeFiles/csrch.dir/clean:
	cd /data/exp/DGSOL/bild/src/csrch && $(CMAKE_COMMAND) -P CMakeFiles/csrch.dir/cmake_clean.cmake
.PHONY : src/csrch/CMakeFiles/csrch.dir/clean

src/csrch/CMakeFiles/csrch.dir/depend:
	cd /data/exp/DGSOL/bild && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/exp/DGSOL /data/exp/DGSOL/src/csrch /data/exp/DGSOL/bild /data/exp/DGSOL/bild/src/csrch /data/exp/DGSOL/bild/src/csrch/CMakeFiles/csrch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/csrch/CMakeFiles/csrch.dir/depend

