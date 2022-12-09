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
include src/blas/CMakeFiles/blas.dir/depend.make

# Include the progress variables for this target.
include src/blas/CMakeFiles/blas.dir/progress.make

# Include the compile flags for this target's objects.
include src/blas/CMakeFiles/blas.dir/flags.make

src/blas/CMakeFiles/blas.dir/dasum.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dasum.f.o: ../src/blas/dasum.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building Fortran object src/blas/CMakeFiles/blas.dir/dasum.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dasum.f -o CMakeFiles/blas.dir/dasum.f.o

src/blas/CMakeFiles/blas.dir/dasum.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dasum.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dasum.f > CMakeFiles/blas.dir/dasum.f.i

src/blas/CMakeFiles/blas.dir/dasum.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dasum.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dasum.f -o CMakeFiles/blas.dir/dasum.f.s

src/blas/CMakeFiles/blas.dir/daxpy.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/daxpy.f.o: ../src/blas/daxpy.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building Fortran object src/blas/CMakeFiles/blas.dir/daxpy.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/daxpy.f -o CMakeFiles/blas.dir/daxpy.f.o

src/blas/CMakeFiles/blas.dir/daxpy.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/daxpy.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/daxpy.f > CMakeFiles/blas.dir/daxpy.f.i

src/blas/CMakeFiles/blas.dir/daxpy.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/daxpy.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/daxpy.f -o CMakeFiles/blas.dir/daxpy.f.s

src/blas/CMakeFiles/blas.dir/dcopy.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dcopy.f.o: ../src/blas/dcopy.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building Fortran object src/blas/CMakeFiles/blas.dir/dcopy.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dcopy.f -o CMakeFiles/blas.dir/dcopy.f.o

src/blas/CMakeFiles/blas.dir/dcopy.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dcopy.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dcopy.f > CMakeFiles/blas.dir/dcopy.f.i

src/blas/CMakeFiles/blas.dir/dcopy.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dcopy.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dcopy.f -o CMakeFiles/blas.dir/dcopy.f.s

src/blas/CMakeFiles/blas.dir/ddot.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/ddot.f.o: ../src/blas/ddot.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building Fortran object src/blas/CMakeFiles/blas.dir/ddot.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/ddot.f -o CMakeFiles/blas.dir/ddot.f.o

src/blas/CMakeFiles/blas.dir/ddot.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/ddot.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/ddot.f > CMakeFiles/blas.dir/ddot.f.i

src/blas/CMakeFiles/blas.dir/ddot.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/ddot.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/ddot.f -o CMakeFiles/blas.dir/ddot.f.s

src/blas/CMakeFiles/blas.dir/dgemm.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dgemm.f.o: ../src/blas/dgemm.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building Fortran object src/blas/CMakeFiles/blas.dir/dgemm.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dgemm.f -o CMakeFiles/blas.dir/dgemm.f.o

src/blas/CMakeFiles/blas.dir/dgemm.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dgemm.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dgemm.f > CMakeFiles/blas.dir/dgemm.f.i

src/blas/CMakeFiles/blas.dir/dgemm.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dgemm.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dgemm.f -o CMakeFiles/blas.dir/dgemm.f.s

src/blas/CMakeFiles/blas.dir/dgemv.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dgemv.f.o: ../src/blas/dgemv.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building Fortran object src/blas/CMakeFiles/blas.dir/dgemv.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dgemv.f -o CMakeFiles/blas.dir/dgemv.f.o

src/blas/CMakeFiles/blas.dir/dgemv.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dgemv.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dgemv.f > CMakeFiles/blas.dir/dgemv.f.i

src/blas/CMakeFiles/blas.dir/dgemv.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dgemv.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dgemv.f -o CMakeFiles/blas.dir/dgemv.f.s

src/blas/CMakeFiles/blas.dir/dger.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dger.f.o: ../src/blas/dger.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building Fortran object src/blas/CMakeFiles/blas.dir/dger.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dger.f -o CMakeFiles/blas.dir/dger.f.o

src/blas/CMakeFiles/blas.dir/dger.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dger.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dger.f > CMakeFiles/blas.dir/dger.f.i

src/blas/CMakeFiles/blas.dir/dger.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dger.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dger.f -o CMakeFiles/blas.dir/dger.f.s

src/blas/CMakeFiles/blas.dir/dnrm2.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dnrm2.f.o: ../src/blas/dnrm2.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building Fortran object src/blas/CMakeFiles/blas.dir/dnrm2.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dnrm2.f -o CMakeFiles/blas.dir/dnrm2.f.o

src/blas/CMakeFiles/blas.dir/dnrm2.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dnrm2.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dnrm2.f > CMakeFiles/blas.dir/dnrm2.f.i

src/blas/CMakeFiles/blas.dir/dnrm2.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dnrm2.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dnrm2.f -o CMakeFiles/blas.dir/dnrm2.f.s

src/blas/CMakeFiles/blas.dir/dscal.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dscal.f.o: ../src/blas/dscal.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building Fortran object src/blas/CMakeFiles/blas.dir/dscal.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dscal.f -o CMakeFiles/blas.dir/dscal.f.o

src/blas/CMakeFiles/blas.dir/dscal.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dscal.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dscal.f > CMakeFiles/blas.dir/dscal.f.i

src/blas/CMakeFiles/blas.dir/dscal.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dscal.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dscal.f -o CMakeFiles/blas.dir/dscal.f.s

src/blas/CMakeFiles/blas.dir/dswap.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dswap.f.o: ../src/blas/dswap.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building Fortran object src/blas/CMakeFiles/blas.dir/dswap.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dswap.f -o CMakeFiles/blas.dir/dswap.f.o

src/blas/CMakeFiles/blas.dir/dswap.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dswap.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dswap.f > CMakeFiles/blas.dir/dswap.f.i

src/blas/CMakeFiles/blas.dir/dswap.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dswap.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dswap.f -o CMakeFiles/blas.dir/dswap.f.s

src/blas/CMakeFiles/blas.dir/dsyrk.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dsyrk.f.o: ../src/blas/dsyrk.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building Fortran object src/blas/CMakeFiles/blas.dir/dsyrk.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dsyrk.f -o CMakeFiles/blas.dir/dsyrk.f.o

src/blas/CMakeFiles/blas.dir/dsyrk.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dsyrk.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dsyrk.f > CMakeFiles/blas.dir/dsyrk.f.i

src/blas/CMakeFiles/blas.dir/dsyrk.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dsyrk.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dsyrk.f -o CMakeFiles/blas.dir/dsyrk.f.s

src/blas/CMakeFiles/blas.dir/dtrmm.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dtrmm.f.o: ../src/blas/dtrmm.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building Fortran object src/blas/CMakeFiles/blas.dir/dtrmm.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dtrmm.f -o CMakeFiles/blas.dir/dtrmm.f.o

src/blas/CMakeFiles/blas.dir/dtrmm.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dtrmm.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dtrmm.f > CMakeFiles/blas.dir/dtrmm.f.i

src/blas/CMakeFiles/blas.dir/dtrmm.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dtrmm.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dtrmm.f -o CMakeFiles/blas.dir/dtrmm.f.s

src/blas/CMakeFiles/blas.dir/dtrmv.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dtrmv.f.o: ../src/blas/dtrmv.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building Fortran object src/blas/CMakeFiles/blas.dir/dtrmv.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dtrmv.f -o CMakeFiles/blas.dir/dtrmv.f.o

src/blas/CMakeFiles/blas.dir/dtrmv.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dtrmv.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dtrmv.f > CMakeFiles/blas.dir/dtrmv.f.i

src/blas/CMakeFiles/blas.dir/dtrmv.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dtrmv.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dtrmv.f -o CMakeFiles/blas.dir/dtrmv.f.s

src/blas/CMakeFiles/blas.dir/dtrsm.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dtrsm.f.o: ../src/blas/dtrsm.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building Fortran object src/blas/CMakeFiles/blas.dir/dtrsm.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dtrsm.f -o CMakeFiles/blas.dir/dtrsm.f.o

src/blas/CMakeFiles/blas.dir/dtrsm.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dtrsm.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dtrsm.f > CMakeFiles/blas.dir/dtrsm.f.i

src/blas/CMakeFiles/blas.dir/dtrsm.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dtrsm.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dtrsm.f -o CMakeFiles/blas.dir/dtrsm.f.s

src/blas/CMakeFiles/blas.dir/dtrsv.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/dtrsv.f.o: ../src/blas/dtrsv.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building Fortran object src/blas/CMakeFiles/blas.dir/dtrsv.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/dtrsv.f -o CMakeFiles/blas.dir/dtrsv.f.o

src/blas/CMakeFiles/blas.dir/dtrsv.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/dtrsv.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/dtrsv.f > CMakeFiles/blas.dir/dtrsv.f.i

src/blas/CMakeFiles/blas.dir/dtrsv.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/dtrsv.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/dtrsv.f -o CMakeFiles/blas.dir/dtrsv.f.s

src/blas/CMakeFiles/blas.dir/idamax.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/idamax.f.o: ../src/blas/idamax.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building Fortran object src/blas/CMakeFiles/blas.dir/idamax.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/idamax.f -o CMakeFiles/blas.dir/idamax.f.o

src/blas/CMakeFiles/blas.dir/idamax.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/idamax.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/idamax.f > CMakeFiles/blas.dir/idamax.f.i

src/blas/CMakeFiles/blas.dir/idamax.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/idamax.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/idamax.f -o CMakeFiles/blas.dir/idamax.f.s

src/blas/CMakeFiles/blas.dir/lsame.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/lsame.f.o: ../src/blas/lsame.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building Fortran object src/blas/CMakeFiles/blas.dir/lsame.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/lsame.f -o CMakeFiles/blas.dir/lsame.f.o

src/blas/CMakeFiles/blas.dir/lsame.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/lsame.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/lsame.f > CMakeFiles/blas.dir/lsame.f.i

src/blas/CMakeFiles/blas.dir/lsame.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/lsame.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/lsame.f -o CMakeFiles/blas.dir/lsame.f.s

src/blas/CMakeFiles/blas.dir/xerbla.f.o: src/blas/CMakeFiles/blas.dir/flags.make
src/blas/CMakeFiles/blas.dir/xerbla.f.o: ../src/blas/xerbla.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Building Fortran object src/blas/CMakeFiles/blas.dir/xerbla.f.o"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /data/exp/DGSOL/src/blas/xerbla.f -o CMakeFiles/blas.dir/xerbla.f.o

src/blas/CMakeFiles/blas.dir/xerbla.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/blas.dir/xerbla.f.i"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /data/exp/DGSOL/src/blas/xerbla.f > CMakeFiles/blas.dir/xerbla.f.i

src/blas/CMakeFiles/blas.dir/xerbla.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/blas.dir/xerbla.f.s"
	cd /data/exp/DGSOL/bild/src/blas && /bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /data/exp/DGSOL/src/blas/xerbla.f -o CMakeFiles/blas.dir/xerbla.f.s

# Object files for target blas
blas_OBJECTS = \
"CMakeFiles/blas.dir/dasum.f.o" \
"CMakeFiles/blas.dir/daxpy.f.o" \
"CMakeFiles/blas.dir/dcopy.f.o" \
"CMakeFiles/blas.dir/ddot.f.o" \
"CMakeFiles/blas.dir/dgemm.f.o" \
"CMakeFiles/blas.dir/dgemv.f.o" \
"CMakeFiles/blas.dir/dger.f.o" \
"CMakeFiles/blas.dir/dnrm2.f.o" \
"CMakeFiles/blas.dir/dscal.f.o" \
"CMakeFiles/blas.dir/dswap.f.o" \
"CMakeFiles/blas.dir/dsyrk.f.o" \
"CMakeFiles/blas.dir/dtrmm.f.o" \
"CMakeFiles/blas.dir/dtrmv.f.o" \
"CMakeFiles/blas.dir/dtrsm.f.o" \
"CMakeFiles/blas.dir/dtrsv.f.o" \
"CMakeFiles/blas.dir/idamax.f.o" \
"CMakeFiles/blas.dir/lsame.f.o" \
"CMakeFiles/blas.dir/xerbla.f.o"

# External object files for target blas
blas_EXTERNAL_OBJECTS =

src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dasum.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/daxpy.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dcopy.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/ddot.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dgemm.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dgemv.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dger.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dnrm2.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dscal.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dswap.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dsyrk.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dtrmm.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dtrmv.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dtrsm.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/dtrsv.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/idamax.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/lsame.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/xerbla.f.o
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/build.make
src/blas/libblas.a: src/blas/CMakeFiles/blas.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/exp/DGSOL/bild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_19) "Linking Fortran static library libblas.a"
	cd /data/exp/DGSOL/bild/src/blas && $(CMAKE_COMMAND) -P CMakeFiles/blas.dir/cmake_clean_target.cmake
	cd /data/exp/DGSOL/bild/src/blas && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/blas.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/blas/CMakeFiles/blas.dir/build: src/blas/libblas.a

.PHONY : src/blas/CMakeFiles/blas.dir/build

src/blas/CMakeFiles/blas.dir/clean:
	cd /data/exp/DGSOL/bild/src/blas && $(CMAKE_COMMAND) -P CMakeFiles/blas.dir/cmake_clean.cmake
.PHONY : src/blas/CMakeFiles/blas.dir/clean

src/blas/CMakeFiles/blas.dir/depend:
	cd /data/exp/DGSOL/bild && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/exp/DGSOL /data/exp/DGSOL/src/blas /data/exp/DGSOL/bild /data/exp/DGSOL/bild/src/blas /data/exp/DGSOL/bild/src/blas/CMakeFiles/blas.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/blas/CMakeFiles/blas.dir/depend
