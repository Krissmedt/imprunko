# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/krissmedt/Documents/runko

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/krissmedt/Documents/runko/build

# Include any dependencies generated for this target.
include corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/depend.make

# Include the progress variables for this target.
include corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/progress.make

# Include the compile flags for this target's objects.
include corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/flags.make

corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o: corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/flags.make
corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o: ../corgi/mpi4cpp/test/test_iarrays.c++
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/krissmedt/Documents/runko/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o"
	cd /home/krissmedt/Documents/runko/build/corgi/mpi4cpp/test && /usr/bin/mpic++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/iarrays.dir/test_iarrays.c++.o -c /home/krissmedt/Documents/runko/corgi/mpi4cpp/test/test_iarrays.c++

corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/iarrays.dir/test_iarrays.c++.i"
	cd /home/krissmedt/Documents/runko/build/corgi/mpi4cpp/test && /usr/bin/mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/krissmedt/Documents/runko/corgi/mpi4cpp/test/test_iarrays.c++ > CMakeFiles/iarrays.dir/test_iarrays.c++.i

corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/iarrays.dir/test_iarrays.c++.s"
	cd /home/krissmedt/Documents/runko/build/corgi/mpi4cpp/test && /usr/bin/mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/krissmedt/Documents/runko/corgi/mpi4cpp/test/test_iarrays.c++ -o CMakeFiles/iarrays.dir/test_iarrays.c++.s

corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o.requires:

.PHONY : corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o.requires

corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o.provides: corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o.requires
	$(MAKE) -f corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/build.make corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o.provides.build
.PHONY : corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o.provides

corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o.provides.build: corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o


# Object files for target iarrays
iarrays_OBJECTS = \
"CMakeFiles/iarrays.dir/test_iarrays.c++.o"

# External object files for target iarrays
iarrays_EXTERNAL_OBJECTS =

corgi/mpi4cpp/test/iarrays: corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o
corgi/mpi4cpp/test/iarrays: corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/build.make
corgi/mpi4cpp/test/iarrays: corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/krissmedt/Documents/runko/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable iarrays"
	cd /home/krissmedt/Documents/runko/build/corgi/mpi4cpp/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/iarrays.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/build: corgi/mpi4cpp/test/iarrays

.PHONY : corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/build

corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/requires: corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/test_iarrays.c++.o.requires

.PHONY : corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/requires

corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/clean:
	cd /home/krissmedt/Documents/runko/build/corgi/mpi4cpp/test && $(CMAKE_COMMAND) -P CMakeFiles/iarrays.dir/cmake_clean.cmake
.PHONY : corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/clean

corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/depend:
	cd /home/krissmedt/Documents/runko/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/krissmedt/Documents/runko /home/krissmedt/Documents/runko/corgi/mpi4cpp/test /home/krissmedt/Documents/runko/build /home/krissmedt/Documents/runko/build/corgi/mpi4cpp/test /home/krissmedt/Documents/runko/build/corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : corgi/mpi4cpp/test/CMakeFiles/iarrays.dir/depend

