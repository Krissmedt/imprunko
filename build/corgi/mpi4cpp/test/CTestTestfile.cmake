# CMake generated Testfile for 
# Source directory: /home/krissmedt/Documents/runko/corgi/mpi4cpp/test
# Build directory: /home/krissmedt/Documents/runko/build/corgi/mpi4cpp/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(init "/usr/bin/mpiexec" "-n" "2" "./init")
add_test(send_recv_nodata "/usr/bin/mpiexec" "-n" "2" "./send_recv_nodata")
add_test(send_recv "/usr/bin/mpiexec" "-n" "2" "./send_recv")
add_test(send_recv_types "/usr/bin/mpiexec" "-n" "2" "./send_recv_types")
add_test(arrays "/usr/bin/mpiexec" "-n" "2" "./arrays")
add_test(isend_irecv_nodata "/usr/bin/mpiexec" "-n" "2" "./isend_irecv_nodata")
add_test(isend_irecv "/usr/bin/mpiexec" "-n" "2" "./isend_irecv")
add_test(isend_irecv_types "/usr/bin/mpiexec" "-n" "2" "./isend_irecv_types")
add_test(iarrays "/usr/bin/mpiexec" "-n" "2" "./iarrays")
add_test(own_datatype "/usr/bin/mpiexec" "-n" "2" "./own_datatype")
