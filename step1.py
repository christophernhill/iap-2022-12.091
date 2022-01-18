#!/usr/bin/env python

#
#  From N processes, each with their own id (rank), use MPI library to report
#  each process rank, total number of processes and the name of the host machine
#  on which the process resides.
#

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = MPI.Get_processor_name()
print('I am rank','%4d'%(rank),'of',size,'executing on',host)

MPI.Finalize()
