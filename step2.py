#!/usr/bin/env python

#
#  From N processes, each with their own id (rank), use MPI library to report
#  each process rank, total number of processes and the name of the host machine
#  on which the process resides.
#

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = MPI.Get_processor_name()
print('I am rank','%4d'%(rank),'of',size,'executing on',host)

#
#  Have all the processes send a message to the
#  one lower rank process, except for rank 0 which 
#  sends to the highest rank process.
#
send_message='Hello from rank %d'%(rank)
msg_dest=rank-1
msg_srce=rank+1
if rank == 0: 
  msg_dest=size-1
if rank == size-1: 
  msg_srce=0
comm.send(send_message,dest=msg_dest)
recv_message=comm.recv(source=msg_srce)
print( 'I just sent "%s" to %4d'%(send_message,msg_dest) )
print( 'I just received "%s" from %4d'%(recv_message,msg_srce) )

MPI.Finalize()

