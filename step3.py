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

#
# Lets divide a 1d array of larr mesh points across the ranks,
# such that each rank has matching number of points within
# 1. Each divided part has nh halo points at each end that hold
# copies of points from its neighbors (rank-1 and rank+1).
#
nh=1    # halo width
larr=3*size
rem=larr%size
nbase=int(larr/size)
if rank < rem:
 myl=nbase+1
 mygstart=rank*(myl)
 mygend=mygstart+myl-1
else:
 myl=nbase
 mygstart=rank*(myl)+rem
 mygend=mygstart+myl-1
total_len=comm.allreduce(myl,op=MPI.SUM)
print( 'Rank %4d local array section length %d'%(rank,myl) )
print( 'Rank %4d total array length %d'%(rank,total_len)   )

mysec=np.zeros(myl+nh*2)
i0loh_indices=[*range(0,nh)]
i0hih_indices=[*range(myl+nh,myl+2*nh)]
i0sec_indices=[*range(nh,myl+nh)]

comm.Barrier()
print( 'Rank %4d local section is %6d to %6d'%(rank,mygstart,mygend)   )

MPI.Finalize()

