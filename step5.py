#!/usr/bin/env python

#
#  From N processes, each with their own id (rank), use MPI library to report
#  each process rank, total number of processes and the name of the host machine
#  on which the process resides.
#

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

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
nh=2    # halo width
larr=10000
larr=size*100
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
#
# Each rank writes rank number to the local sections of mysec
# and then sends and receives updates from neighbors
#
if nh > myl:
    print('ERROR: nh must be less than myl')

mysec[i0sec_indices]=rank
i0rm1=rank-1
i0rp1=rank+1
if rank == 0: 
  i0rm1=size-1
if rank == size-1: 
  i0rp1=0
# Send to my n-1 side and receive from my n+1 side
comm.send(mysec[i0sec_indices][0:nh],dest=i0rm1)
mysec[i0hih_indices]=comm.recv(source=i0rp1)
# Send to my n+1 side and receive from my n-1 side
comm.send(mysec[i0sec_indices][-nh:],dest=i0rp1)
mysec[i0loh_indices]=comm.recv(source=i0rm1)

#
# Now lets check what is in each section
#
print('Rank %4d mysec values ='%(rank),mysec)

def halo(fld,isec,im1,ip1,ihi,ilo,nh):
  comm.send(fld[isec][0:nh],dest=im1)
  fld[ihi]=comm.recv(source=ip1)
  comm.send(fld[isec][-nh:],dest=ip1)
  fld[ilo]=comm.recv(source=im1)

#
# OK now lets start thinking about a numerical problem
# - we will use several variables
#   1. field values and the first and second derivatives in x
#   2. grid locations
#   3. diffusion coefficient
#   4. initial conditions
#   5. timestep, domain size

# Set parameters
Lx=1.
dx=Lx/larr
rdx=1./dx
dt=5.

# 1. setting initial conditions
# Lets create an array of grid locations
xc=np.zeros(myl+nh*2)
xc[i0sec_indices]=( np.linspace(mygstart,mygend,myl)+0.5 )*dx
print('Rank %4d xc values ='%(rank),xc)
halo(xc,i0sec_indices,i0rm1,i0rp1,i0hih_indices,i0loh_indices,nh)
print('Rank %4d xc values ='%(rank),xc)

phi_init=np.zeros(myl+nh*2)
phi_init[i0sec_indices]=np.exp( -50.*(xc[i0sec_indices]/Lx-1./2.)**2 )
print('Rank %4d phi_init values ='%(rank),phi_init)
halo(phi_init,i0sec_indices,i0rm1,i0rp1,i0hih_indices,i0loh_indices,nh)
print('Rank %4d phi_init values ='%(rank),phi_init)

plt.plot(phi_init)
plt.savefig('phi_init_rank%d.png'%(rank))

MPI.Finalize()

