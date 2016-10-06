#read 2D reconstructed density field and plot grid on top
import matplotlib.pyplot as plt
import numpy as np
from pylab import * 
import readsnapHDF5 as rs
import sys
import snapHDF5 as ws
###############################################################################
#COMMAND-LINE PARAMETERS
num = int(sys.argv[1])

#OTHER PARAMETERS
# disk parameters
q=1.0
eb = 0.5
alphacoeff = 0.1
rho0 = 6.35983e-4
r_cav = 10.0
l=1.0 #temperature profile
p=3.0 #density profile
R_0=1.0
aspect_ratio_0=0.1
csnd = aspect_ratio_0
outer_rad=70.0
inner_rad=1 + eb

mu = q/(1.0+q)
eta = q/(1.0+q)**2

base="output_long/"

########################################################################
#FUNCTIONS
def diskdens(R):
  return rho0 * (r_cav/R)**p * np.exp(-(r_cav/R)**2) 

vecdiskdens = np.vectorize(diskdens)

def soundspeed(R):
	return csnd*R**(-l*0.5)

vecsoundspeed = np.vectorize(soundspeed)

def velprofilesq(R):
  return (1/R*(1.0 + 0.75 * eta/R**2*(1+1.5 * eb**2)) +
	  soundspeed(R)**2*(-l - p  + 2.0 * (r_cav/R)**2))
vecvelprofilesq = np.vectorize(velprofilesq)

def DvphiDRprofile(R):
  return (-1/R*(1.0 + 0.75 * eta/R**2*(1+1.5 * eb**2)) + \
             -1.5 * eta/R**3*(1+1.5 * eb**2) \
             -l*soundspeed(R)**2*(-l - p  + 2.0 * (r_cav/R)**2)\
             -4 *soundspeed(R)**2* (r_cav/R)**2)/R

vecDvphiDRprofile = np.vectorize(DvphiDRprofile)



def viscosityprofile(R):
	return alphacoeff * soundspeed(R) * soundspeed(R) / np.sqrt(R * R * R)

def velradprofile(R):
	return -3 * viscosityprofile(R) / R * (0.5 - p) \
	    -3 * viscosityprofile(R) / R * (- l - 1.5)
vecvelradprofile = np.vectorize(velradprofile)

################################################################################

x=linspace(1,30,200)
vphiprofilesq = vecvelprofilesq(x)
DvphiDRprofile = vecDvphiDRprofile(x)
vphiprofilesqgrad = np.gradient(vphiprofilesq,np.diff(x).mean())
#plt.plot(x,vphiprofilesqgrad,'ro',mew=0.0,ms=2.0)
#plt.plot(x,DvphiDRprofile,'b')
#plt.show()

#open the snapshot header
filename=base+"snap_"+str(num).zfill(3)
header = rs.snapshot_header(filename)
time = header.time
BoxX, BoxY = header.boxsize, header.boxsize


pos = rs.read_block(filename,"POS ",parttype=0)
vel  = rs.read_block(filename,"VEL ",parttype=0)
mass  = rs.read_block(filename,"MASS",parttype=0)
dens  = rs.read_block(filename,"RHO ",parttype=0)
vol   = rs.read_block(filename,"VOL ",parttype=0)
u   = rs.read_block(filename,"U   ",parttype=0)
ids  = rs.read_block(filename,"ID  ",parttype=0)

radius = np.sqrt((pos[:,0] - 0.5 * BoxX)**2 + (pos[:,1] - 0.5 * BoxY)**2)        
phi = np.arctan2((pos[:,1] - 0.5 * BoxY),(pos[:,0] - 0.5 * BoxX))

vphi = -np.sin(phi) * vel[:,0] + np.cos(phi) * vel[:,1]
vr   =  np.cos(phi) * vel[:,0] + np.sin(phi) * vel[:,1]

densprofile = vecdiskdens(radius[np.argsort(radius)])
vphiprofilesq = vecvelprofilesq(radius[np.argsort(radius)])
vrprofile = vecvelradprofile(radius[np.argsort(radius)])

#plt.plot(radius,dens,'ro',ms=0.8,mew=0.0)
#plt.plot(radius[np.argsort(radius)],densprofile,color='b',ms=0.8,mew=0.0)
#plt.axis([0,20,0,0.001])

#plt.plot(radius,vphi,'ro',ms=0.8,mew=0.0)
#plt.plot(radius[np.argsort(radius)],np.sqrt(vphiprofilesq),color='b',ms=0.8,mew=0.0)
#plt.axis([0,20,0,1.5])


#plt.plot(radius,vr,'ro',ms=0.8,mew=0.0)
#plt.plot(radius[np.argsort(radius)],vrprofile,color='b',ms=0.8,mew=0.0)
#plt.axis([15,20,0,1.0e-3])

#plt.show()

# replace primitive quantities in all cells meeting given criteria

#replace outside a certain radius
Rcut = 14

# impose a smooth transition between the original and the "corrected" values
indtrans = (radius > 0.9*Rcut) & (radius < 1.2*Rcut) & (ids > 0)
coeff = (radius[indtrans] - 0.9*Rcut)**2/(1.2 * Rcut - 0.9*Rcut)**2 
dens[indtrans] = coeff*vecdiskdens(radius[indtrans]) + (1-coeff)*dens[indtrans]
mass[indtrans] = dens[indtrans] * vol[indtrans] 
vphi[indtrans] = coeff*np.sqrt(vecvelprofilesq(radius[indtrans])) + (1-coeff)*vphi[indtrans]


#over-write the original values
ind = (radius > 1.2*Rcut) & (ids > 0)
dens[ind] = vecdiskdens(radius[ind])
mass[ind] = dens[ind] * vol[ind]
vphi[ind] = np.sqrt(vecvelprofilesq(radius[ind]))

vphi[radius > outer_rad] = 0.0
vel[:,0] = -vphi*np.sin(phi)
vel[:,1] = +vphi*np.cos(phi)

#plt.plot(radius,dens,'ro',ms=0.8,mew=0.0)
#plt.plot(radius[np.argsort(radius)],densprofile,color='b',ms=0.8,mew=0.0)
#plt.axis([0,20,0,0.001])

#plt.plot(radius,vphi,'ro',ms=0.8,mew=0.0)
#plt.plot(radius[np.argsort(radius)],np.sqrt(vphiprofilesq),color='b',ms=0.8,mew=0.0)
#plt.axis([0,20,0,1.5])

#plt.show()


#write to disk the "fixed" snapshot 
Ngas = pos.shape[0]

filename_fixed =base+"snap_fixed_"+str(num).zfill(3)+".hdf5"
f=ws.openfile(filename_fixed)
npart=np.array([Ngas,0,0,0,0,0], dtype="uint32")
massarr=np.array([0,0,0,0,0,0], dtype="float64")
header=ws.snapshot_header(npart=npart, nall=npart, massarr=massarr,time=time,
                          boxsize=BoxX, double = np.array([1], dtype="int32"))

        
ws.writeheader(f, header)
ws.write_block(f, "POS ", 0, pos)
ws.write_block(f, "VEL ", 0, vel)
ws.write_block(f, "MASS", 0, mass)
ws.write_block(f, "RHO ", 0, dens)
ws.write_block(f, "U   ", 0, u)
ws.write_block(f, "ID  ", 0, ids)
ws.write_block(f, "VOL ", 0, vol)
ws.closefile(f)
