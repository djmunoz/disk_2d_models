#Python script to generate AREPO initial conditions of
#circumbinary disks

import numpy as np
import snapHDF5 as ws
import circumstellar as d
import matplotlib.pyplot as plt
import math
import numpy.random as ran
import sys
########################################################################
########################################################################
#COMMAND LINE PARAMETERS ########
q = float(sys.argv[1])  #binary mass ratio
eb = float(sys.argv[2]) #binary eccentricity
if (len(sys.argv) > 3):
  h0 = float(sys.argv[3]) #disk aspect ratio at a reference radius
else:
  h0 = 0.1
if (len(sys.argv) > 4):
  alpha = float(sys.argv[4]) #disk Shakura-Sunyaev viscosity coefficient
else:
  alpha = 0.1

print alpha


# OTHER PARAMETERS ##############
#mesh parameters
BoxX=160.0
BoxY=160.0
Nx=128
Ny=128
Nr=800
Ntheta=600
#Nr=192
#Ntheta=128

#disk parameters ##################
outer_rad=70.0
inner_rad=1 + eb
transition_rad=8.0

Sigma0 = 6.35983e-4
Sigmapeak = 2.6069978e-4
r_cav = 5.0
ErrTolIntAccuracy=0.06
CourantFac=0.3
MaxSizeTimestep=1.0
MinSizeTimestep=1.0e-8

l=1.0 #temperature profile
p=0.5 #density profile
xi=4.0 #cavity steepness
R_0=1.0
aspect_ratio_0=h0
csnd0 = aspect_ratio_0
alphacoeff=alpha


#mesh options
mesh_type="polar"
add_cells_center= False
add_cells_innerbound= True #False
add_cells_outerbound= True #False

test_functions = False


numerical_derivatives = True

########################################################################
#FUNCTIONS
def diskdens(R,p,xi,r_cav): # Disk surface density Sigma
  Sigma0 = Sigmapeak *(xi/p)**(p/xi)*np.exp(p/xi)
  return Sigma0 * (r_cav/R)**p * np.exp(-(r_cav/R)**xi) 

vecdiskdens = np.vectorize(diskdens)

x=np.logspace(-1,2,1000)
plt.plot(x,vecdiskdens(x,3.0,2.0,10),'b-')
plt.plot(x,vecdiskdens(x,p,xi,r_cav),'r-')
plt.show()

def soundspeed(R): # Disk sounds speed profile
  return csnd0*R**(-l*0.5)

vecsoundspeed = np.vectorize(soundspeed)

def Omegaprofilesq(R): # Disk orbital frequency from an axisymmetric gravitational potential
  return 1/R**3*(1.0 + 0.75 * eta/R**2*(1+1.5 * eb**2))
vecOmegaprofilesq = np.vectorize(Omegaprofilesq)


def velprofilesq(R):
  return (1/R*(1.0 + 0.75 * eta/R**2*(1+1.5 * eb**2)) +
	  soundspeed(R)**2*(-l - p  + xi * (r_cav/R)**xi))
vecvelprofilesq = np.vectorize(velprofilesq)

def viscosityprofile(R):
	return alphacoeff * soundspeed(R) * soundspeed(R) / np.sqrt(Omegaprofilesq(R))
vecviscosityprofile = np.vectorize(viscosityprofile)

#FUNCTION DERIVATIVES

def ddiskdens_dRprofile(R):
  return Sigma0 * (r_cav/R)**p * np.exp(-(r_cav/R)**xi) * (-p + xi *(r_cav/R)**xi)/R

vecddiskdens_dRprofile = np.vectorize(ddiskdens_dRprofile)

def dOmega_dRprofile(R):
  return 0.5*(-3/R**4*(1.0 + 0.75 * eta/R**2*(1+1.5 * eb**2))\
                 - 1.5*eta/R**6*(1+1.5 * eb**2))/ \
                 np.sqrt(1/R**3*(1.0 + 0.75 * eta/R**2*(1+1.5 * eb**2)))

vecdOmega_dRprofile = np.vectorize(dOmega_dRprofile)

def d2Omega_dR2profile(R):
  return (0.5*(+12/R**5*(1.0 + 0.75 * eta/R**2*(1+1.5 * eb**2))\
                  +13.5*eta/R**7*(1+1.5 * eb**2)) \
            - (0.25 * (-3/R**4*(1.0 + 0.75 * eta/R**2*(1+1.5 * eb**2))- 1.5*eta/R**6*(1+1.5 * eb**2))**2 \
                 /(1/R**3*(1.0 + 0.75 * eta/R**2*(1+1.5 * eb**2)))))/np.sqrt(1/R**3*(1.0 + 0.75 * eta/R**2*(1+1.5 * eb**2)))

vecd2Omega_dR2profile = np.vectorize(d2Omega_dR2profile)

def dviscosity_dRprofile(R):
  return alphacoeff * soundspeed(R)**2 * (-l/R/np.sqrt(Omegaprofilesq(R)) \
                                             -1/Omegaprofilesq(R) * dOmega_dRprofile(R))
vecdviscosity_dRprofile = np.vectorize(dviscosity_dRprofile)

def dvphi_dRprofile(R):
  return (-1/R*(1.0 + 0.75 * eta/R**2*(1+1.5 * eb**2)) + \
             -1.5 * eta/R**3*(1+1.5 * eb**2) \
             -l*soundspeed(R)**2*(-l - p  + xi * (r_cav/R)**xi)\
             -2 * xi *soundspeed(R)**2* (r_cav/R)**xi)/R

vecdvphi_dRprofile = np.vectorize(dvphi_dRprofile)




########################################################################
########################################################################
#INITIALIZE...

print "\n\nGenerating initial conditions for a circumbinary disk"
print "around a stellar binary with mass ratio %4.2f and eccentricity %5.3f\n" % (q,eb)

mu = q/(1.0+q)
eta = q/(1.0+q)**2


disk = d.disk(outer_radius=outer_rad, inner_radius=inner_rad, rho0 =Sigma0,
              l=l, p=p,R_0=R_0,
              mesh_type=mesh_type,mesh_alignment="interleaved",
              eos="isothermal",
              aspect_ratio=aspect_ratio_0,central_mass = 1.0,
              potential="keplerian",double=1,gamma=1.0001)
disk_inner = d.disk(outer_radius=transition_rad, inner_radius=inner_rad, rho0=Sigma0,
              l=l, p=p,R_0=R_0,
              mesh_type=mesh_type,mesh_alignment="interleaved",
              eos="isothermal",
              aspect_ratio=aspect_ratio_0,central_mass = 1.0,
              potential="keplerian",double=1,gamma=1.0001)
disk_outer = d.disk(outer_radius=outer_rad, inner_radius=transition_rad, rho0=Sigma0,
              l=l, p=p,R_0=R_0,
              mesh_type=mesh_type,mesh_alignment="interleaved",
              eos="isothermal",
              aspect_ratio=aspect_ratio_0,central_mass = 1.0,
              potential="keplerian",double=1,gamma=1.0001)


#get position of mesh-generating points
Nr_inner = int(np.log10(transition_rad/inner_rad)/np.pi * Ntheta)
Nr_outer = int(np.log10(outer_rad/transition_rad)/np.pi * Ntheta/1.5)
print "Radial zones: %i+%i=%i" % (Nr_inner,Nr_outer,Nr_inner+Nr_outer)
pos_inner,ids_inner = d.get_disk_grid_2D(disk_inner,N1=Nr_inner, N2=Ntheta,BoxSizeX=BoxX,BoxSizeY=BoxY)
pos_outer,ids_outer = d.get_disk_grid_2D(disk_outer,N1=Nr_outer, N2=Ntheta/2,BoxSizeX=BoxX,BoxSizeY=BoxY)

pos=np.concatenate((pos_inner,pos_outer),axis=0)
ids=np.append(ids_inner,np.arange(ids_inner.shape[0],ids_inner.shape[0]+ids_outer.shape[0]))
print "shape",pos.shape, ids.shape

#add a background grid (buffer cells)
pos,ids = d.add_background_grid_2D(disk,pos=pos,ids=ids,BoxSizeX=BoxX,BoxSizeY=BoxY,type="cartesian",
                                   Ncells=64, noise=0.05, equal_ids = -3)
base = "output_long/"

#add moving rings of cells to define the boundary to the region of computational interest
if (add_cells_outerbound):
  #first the outer boundary
  if (disk.mesh_type == 'cartesian'):
    Nphi = np.int(np.pi * Ny)
  elif (disk.mesh_type == 'ring'):
    Nphi = np.int(Nx*np.pi*2*outer_rad/(outer_rad-inner_rad))
  else:
    Nphi = Ntheta
  Nphi_out = Nphi
  #Nphi_out = Ntheta/2
  print "# of cells: Outermost annulus ",Nphi_out
  R = outer_rad
  thickness = float(2.0*math.pi*R/Nphi)
  pos,ids = d.add_ring_of_cells(pos=pos,ids=ids,R=R+1.5*thickness,Ntheta=Nphi_out,
                                BoxSizeX=BoxX,BoxSizeY=BoxY,equal_ids=-2)
  pos,ids = d.add_ring_of_cells(pos=pos,ids=ids,R=R+thickness/2.0,Ntheta=Nphi_out,
                                BoxSizeX=BoxX,BoxSizeY=BoxY,equal_ids=-2)
  pos,ids = d.add_ring_of_cells(pos=pos,ids=ids,R=R-thickness/2.0,Ntheta=Nphi_out,
                                BoxSizeX=BoxX,BoxSizeY=BoxY,equal_ids=-1)
  #and add some buffer cells outside the outer boundary
  pos,ids = d.add_ring_of_cells(pos=pos,ids=ids,R=R+thickness*2.5,Ntheta=Nphi_out,
                                BoxSizeX=BoxX,BoxSizeY=BoxY,equal_ids=-3)

if (add_cells_innerbound):
  #now the inner boundary
  if (disk.mesh_type == 'cartesian'):
    Nphi = np.int(np.pi * Ny)
  elif (disk.mesh_type == 'ring'):
    Nphi = np.int(Nx*np.pi*2*inner_rad/(outer_rad-inner_rad))
  else:
    Nphi = Ntheta
    Nphi_in = Nphi/2
    print "# of cells: Innermost annulus ",Nphi_in
  R = inner_rad
  thickness = float(2.0*math.pi*R/Nphi_in)
  #pos,ids = d.add_ring_of_cells(pos=pos,ids=ids,R=R+5*thickness/2.0,Ntheta=Nphi_in,
  #                              BoxSizeX=BoxX,BoxSizeY=BoxY,equal_ids=-1)
  pos,ids = d.add_ring_of_cells(pos=pos,ids=ids,R=R+3*thickness/2.0,Ntheta=Nphi_in,
                                BoxSizeX=BoxX,BoxSizeY=BoxY,equal_ids=-1)
  pos,ids = d.add_ring_of_cells(pos=pos,ids=ids,R=R+thickness/2.0,Ntheta=Nphi_in,
                                BoxSizeX=BoxX,BoxSizeY=BoxY,equal_ids=-1)
  pos,ids = d.add_ring_of_cells(pos=pos,ids=ids,R=R-thickness/2.0,Ntheta=Nphi_in,
                                BoxSizeX=BoxX,BoxSizeY=BoxY,equal_ids=-2)
  ##and add some buffer cells at the center
  pos,ids = d.add_ring_of_cells(pos=pos,ids=ids,R=R-thickness,Ntheta=Nphi_in,
                                BoxSizeX=BoxX,BoxSizeY=BoxY,equal_ids=-3)
  pos,ids = d.add_ring_of_cells(pos=pos,ids=ids,R=R-2*thickness,Ntheta=Nphi_in,
                                BoxSizeX=BoxX,BoxSizeY=BoxY,equal_ids=-3)

if (add_cells_center):
  #add some more background at the center
  #pos_center_x,pos_center_y = np.meshgrid(np.linspace(-inner_rad,inner_rad,30),
  #				        np.linspace(-inner_rad,inner_rad,30))
  delta = 2*np.pi * inner_rad / Nphi_in
  pos_center_x,pos_center_y = np.meshgrid(np.linspace(-inner_rad+0.5*delta,inner_rad-0.5*delta,int((2*inner_rad-delta)/delta/1.5)),
                                          np.linspace(-inner_rad+0.5*delta,inner_rad-0.5*delta,int((2*inner_rad-delta)/delta)/1.5))
  pos_center_x=pos_center_x.flatten()
  pos_center_y=pos_center_y.flatten()
  pos_center_rad = np.sqrt(pos_center_x**2 + pos_center_y**2)
  pos_center_x = pos_center_x[pos_center_rad < inner_rad]
  pos_center_y = pos_center_y[pos_center_rad < inner_rad]
  pos_center_x=pos_center_x + ran.normal(0,inner_rad/Nphi_in/2,len(pos_center_x))
  pos_center_y=pos_center_y + ran.normal(0,inner_rad/Nphi_in/2,len(pos_center_y))
  
  id_center = np.arange(ids.max()+1,ids.max()+1+pos_center_x.shape[0])
  ind_center = pos_center_x**2 + pos_center_y**2 < 0.8*inner_rad**2
  pos_center_x,pos_center_y = pos_center_x+BoxX*0.5, pos_center_y+BoxY*0.5
  pos_center=np.array([pos_center_x,pos_center_y,np.zeros(pos_center_x.shape[0])]).T
  pos=np.concatenate((pos,pos_center),axis=0)
  ids=np.append(ids,id_center)


#add some more background grid
pos,ids = d.add_ring_of_cells(pos=pos,ids=ids,R=outer_rad*1.08,Ntheta=160,
                              BoxSizeX=BoxX,BoxSizeY=BoxY,noise=0.05,equal_ids=-3)
pos,ids = d.add_ring_of_cells(pos=pos,ids=ids,R=outer_rad*1.13,Ntheta=80,
                              BoxSizeX=BoxX,BoxSizeY=BoxY,noise=0.05,equal_ids=-3)

print pos[ids == 72743,:]

#plt.scatter(pos[:,0],pos[:,1],s=1, c='b',marker='o')
#plt.show()

print ids[ids == -3].shape
ids  = np.where(ids > -3, ids[:],(np.arange(-3,-ids.shape[0]-3,-1))[:])
print ids.shape
print ids[ids == -3].shape
print ids[ids < -3].shape
#assign velocities, densities and internal energies
vel,dens,u = d.assign_quantities(disk=disk,pos=pos,BoxSizeX=BoxX,BoxSizeY=BoxY)

radius = np.sqrt((pos[:,0] - 0.5* BoxX)**2 + (pos[:,1] - 0.5* BoxY)**2)
phi = np.arctan2((pos[:,1] - 0.5* BoxY), (pos[:,0] - 0.5* BoxX))
#dens[radius < outer_rad] = Sigma0 * (r_cav/radius[radius < outer_rad])**p * np.exp(-(r_cav/radius[radius < outer_rad])**2) 
dens[radius < outer_rad] = vecdiskdens(radius[radius < outer_rad],p,xi,r_cav)
#dens[radius >= outer_rad] = Sigma0 * (r_cav/outer_rad)**p * np.exp(-(r_cav/outer_rad)**2) 
R_edge=2.5
dens_cut=vecdiskdens(R_edge,p,xi,r_cav)
print "Cut in density=",dens_cut
dens_out = vecdiskdens(outer_rad,p,xi,r_cav)
dens[radius >= outer_rad] = dens_out
print "Outer density",dens_out
dens[dens < dens_cut] = dens_cut
dens[radius < inner_rad] = dens.min()


plt.plot(pos[ids >0,0],pos[ids>0,1],'k.')
plt.plot(pos[ids ==-2,0],pos[ids ==-2,1],'b.')
plt.plot(pos[ids ==-1,0],pos[ids ==-1,1],'r.')
plt.axis([BoxX/2-2,BoxX/2+2,BoxY/2-2,BoxY/2+2])
plt.show()

csnd=csnd0*radius**(-l*0.5)
velphisq=  (1/radius*(1.0 + 0.75 * eta/radius**2*(1+1.5 * eb**2)) +
            csnd**2*(-l - p  + 2.0 * (r_cav/radius)**2))
velphisq[velphisq <= 0.0] = 0.0
velphi = np.sqrt(velphisq)

velrad = 1/radius/dens /(2* np.sqrt(vecOmegaprofilesq(radius))*radius + radius**2 * vecdOmega_dRprofile(radius)) * \
    ((vecdviscosity_dRprofile(radius) * radius**3 * dens * vecdOmega_dRprofile(radius)) \
       + (vecviscosityprofile(radius) * 3 * radius**2* dens * vecdOmega_dRprofile(radius)) \
       + (vecviscosityprofile(radius) * radius**3 * vecddiskdens_dRprofile(radius) * vecdOmega_dRprofile(radius))\
       + (vecviscosityprofile(radius) * radius**3 * dens * vecd2Omega_dR2profile(radius)))

velrad[radius > outer_rad] = 0.0
velphi[radius > outer_rad] = 0.0
#ids[radius > outer_rad] = -3
ind = dens <= dens_cut
velphi[ind] = velphi[ind]*(radius[ind]**2-inner_rad**2)/(R_edge**2-inner_rad**2)
velphi[radius < inner_rad] = 0

vel[:,0] = -velphi*np.sin(phi) + velrad * np.cos(phi)
vel[:,1] = +velphi*np.cos(phi) + velrad * np.sin(phi)



ids[ids >0]=np.arange(1,ids[ids>0].shape[0]+1)
#############################################################################################
if (test_functions):
  x = np.linspace((radius.min()),(radius.max()),500)
  func = np.sqrt(vecOmegaprofilesq(x))
  func2 = np.gradient(func,np.diff(x).mean())
  #plt.plot(radius,vecdOmega_dRprofile(radius),'bo')
  #plt.plot(x,func2,'red',lw=2)
  #plt.axis([3,70,-0.05,0])
  #plt.show()
  func2prime = np.gradient(func2,np.diff(x).mean())
  plt.plot(radius,vecd2Omega_dR2profile(radius),'bo')
  plt.plot(x,func2prime,'red',lw=2)
  plt.axis([3,70,-0.0001,0.002])
  plt.show()

  func3 = vecviscosityprofile(x) * x**3 * vecdiskdens(x,p,xi,r_cav) * func2
  func4 = np.gradient(func3,np.diff(x).mean())


  func5 = 1/x/ vecdiskdens(x,p,xi,r_cav) / (2 * func * x + x**2 * func2) * func4
  plt.plot(radius,velrad,'bo')
  plt.plot(x,func5,'red',lw=2)
  plt.axis([3,70,-1e-2,1e-2])
  plt.show()


#plt.plot(radius,velphi,'bo')
#plt.axis([1,10,0,1.5])
#plt.loglog(radius,dens,'bo')
#plt.axis([1,100,0,0.001])
plt.loglog(radius,velphi,'bo')
plt.loglog(np.sort(radius),1/np.sqrt(np.sort(radius)),'-')
#plt.axis([1,100,0,0.001])
plt.show()

accel = 1.0 / radius**2
if (mesh_type == "polar"):
  cellsize = 2.0*np.pi*radius/Ntheta
elif (mesh_type == "rings"):
  cellsize = 2.0*np.pi*inner_rad/N_in

timestep = np.minimum(CourantFac*cellsize/csnd,ErrTolIntAccuracy*np.sqrt(cellsize/accel))
timestep_nbin=np.log10(MaxSizeTimestep/MinSizeTimestep)/np.log10(2.0)
timestep_bins = 2.0**np.linspace(np.log10(MinSizeTimestep)/np.log10(2.0),np.log10(MaxSizeTimestep)/np.log10(2.0),timestep_nbin)
hist=np.histogram(timestep,bins=timestep_bins)
print hist

Ngas = pos.shape[0]
print "shape",pos.shape, vel.shape,dens.shape,ids.shape

f=ws.openfile("disk.dat.hdf5")
npart=np.array([Ngas,0,0,0,0,0], dtype="uint32")
massarr=np.array([0,0,0,0,0,0], dtype="float64")
header=ws.snapshot_header(npart=npart, nall=npart, massarr=massarr,
                          boxsize=BoxX, double = np.array([1], dtype="int32"))

        
ws.writeheader(f, header)
ws.write_block(f, "POS ", 0, pos)
ws.write_block(f, "VEL ", 0, vel)
ws.write_block(f, "MASS", 0, dens)
ws.write_block(f, "U   ", 0, u)
ws.write_block(f, "ID  ", 0, ids)
ws.closefile(f)

R_edge=2.5
print "Mass resolution inside the cavity",np.pi*(R_edge*2*np.pi/Nphi_in)**2*vecdiskdens(R_edge,p,xi,r_cav)
