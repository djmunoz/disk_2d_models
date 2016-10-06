
#routines for creating AREPO grids in different regular geometries for
#circumstellar disk simulations

import numpy as np
import os
import math
from circumstellar import *
import scipy.special as spec
import matplotlib.pyplot as plt
import random as ran
from scipy.stats import uniform

#################################################################################################
#constants

#################################################################################################
class Box:
    def __init__(self, *args, **kwargs):
        #read arguments
        self.X = kwargs.get("X")
        self.Y = kwargs.get("Y")
        self.Z = kwargs.get("Z")

########################### 
#CLASS FOR DISK PROPERTIES
###########################  
class disk:
    def __init__(self, *args, **kwargs):
        #read arguments
        self.eos = kwargs.get("eos")
        self.potential = kwargs.get("potential")
        self.l = kwargs.get("l")
        self.p = kwargs.get("p")
        self.rho0 = kwargs.get("rho0")
        self.R_0 = kwargs.get("R_0")
        self.gamma = kwargs.get("gamma")
        self.aspect_ratio = kwargs.get("aspect_ratio")
        self.inner_radius = kwargs.get("inner_radius")
        self.outer_radius = kwargs.get("outer_radius")
        self.truncation = kwargs.get("truncation")
        self.central_mass = kwargs.get("central_mass")
        self.star_softening = kwargs.get("star_softnening")
        self.gravityG = kwargs.get("gravityG")
        self.profile_type = kwargs.get("profile_type")
        self.ndims = kwargs.get("ndims")
        self.mesh_type = kwargs.get("mesh_type")
        self.mesh_alignment = kwargs.get("mesh_alignment")
        self.double = kwargs.get("double")
        
        
        #set default values
        if (self.eos == None):
            self.eos = "isothermal"
            
        if (self.potential == None ):
            self.potential = "keplerian"

        if (self.l == None ):
            self.l = np.array([1], dtype="float64")

        if (self.p == None ):
            self.p =  np.array([1], dtype="float64")

        if (self.rho0 == None ):
            self.rho0 =  np.array([0.001], dtype="float64")

        if (self.gamma == None ):
            self.gamma = np.array([5.0/3.0], dtype="float64")         

        if (self.aspect_ratio == None ):
            self.aspect_ratio = np.array([0.05], dtype="float64")

        if (self.inner_radius == None ):
            self.inner_radius = np.array([0.0], dtype="float64")

        if (self.outer_radius == None ):
            self.outer_radius = np.array([1.0], dtype="float64")

        if (self.truncation == None ):
            self.truncation = False

        if (self.R_0 == None ):
            self.R_0 =  self.inner_radius

        if (self.central_mass == None ):
            self.central_mass = np.array([1.0], dtype="float64")

        if (self.star_softening == None):
            self.star_softening =  np.array([0.0], dtype="float64")
            
        if (self.gravityG == None ):
            self.gravityG = np.array([1.0], dtype="float64")

        if (self.profile_type == None ):
            self.profile_type = "powerlaw"

        if (self.ndims == None ):
            self.ndims = np.array([2], dtype="uint32")

        if (self.mesh_type == None):
            self.mesh_type = "polar"

        if (self.mesh_alignment == None):
            self.mesh_alignment = "aligned" #options: interleaved, random

        if (self.double == None):
            self.double = np.array([0], dtype="uint32")

        self.cs_0 = self.aspect_ratio * math.sqrt(self.gravityG * self.central_mass / self.R_0)

        if (self.double == 0):
            self.dt = 'float32'
        if (self.double == 1):
            self.dt = 'float64'
                
#################################################################################################
def get_disk_grid_2D(disk,N1, N2,BoxSizeX,BoxSizeY,**kwargs):

    if (disk.double == 0):
        dt="float32"
    if (disk.double == 1):
        dt="float64"

    mesh_option = kwargs.get("from_file")

    Ntot = N1 * N2
    if(disk.mesh_type == "ring"):
      Ntot = 0
      delta = (disk.outer_radius - disk.inner_radius)/N1
      R = np.linspace(disk.inner_radius + 0.5 * delta,disk.outer_radius,N1) 
      for i in range(0,N1):
	Ntot+= int(2*np.pi*R[i]/delta)
    
    pos=np.zeros([Ntot, 3], dtype=dt)

    if (mesh_option == None):
        if (disk.mesh_type == "cartesian"):
            offsetX=BoxSizeX/2.0-disk.outer_radius
            offsetY=BoxSizeY/2.0-disk.outer_radius
            deltax=float(2.0 * disk.outer_radius / N1)
            deltay=float(2.0 * disk.outer_radius / N2)
            for i in range(0,N1):
                for j in range(0,N2):
                    pos[i+j*N1,0]=float(i*deltax+deltax/2.0)+offsetX
                    pos[i+j*N1,1]=float(j*deltay+deltay/2.0)+offsetY
                    pos[i+j*N1,2]=0.0
        
        if (disk.mesh_type == "polar"):
            offsetX=BoxSizeX/2.0
            offsetY=BoxSizeY/2.0
            deltalog = (math.log(disk.outer_radius)-math.log(disk.inner_radius))/ N1
            logR = (np.arange(0,N1, dtype=dt)+0.5)/N1 * (math.log(disk.outer_radius)-math.log(disk.inner_radius))+ \
                   math.log(disk.inner_radius)
            
            ip = 0
            for i in range(0,N1):
                R = math.exp(logR[i])
                for j in range(0,N2):
                    phi=float((j+0.5)*2.0*math.pi/N2)
                    if (disk.mesh_alignment == "interleaved"):
                        phi+=(-1)**i*0.25*2.0*math.pi/N2
                    pos[ip,0]= R * math.cos(phi)+offsetX
                    pos[ip,1]= R * math.sin(phi)+offsetY
                    pos[ip,2]=0.0
                    ip+=1
            
        if (disk.mesh_type == "ring"):
            offsetX=BoxSizeX/2.0
            offsetY=BoxSizeY/2.0
            delta = (disk.outer_radius - disk.inner_radius)/N1
            R = np.linspace(disk.inner_radius + 0.5 * delta,disk.outer_radius,N1) 
            ip = 0
            for i in range(0,N1): 
                for j in range(0,int(2*np.pi*R[i]/delta)):
                    phi=float((j+0.5)*2.0*math.pi/int(2*np.pi*R[i]/delta))
                    if (disk.mesh_alignment == "interleaved"):
                        phi+=(-1)**i*0.25*2.0*math.pi/int(2*np.pi*R[i]/delta)
                    pos[ip,0]= R[i] * math.cos(phi)+offsetX
                    pos[ip,2]=0.0
                    pos[ip,1]= R[i] * math.sin(phi)+offsetY
                    ip+=1

        if (disk.mesh_type == "unstructured"):
            offsetX=BoxSizeX/2.0
            offsetY=BoxSizeY/2.0
            ip = 0
            for i in range(0,N1):
                R = float((i+0.5)*(disk.outer_radius-disk.inner_radius)/N1) + disk.inner_radius
                delta_R = float((i+0.5)*(disk.outer_radius-disk.inner_radius)/N1) + disk.inner_radius - R
                for j in range(0,N2):
                    phi=float((j+0.5)*2.0*math.pi/N2)
                    pos[ip,0]= R * math.cos(phi)+offsetX
                    pos[ip,1]= R * math.sin(phi)+offsetY
                    pos[ip,2]=0.0
                    ip+=1     

    else:
        import readsnapHDF5 as rs
        pos = rs.read_block(mesh_option,"POS ",parttype=0)

        
    pos[:,0]=np.where( (np.sqrt((pos[:,0]-BoxSizeX/2.0)**2+(pos[:,1]-BoxSizeY/2.0)**2) < disk.outer_radius) &
                       (np.sqrt((pos[:,0]-BoxSizeX/2.0)**2+(pos[:,1]-BoxSizeY/2.0)**2) > disk.inner_radius), pos[:,0],-1)
   

    ind=np.where(pos[:,0] >= 0)[0]
    newpos=np.zeros([len(ind), 3], dtype=dt)
    newpos[:,0]=pos[ind,0]
    newpos[:,1]=pos[ind,1]
    newpos[:,2]=pos[ind,2]
 
    ids=np.arange(1,len(ind)+1, dtype="int32")

    return newpos,ids

#################################################################################################
def get_disk_grid_3D(disk,NR, Nphi, Box ,**kwargs):

    if (disk.double == 0):
        dt="float32"
    if (disk.double == 1):
        dt="float64"

    mesh_option = kwargs.get("from_file")

    #Ntheta is obtained from the values of NR and Nphi
    
    R = sample_radial_positions(disk, NR)
    phi = sample_azimuthal_angles(disk,Nphi)

    Ntheta1 = disk.aspect_ratio/2.0/math.sqrt(2.0*math.pi) * Nphi
    Ntheta0 = 32
    theta = sample_polar_angles(disk,Ntheta0,Ntheta1)
    Ntheta = theta.shape[0]
    
    pos=np.zeros([NR*Nphi*Ntheta, 3], dtype=dt)

    if (mesh_option == None):
        offsetX=Box.X/2.0
        offsetY=Box.Y/2.0
        offsetZ=Box.Z/2.0
        ip = 0
        for i in range(0,NR):
            for j in range(0,Nphi):
                for k in range(0,Ntheta):
                    pos[ip,0]= R[i] * math.cos(phi) *  math.sin(theta) +offsetX
                    pos[ip,1]= R[i] * math.sin(phi)*  math.sin(theta) +offsetY
                    pos[ip,2]= R[i] * math.cos(theta)+ offsetZ
                    ip+=1

    else:
        import readsnapHDF5 as rs
        pos = rs.read_block(mesh_option,"POS ",parttype=0)

        
    pos[:,0]=np.where( (np.sqrt((pos[:,0]-BoxSizeX/2.0)**2+(pos[:,1]-BoxSizeY/2.0)**2) < disk.outer_radius) &
                       (np.sqrt((pos[:,0]-BoxSizeX/2.0)**2+(pos[:,1]-BoxSizeY/2.0)**2) > disk.inner_radius), pos[:,0],-1)
   

    ind=np.where(pos[:,0] >= 0)[0]
    newpos=np.zeros([len(ind), 3], dtype=dt)
    newpos[:,0]=pos[ind,0]
    newpos[:,1]=pos[ind,1]
    newpos[:,2]=pos[ind,2]
 
    ids=np.arange(1,len(ind)+1, dtype="int32")

    return newpos,ids

#################################################################################################
def get_disk_grid_unstructured_3D(disk,Ngas,BoxSizeX,BoxSizeY,BoxSizeZ,**kwargs):

    if (disk.double == 0):
        dt="float32"
    if (disk.double == 1):
        dt="float64"

  #  Rsize=512
  #  Zsize=512
  #  RhoGas=np.zeros([Rsize,Zsize], dtype=dt)

#    radialbins=((2.0-p)*(np.arange(0,Rsize-1, dtype=dt)+0.5)*((disk.outer_radius)**(2.0-disk.p) 
 #                                                                     /(2.0-disk.p))/Nsize)**(1.0/(2.0 - disk.p))
 #   zbins =spec.erfinv(np.arange(1,Zsize,dtype=dt)/Zsize*2-1)*np.sqrt(2.0)

    
  #  for i in range(0,Rsize):
  #      for j in range(0,Zsize):
  #          RhoGas[i][j] = get_surface_density(R,disk) * math.exp(-zbins[j]**2/2.0)


    offsetX=BoxSizeX/2.0
    offsetY=BoxSizeY/2.0
    offsetZ=BoxSizeZ/2.0

    pos=np.zeros([Ngas,3], dtype=dt)
    for i in range(0,Ngas):
        R = sample_radial_position(disk)
        z = sample_vertical_position(R,disk)
        phi = ran.rand()*2.0*math.pi

        if(ran.rand() > 0.5):
            z = -z;

        pos[i,0] = R* math.cos(phi)+offsetX
        pos[i,1] = R* math.sin(phi)+offsetY
        pos[i,2] = z +offsetZ
        
        
    ids=np.arange(1,Ngas+1, dtype="int32")
  
    return pos,ids
###################################
#def add_background_grid_3D(pos,ids,disk,Ncells,BoxSizeX,BoxSizeY,BoxSizeZ):
#    
#    
#    
#    
#
#
#    return pos,ids
#
###################################
def sample_polar_angles(disk,N0,N1):


    costheta0=np.arange(-1.0+1.0/N0/2.0,1.0,1.0/N0)

    arr1=np.arange(1.0/N1/2.0,1.0,1.0/N1)
    costheta1=spec.erfinv(2.0*arr1-1.0)*np.sqrt(2.0)*disk.aspect_ratio

    delta = disk.aspect_ratio * math.sqrt(math.log(math.sqrt(math.pi/2.0) * N1 / N0/ disk.aspect_ratio))


    costheta0[:]=np.where((costheta0[:] < -delta) | (costheta0[:] > math.pi/2.0+delta), costheta0[:],-1e30)
    costheta1[:]=np.where((costheta1[:] > -delta) & (costheta1[:] < math.pi/2.0+delta), costheta1[:],-1e30)


    costheta=np.append(costheta0,costheta1)[np.where(np.append(costheta0,costheta1) >= -1)[0]]
    costheta=costheta.sort()

    theta = math.acos(costheta)
                 
    return theta

###################################
def sample_radial_position(disk, NR):

    logR = (np.arange(0,NR, dtype=disk.dt)+0.5)/NR * (math.log(disk.outer_radius)-math.log(disk.inner_radius))+ \
               math.log(disk.inner_radius)
    R = math.exp(logR)
    
    return R
###################################
def sample_vertical_position(R, disk):
    
    q = 1
    while (q > 0.995):
        q = ran.rand()
        
    z = spec.erfinv(2.0*q - 1.0) * np.sqrt(2.0) * disk.aspect_ratio * R
     
    
    return z


###################################
#ADD DISK BOUNDARIES

def add_disk_boundaries_in_2D(pos,ids,R,Ntheta,BoxSizeX,BoxSizeY,zval = 0,dt='float32'):
     
    thickness = float(2.0*math.pi*R/Ntheta)
    pos,ids = add_ring_of_cells(pos=pos,ids=ids,R=R+thickness/2.0,Ntheta=Ntheta,
                                BoxSizeX=BoxSizeX,BoxSizeY=BoxSizeY,zval = zval,equal_ids=-1, dt=dt)
    
    pos,ids = add_ring_of_cells(pos=pos,ids=ids,R=R-thickness/2.0,Ntheta=Ntheta,
                                BoxSizeX=BoxSizeX,BoxSizeY=BoxSizeY,zval = zval,equal_ids=-2,dt=dt)


    return pos, ids

def add_disk_boundaries_out_2D(pos,ids,R,Ntheta,BoxSizeX,BoxSizeY,zval = 0,dt='float32'):

    thickness = float(2.0*math.pi*R/Ntheta)
    pos,ids = add_ring_of_cells(pos=pos,ids=ids,R=R+thickness/2.0,Ntheta=Ntheta,
                                BoxSizeX=BoxSizeX,BoxSizeY=BoxSizeY,zval = zval,equal_ids=-2, dt=dt)
    
    pos,ids = add_ring_of_cells(pos=pos,ids=ids,R=R-thickness/2.0,Ntheta=Ntheta,
                                BoxSizeX=BoxSizeX,BoxSizeY=BoxSizeY,zval = zval,equal_ids=-1,dt=dt)


    return pos, ids
###################################
#ADD  A RING OF CELLS

def add_ring_of_cells(pos,ids,R,Ntheta,BoxSizeX,BoxSizeY,equal_ids=0,dt='float32',**kwargs):

    noise = kwargs.get("noise")
    zval = kwargs.get("zval")
    if (zval == None):
        zval = np.array([0], dtype="float64")
    if (noise == None):
        noise = np.array([0], dtype="float64")      

    Ncells=Ntheta
    thickness = float(2.0*math.pi*R/Ntheta)
    ring=np.zeros([Ncells, 3], dtype=dt)
    for i in range(0,Ncells):
        phi=float((i+0.5)*2.0*math.pi/Ncells)
        ring[i,0] = R * math.cos(phi)+BoxSizeX/2.0 + ran.gauss(0,thickness * noise)
        ring[i,1] = R * math.sin(phi)+BoxSizeY/2.0 + ran.gauss(0,thickness * noise)
        ring[i,2] = zval

 
    pos[:,0]=np.where( (np.sqrt(  (pos[:,0]-BoxSizeX/2.0)**2 + (pos[:,1]-BoxSizeY/2.0)**2) > thickness/1.99+R) |
                       (np.sqrt(  (pos[:,0]-BoxSizeX/2.0)**2 + (pos[:,1]-BoxSizeY/2.0)**2) < -thickness/1.99+R) , pos[:,0],-1)
    ind=np.where(pos[:,0] >= 0)[0]
    newpos=np.zeros([len(ind), 3], dtype=dt)
       
    newpos[:,0]=pos[ind,0]
    newpos[:,1]=pos[ind,1]
    newpos[:,2]=pos[ind,2]

    newids=ids[ind]

    

    if (equal_ids == 0):
        ids=np.append(newids,np.arange(np.max(newids)+1,np.max(newids)+1+Ncells, dtype="int32"))
    else:
        ids=np.append(newids,np.zeros(Ncells, dtype="int32")+equal_ids)
        
    pos=np.concatenate((newpos,ring),axis=0)
    
    return pos, ids

###################################
#ADD_BACKGROUND_GRID
    
def add_background_grid_2D(disk, pos,ids,BoxSizeX, BoxSizeY,equal_ids=0,**kwargs):

    mesh_option = kwargs.get("type")
    Ncells = kwargs.get("Ncells")
    noise = kwargs.get("noise")

    radius = np.sqrt((pos[:,0]-BoxSizeX*0.5)**2 + (pos[:,1]-BoxSizeY*0.5)**2)
    max_radius = max(radius.max(),disk.outer_radius)
    
    if (mesh_option == None):
        #open 2-D mesh information from library or current directory

        Npart, BoxXBack, BoxYBack, pos_back = read_background_grid()
    
    else:
        Npart = Ncells*Ncells
        back=np.zeros([Ncells*Ncells, 3])
        deltax=float(BoxSizeX/ Ncells)
        deltay=float(BoxSizeY/ Ncells)
        for i in range(0,Ncells):
            for j in range(0,Ncells):
                back[i+j*Ncells,0]=float(i*deltax+deltax/2.0)+ ran.gauss(0,deltax * noise) 
                back[i+j*Ncells,1]=float(j*deltay+deltay/2.0)+ ran.gauss(0,deltay * noise) 
                back[i+j*Ncells,2]=0.0
        back[:,0]=np.where( (np.sqrt((back[:,0]-BoxSizeX/2.0)**2+(back[:,1]-BoxSizeY/2.0)**2) > max_radius), back[:,0],-1)
        ind=np.where(back[:,0] >= 0)[0]
        pos_back=np.zeros([len(ind), 3])
        pos_back[:,0]=back[ind,0] 
        pos_back[:,1]=back[ind,1] 
        pos_back[:,2]=back[ind,2]
        
    pos=np.concatenate((pos,pos_back),axis=0)
   
    if (equal_ids == 0):
        ids=np.append(ids,np.arange(np.max(ids)+1,np.max(ids)+1+len(ind), dtype="int32"))
    else:
        ids=np.append(ids,np.repeat(equal_ids,len(ind)))


    return pos, ids
    
###################################
#READ BACK_GROUND_GRID

def read_background_grid():
    #read in background particle positions
    if (os.path.exists("voronoi_background_mesh")):
        filename="voronoi_background_mesh"
    else:
        filename="~dmunoz/lib/python/OwnModules/voronoi_background_mesh"

    Npart=0
    f = open(filename, 'rb')
    blocksize = np.fromfile(f,dtype=np.uint32,count=1)
    Npart = np.fromfile(f,dtype=np.uint32,count=1)[0]
    BoxSizeX = (np.fromfile(f,dtype=np.float32,count=1))[0]
    BoxSizeY = (np.fromfile(f,dtype=np.float32,count=1))[0]
    pos_back = (np.fromfile(f,dtype=np.dtype((np.float64,3)),count=Npart))
    f.close()

    pos = np.zeros([Npart, 3], dtype='float64')
    pos[:,0] = pos_back[:,1]
    pos[:,1] = pos_back[:,2]
    pos[:,2] = pos_back[:,0]
    
    return Npart,BoxSizeX, BoxSizeY, pos


#########################        
#Physical quantities at the mesh-generating points

def assign_quantities(disk,pos,BoxSizeX,BoxSizeY):
    
       
    Npart = pos.shape[0]
    vel=np.zeros([Npart, 3], dtype=disk.dt)
    dens=np.zeros(Npart, dtype=disk.dt)
    u=np.zeros(Npart, dtype=disk.dt)


    #if the disk is flared (i.e. l != 1 and the aspect ratio varies with R). In that case
    #the input parameter 'aspect_ratio' corresponds to the aspect ratio at the reference radius R_0.
    #At R_0 the disk should approximately be in Keplerian rotation.
    

    for i in range(0,Npart):
        R2 = (pos[i,0]-BoxSizeX/2.0)**2+(pos[i,1]-BoxSizeY/2.0)**2
        R = math.sqrt(R2)
        phi = math.atan2((pos[i,1]-BoxSizeY/2.0),(pos[i,0]-BoxSizeX/2.0))

        if (R < 1.1*disk.outer_radius):
            
            #orbital velocity
            vphi=get_orbital_velocity(R,disk)
            #density
            rho = get_surface_density(R,disk)
            #sound speed
            cs =  get_sound_speed(R, disk)
            if (disk.eos == 'isothermal'):
                press = get_pressure(rho, cs, 1.0)
            if (disk.eos == 'adiabatic'):
                press = get_pressure(rho, cs, disk.gamma)
            utherm=get_internal_energy(rho,press,disk.gamma)    

        else:
            #orbital velocity
            vphi= 0.0
            #density
            rho = get_surface_density(disk.outer_radius,disk)
            #sound speed
            cs =  get_sound_speed(disk.outer_radius, disk)
            if (disk.eos == 'isothermal'):
                press = get_pressure(rho, cs, 1.0)
            if (disk.eos == 'adiabatic'):
                press = get_pressure(rho, cs, disk.gamma)
            utherm=get_internal_energy(rho,press,disk.gamma)   

                    
        #save the velocity
        vel[i,0] = -vphi * math.sin(phi)
        vel[i,1] = vphi * math.cos(phi)
        #save the density
        dens[i] = rho
        #save the energy density
        u[i] = utherm
       
                    
            
    return vel,dens,u

#########################
#surface density field
def get_surface_density(R, disk):
    
    if (disk.truncation == False):
        sigma = disk.rho0 * (disk.R_0 / R)**disk.p
        sigma = disk.rho0 * (disk.R_0 /np.sqrt(disk.star_softening**2 + R**2))**disk.p
    else:
        sigma = 1e-20 + disk.rho0 * (disk.R_0 / R)**disk.p / (math.exp((R/disk.outer_radius - 1.0)/disk.aspect_ratio**2) + 1.0)
    
    return sigma
#########################
# density field
def get_density(R,z,disk):

    if (disk.ndims == 2):
        rho = get_surface_density(R,disk)

    if (disk.ndims == 3):   
        rho = disk.rho0 * (disk.R_0 /np.sqrt(disk.star_softening**2 + R**2))**disk.p *\
              math.exp(-z**2/2/(disk.aspect_ratio *R)**2)

    return rho

#########################
#scale height of the disk
def get_scale_heigth(R, disk):
    
    
    return H


#########################
#orbital velocity of gas
def get_orbital_velocity(R,disk):
    R2 = R * R 
    v_sq=0

    if (disk.potential == 'keplerian'):
        v_sq = kepler_omega(R,disk)**2 * R2
    if (disk.potential == 'selfgravity'):
        v_sq = (kepler_omega(R,disk)**2 + self_gravity_omega(R, disk)**2) * R2    

    v_sq = v_sq + get_pressure_buffer(R,disk)
    v = np.sqrt(v_sq)

#    if (R < 0.4):
#        print v
    return v

#########################
#gas pressure
def get_pressure(rho, cs, gamma):
    return cs * cs * rho / gamma

#########################
#thermal energy
def get_internal_energy(rho,press,gamma):
    u = press/rho/(gamma - 1)

    return u

##########################
#SOUND SPEED IN A DISK
def get_sound_speed(R, disk):
    R2 = R * R 
    R_1 = 1.0/np.sqrt( R2 + disk.star_softening * disk.star_softening)
    return disk.cs_0 * np.sqrt(disk.R_0 * R_1)**disk.l


#########################
#The pressure effect in the orbital velocity of gas

def get_pressure_buffer(R,disk):

    R2 = R * R 
    R_1 = 1.0/np.sqrt( R2 + disk.star_softening * disk.star_softening)

    if (disk.eos == "isothermal"):        
        Gamma = 1.0
    elif (disk.eos == "adiabatic"):
        Gamma = disk.gamma
        
    if (disk.truncation == False):
        delta_v_sq = -disk.p - disk.l / Gamma
    else:
        delta_v_sq = - disk.p - disk.l / Gamma \
                     - math.exp((R/disk.outer_radius - 1.0)/disk.aspect_ratio**2)/ disk.aspect_ratio**2 \
                     * 1.0/(math.exp((R/disk.outer_radius - 1.0)/disk.aspect_ratio**2) + 1.0) 
        
        
    delta_v_sq = delta_v_sq * disk.cs_0**2 * (R / disk.R_0)**(- disk.l )
    

        
#    if (R < 0.4):
#        print delta_v_sq, R, disk.gamma
    return delta_v_sq


########################
#########################
#Some rotation curve laws

def kepler_omega(R, disk):
    R2 = R * R 
    R_1 = 1.0/np.sqrt( R2 + disk.star_softening * disk.star_softening)
    R_2 = 1.0/(R2 + disk.star_softening * disk.star_softening)
    R_3 = R_1 * R_2

    omega = np.sqrt(disk.gravityG * disk.central_mass * R_3)

    return omega




def self_gravity_omega(R,disk):
    R2 = R * R 
    
    omega = math.pi * np.sqrt(math.pi) * disk.gravityG * 2.0**(disk.p/4.0) * disk.rho0 * R2 / disk.R_0 * spec.gamma(disk.p/2.0 + 0.5)/ spec.gamma(disk.p/2.0) \
            * spec.hyp2f1(disk.p/2.0 + 0.5, 1.5, 2.0, - R2 / disk.R_0 / disk.R_0)
    
    return omega

#########################
#def add_particles(pos,vel,dens,u,ids, Npart,masses,disk,BoxSizeX,BoxSizeY):
def add_particles(ids, Npart,masses,disk,BoxSizeX,BoxSizeY):

    #get a random sample of semi-major axis following the same
    #power-law profile of the gas disk
    R = (uniform.rvs(size=Npart) *(disk.outer_radius**(1-disk.p)-disk.inner_radius**(1-disk.p))+disk.inner_radius**(1-disk.p))**(1/(1-disk.p))
    
    #and we get a random sample of the orbital phase
    phi = uniform.rvs(size=Npart)*2.0*math.pi

    pos=np.zeros([Npart, 3], dtype=disk.dt)
    vel=np.zeros([Npart, 3], dtype=disk.dt)

    pos[:,0],pos[:,1],pos[:,2] = R * np.cos(phi) + BoxSizeX/2.0, R * np.sin(phi) + BoxSizeY/2.0, 0.0
    vel[:,0],vel[:,1],vel[:,2] = -kepler_omega(R,disk) * R * np.sin(phi), kepler_omega(R,disk) * R * np.cos(phi),0.0

    
    #pos=np.concatenate((pos,pos_part),axis=0)
    #vel=np.concatenate((vel,vel_part),axis=0)
    #dens=np.append(dens,np.zeros(Npart, dtype=disk.dt)+masses)
    #u=np.append(u,np.zeros(Npart, dtype=disk.dt))
    #ids=np.append(ids,np.arange(np.max(ids)+1,np.max(ids)+1+Npart, dtype="int32"))
    ids=np.arange(np.max(ids)+1,np.max(ids)+1+Npart, dtype="int32")

    return pos,vel,ids
