import disk_structure
import numpy as np
import matplotlib.pyplot as plt

d = disk_structure.disk(p=1,R0=3.0)
print d.sigma_type

#print d.evaluate_sigma(1.0,10.0)

Rin,Rout = 1.0, 10.0
Nvals = 100
scale='log'
scale='linear'
rvals,dpdR= d.evaluate_pressure_gradient(Rin,Rout,Nvals=Nvals,scale=scale)



plt.plot(rvals,dpdR,'bo',ls='-',ms=2.0)
#plt.plot(rvals,p,'bo',ls='-',ms=2.0)
plt.savefig("test.png")
plt.show()

mesh=disk_structure.disk_mesh(d,NR=180,Nphi=120,mesh_alignment="interleaved",fill_box=True,N_inner_boundary_rings=2)
mesh.create()

ss = disk_structure.snapshot()
ss.create(d,mesh)
ss.write_snapshot(d,mesh)



print ss.ids[ss.ids < 0].shape

print ss.pos.shape, ss.ids.shape

plt.plot(ss.pos[:,0],ss.pos[:,1],'ko',ms=1.0)
plt.plot(ss.pos[ss.ids==-1,0],ss.pos[ss.ids==-1,1],'ro',ms=2.0,mew=0)
plt.plot(ss.pos[ss.ids==-2,0],ss.pos[ss.ids==-2,1],'bo',ms=2.0,mew=0)
plt.plot(ss.pos[ss.ids==-3,0],ss.pos[ss.ids==-3,1],'go',ms=2.0,mew=0)
plt.show()



