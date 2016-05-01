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
#rvals = np.logspace(np.log10(Rin),np.log10(Rout),Nvals)
rvals = np.linspace(Rin,Rout,Nvals)
p=d.evaluate_pressure(Rin,Rout,Nvals=Nvals,scale=scale)
dpdR= d.evaluate_pressure_gradient(Rin,Rout,Nvals=Nvals,scale=scale)

plt.plot(rvals,dpdR,'bo',ls='-',ms=2.0)
#plt.plot(rvals,p,'bo',ls='-',ms=2.0)
plt.savefig("test.png")
plt.show()

