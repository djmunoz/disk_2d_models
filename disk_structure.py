import numpy as np


def powerlaw_sigma(R,sigma0,p,R0):
    return sigma0*(R/R0)**(-p)

def similarity_sigma(R,sigma0,gamma,Rc):
    return sigma0*(R/Rc)**(-gamma) * np.exp(-(R/Rc)**(2.0-gamma))

def soundspeed(R,csnd0,l,R0):
    return csnd0 * (R/R0)**(-l*0.5)


class powerlaw_disk(object):
    def __init__(self, *args, **kwargs):
        print kwargs
        self.sigma0 = kwargs.get("sigma0")
        self.p = kwargs.get("p")
        self.R0 = kwargs.get("R0")
        print kwargs.get("p")

        #set default values
        if (self.sigma0 == None):
            self.sigma0 = 1.0
        if (self.p == None):
            self.p = 1.0
        if (self.R0 == None):
            self.R0 = 1.0

    def evaluate(self,R):
        return powerlaw_sigma(R,self.sigma0,self.p,self.R0)

class similarity_disk(object):
    def __init__(self, *args, **kwargs):
        self.sigma0 = kwargs.get("sigma0")
        self.gamma = kwargs.get("gamma")
        self.Rc = kwargs.get("Rc")

        #set default values
        if (self.sigma0 == None):
            self.sigma0 = 1.0
        if (self.gamma == None):
            self.gamma = 1.0
        if (self.Rc == None):
            self.Rc = 1.0

    def evaluate(self,R):
        return similarity_sigma(R,self.sigma0,self.gamma,self.Rc)

class disk(object):
    def __init__(self, *args, **kwargs):
        #define the properties of the axi-symmetric disk model
        self.sigma_type = kwargs.get("sigma_type")

        #Temperature profile properties
        self.csndR0 = kwargs.get("csndR0") #reference radius
        self.csnd0 = kwargs.get("csnd0") # soundspeed scaling
        self.l = kwargs.get("l") # temperature profile index



        #set defaults 
        if (self.sigma_type == None):
            self.sigma_type="powerlaw"
        if (self.l == None):
            self.l = 1.0
        if (self.csnd0 == None):
            self.csnd0 = 0.05
            

        if (self.sigma_type == "powerlaw"):
            print kwargs
            self.sigma_disk = powerlaw_disk(**kwargs)
            if (self.csndR0 == None):
                self.csndR0 = self.sigma_disk.R0

        if (self.sigma_type == "similarity"):
            self.sigma_disk = similarity_disk(**kwargs)
            if (self.csndR0 == None):
                self.csndR0 = self.sigma_disk.Rc



    def evaluate_sigma(self,Rin,Rout,Nvals=1000,scale='log'):
        if (scale == 'log'):
            rvals = np.logspace(np.log10(Rin),np.log10(Rout),Nvals)
        elif (scale == 'linear'):
            rvals = np.linspace(Rin,Rout,Nvals)
        else: 
            print "[error] scale type ", quant, "not known!"
            sys.exit() 
        return self.sigma_disk.evaluate(rvals)
    
    def evaluate_soundspeed(self,Rin,Rout,Nvals=1000,scale='log'):
        if (scale == 'log'):
            rvals = np.logspace(np.log10(Rin),np.log10(Rout),Nvals)
        elif (scale == 'linear'):
            rvals = np.linspace(Rin,Rout,Nvals)
        else: 
            print "[error] scale type ", quant, "not known!"
            sys.exit()
        return soundspeed(rvals,self.csnd0,self.l,self.csndR0)

    def evaluate_pressure(self,Rin,Rout,Nvals=1000,scale='log'):
        if (scale == 'log'):
            rvals = np.logspace(np.log10(Rin),np.log10(Rout),Nvals)
        elif (scale == 'linear'):
            rvals = np.linspace(Rin,Rout,Nvals)
        else: 
            print "[error] scale type ", quant, "not known!"
            sys.exit()
        return self.evaluate_sigma(Rin,Rout,Nvals,scale=scale) * \
            self.evaluate_soundspeed(Rin,Rout,Nvals,scale=scale)**2

    def evaluate_pressure_gradient(self,Rin,Rout,Nvals=1000,scale='log'):
        if (scale == 'log'):
            rvals = np.logspace(np.log10(Rin),np.log10(Rout),Nvals)
        elif (scale == 'linear'):
            rvals = np.linspace(Rin,Rout,Nvals)
        else: 
            print "[error] scale type ", quant, "not known!"
            sys.exit()

        press = self.evaluate_pressure(Rin,Rout,Nvals,scale=scale)
        if (scale == 'log'):
            dPdlogR = np.gradient(press)/np.gradient(np.log10(rvals))
            print np.diff(np.log10(rvals))
            dPdR = dPdlogR/rvals/np.log(10)
        elif (scale == 'linear'):
            dPdR = np.gradient(press)/np.gradient(rvals)
        return dPdR


#class disk_mesh(disk):
    
