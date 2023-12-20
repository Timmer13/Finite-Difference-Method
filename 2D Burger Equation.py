import numpy as np
import math
from matplotlib import pyplot, cm           #plotting library that we will use to plot our results
from mpl_toolkits.mplot3d import Axes3D     #To plot a projected 3D result, make sure that you have added the Axes3D library
from sklearn.linear_model import LinearRegression

lineSingle = '------------------------------------------------'

print("Solving Burger's equation using Finite Difference Method\n")

#meshing

switch = input("Enter 1 to hold mesh size or 2 to hold dt: ")
switch1 = input("Enter 1,2,3 for 3 examples: ")
if switch == str(1):
    mbig = [30]
    nt = list(range(400,800,100))
elif switch == str(2):
    mbig = list(range(10,50,10))
    nt = [500] 
    


    
e = []
rel_e = []




        
#Actual Solution
def uu(x,y,t):
    u1 = math.exp(-t)*math.sin(math.pi*y)*math.sin(math.pi*x)
    if switch1 == str(1) : uu=u1
    return uu

#Initial Solution
def u0(x,y):
    u1 = math.sin(math.pi*y)*math.sin(math.pi*x)
    if switch1 == str(1) : u0=u1
    return u0

#RHS of Dirichlet problem

def f(x,y,t):
    # the RHS f
    f1 = -math.exp(-t)*u0(x,y)+2*math.pi**2*math.exp(-t)*u0(x,y)+math.pi*math.exp(-t)*math.sin(math.pi*(x+y))*math.exp(-t)*u0(x,y)
    if switch1 == str(1) : f=f1
    return f
  
print("N   Iteration Error     Rel_error")
for m in mbig:
  tn = .5
  nu = .1
  
  nx = m            #Grid Points along X direction
  ny = m           #Grid Points along Y direction
  xmin = 0
  xmax = 1
  ymin = 0
  ymax = 1
  dx = (xmax - xmin) / (nx - 1)         
  dy = (ymax - ymin) / (ny - 1)
  un = np.zeros((ny,nx))            
  x = np.linspace(xmin,xmax,nx)
  y = np.linspace(ymin,ymax,ny)


  dx = (1) / (nx - 1)         
  dy = (1) / (ny - 1)
  

         
# Actual Solution at final time            
  actual = np.zeros((nx,ny))
  for i in range (nx):
      for j in range (ny):
          actual[i,j]=uu(i*dx,j*dy,tn)


  for n in nt:
# Initial Condition  
      u = np.zeros((ny,nx)) 
      for i in range (nx):
          for j in range (ny):
              u[i,j]=u0(i*dx,j*dy)
      for it in range(n):
          dt = tn / n
          un = u.copy()
          for i in range(1,nx-1):
              for j in range(1,ny-1):
                  u[i,j] = (un[i, j] -(un[i, j] * dt / (2*dx) * (un[i+1, j] - un[i-1, j])) -un[i, j] * dt / (2*dy) * (un[i, j+1] - un[i, j-1])) + (nu*dt/(dx**2))*(un[i+1,j]-2*un[i,j]+un[i-1,j])+(nu*dt/(dx**2))*(un[i,j-1]-2*un[i,j]+un[i,j+1])+f(i*dx,j*dy,it)*dt

          u[:,0]=0
          u[:,-1]=0
          u[0,:]=0
          u[-1,:]=0



        
#Compute l2 error
      error = 0
      denom = 0
      for i in range(1,nx-1):
          for j in range (ny):
              error += pow((actual[i,j]-u[i,j]),2)
              denom +=  pow(actual[i,j],2)
      error = math.sqrt(error)
      rel_error = error/denom
      print("%.4f   %.4f   %.6f   %.6f" % (m,n,error,rel_error))
      e.append(error)
      rel_e.append(rel_error)

if switch == str(1):
    pyplot.plot(nt,rel_e)
    pyplot.title('Plot for relative L2 error with different iterations w/ mesh points = ' + str(mbig[0]))
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Relative l2error')
    pyplot.show()
    pyplot.loglog(nt,rel_e)
    pyplot.title('Log-log plot for relative L2 error with different iterations w/ mesh points = ' + str(mbig[0]))
    pyplot.xlabel('Log number of iterations')
    pyplot.ylabel('Log relative l2error')
    pyplot.show()
    i = len(rel_e)
    x = np.array(np.log(nt[0:i])).reshape((-1,1))
    y = np.array(np.log(rel_e[0:i]))
    model = LinearRegression().fit(x, y)
    print('coefficient of determination:', model.score(x,y))
    print('slope:', model.coef_)
elif switch == str(2):
    pyplot.plot(mbig,rel_e)
    pyplot.title('Plot for relative L2 error with different number of mesh points w/ iterations = ' + str(nt[0]))
    pyplot.xlabel('Number of mesh points in x')
    pyplot.ylabel('Relative l2error')
    pyplot.show()
    pyplot.loglog(mbig,rel_e)
    pyplot.title('Log-log plot for relative L2 error with different number of mesh points w/ iterations = ' + str(nt[0]))
    pyplot.xlabel('Log number of mesh points in x')
    pyplot.ylabel('Log relative l2error')
    pyplot.show()
    i = len(rel_e)
    x = np.array(np.log(mbig[0:i])).reshape((-1,1))
    y = np.array(np.log(rel_e[0:i]))
    model = LinearRegression().fit(x, y)
    print('coefficient of determination:', model.score(x,y))
    print('slope:', model.coef_)
    

    

