import numpy as np                         #give mathamatical or matrix expressions like array
import math
from matplotlib import pyplot, cm           #plotting library that we will use to plot our results
from mpl_toolkits.mplot3d import Axes3D     #To plot a projected 3D result, make sure that you have added the Axes3D library
from sklearn.linear_model import LinearRegression

lineSingle = '------------------------------------------------'

print("Solving Dirichlet Problem using Finite Difference Method\n")

#meshing

switch = input("Enter 1 to hold mesh size or 2 to hold iteration: ")
switch1 = input("Enter 1,2,3 for 3 examples: ")
if switch == str(1):
    mbig = [80]
    iteration = list(range(1000,15000,1000))
elif switch == str(2):
    mbig = list(range(10,200,10))
    iteration = [10000] 
    


    
e = []
rel_e = []

#Actual Solution
def g(x,y):
    g1 = math.sin(math.pi*x)*math.cos(math.pi*y)
    g2 = y*math.sin(math.pi*x)*pow(math.sin(math.pi*y),2)
    g3 = pow(math.cos(math.pi*x),2)*math.cos(math.pi*y)
    if switch1 == str(1) : g=g1
    if switch1 == str(2) : g=g2
    if switch1 == str(3) : g=g3
    return g

#RHS of Dirichlet problem

def f(x,y):
    # the RHS f
    f1 = (2*pow(math.pi,2)+1)*math.sin(math.pi*x)*math.cos(math.pi*y)
    f2 = math.pi*math.sin(math.pi*x)*(3*math.pi*y*pow(math.sin(math.pi*y),2)-4*math.cos(math.pi*y)*math.sin(math.pi*y)-2*math.pi*y*pow(math.cos(math.pi*y),2))+y*math.sin(math.pi*x)*pow(math.sin(math.pi*y),2)
    f3 = -2*pow(math.pi,2)*pow(math.sin(math.pi*x),2)*math.cos(math.pi*y)+(3*math.pi**2+1)*pow(math.cos(math.pi*x),2)*math.cos(math.pi*y)
    if switch1 == str(1) : f=f1
    if switch1 == str(2) : f=f2
    if switch1 == str(3) : f=f3
    return f
  
print("N   Iteration Error     Rel_error")
for m in mbig:
  nx = m            #Grid Points along X direction
  ny = m           #Grid Points along Y direction
  xmin = 0
  xmax = 1
  ymin = 0
  ymax = 1
  dx = (xmax - xmin) / (nx - 1)         
  dy = (ymax - ymin) / (ny - 1)
  p = np.zeros((ny,nx))
  pd = np.zeros((ny,nx))
  b = np.zeros((ny,nx))              
  x = np.linspace(xmin,xmax,nx)
  y = np.linspace(ymin,ymax,ny)


  dx = (1) / (nx - 1)         
  dy = (1) / (ny - 1)
  actual = np.zeros((nx,ny))
  for i in range (nx):
      for j in range (ny):
          actual[i,j]=g(i*dx,j*dy)

  for i in range (nx):
      for j in range (ny):
          b[i,j]=f(i*dx,j*dy)

  for n in iteration:
      for it in range(n):
          pd = p.copy()
      
    #Central Difference Scheme

          p[1:-1,1:-1] = (((pd[1:-1,2:] + pd[1:-1,:-2])*dx**2 + (pd[2:,1:-1] + pd[:-2,1:-1])*dy**2
                         + b[1:-1,1:-1]*dx**2 * dy**2) / (dx**2*dy**2+2*(dx**2 + dy**2)))

    #Boundary Condition
    
          p[0,:] = p[1,:]
          p[m-1,:] = p[0,:]
          p[:,0] = (4 * p[:,1] - p[:,2]) / 3
          p[:,nx-1] = (4 * p[:,nx-2] - p[:,nx-3]) / 3



        
#Compute l2 error
      error = 0
      denom = 0
      for i in range(1,nx-1):
          for j in range (ny):
              error += pow((actual[i,j]-p[i,j]),2)
              denom +=  pow(actual[i,j],2)
      error = math.sqrt(error)
      rel_error = error/denom
      print("%s   %s   %.6f   %.6f" % (m,n,error,rel_error))
      e.append(error)
      rel_e.append(rel_error)

if switch == str(1):
    pyplot.plot(iteration,rel_e)
    pyplot.title('Plot for relative L2 error with different iterations w/ mesh points = ' + str(mbig[0]))
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Relative l2error')
    pyplot.show()
    pyplot.loglog(iteration,rel_e)
    pyplot.title('Log-log plot for relative L2 error with different iterations w/ mesh points = ' + str(mbig[0]))
    pyplot.xlabel('Log number of iterations')
    pyplot.ylabel('Log relative l2error')
    pyplot.show()
    i = rel_e.index(min(rel_e))+1
    x = np.array(np.log(iteration[0:i])).reshape((-1,1))
    y = np.array(np.log(rel_e[0:i]))
    model = LinearRegression().fit(x, y)
    print('coefficient of determination:', model.score(x,y))
    print('slope:', model.coef_)
elif switch == str(2):
    pyplot.plot(mbig,rel_e)
    pyplot.title('Plot for relative L2 error with different number of mesh points w/ iterations = ' + str(iteration[0]))
    pyplot.xlabel('Number of mesh points in x')
    pyplot.ylabel('Relative l2error')
    pyplot.show()
    pyplot.loglog(mbig,rel_e)
    pyplot.title('Log-log plot for relative L2 error with different number of mesh points w/ iterations = ' + str(iteration[0]))
    pyplot.xlabel('Log number of mesh points in x')
    pyplot.ylabel('Log relative l2error')
    pyplot.show()
    i = rel_e.index(min(rel_e))+1
    x = np.array(np.log(mbig[0:i])).reshape((-1,1))
    y = np.array(np.log(rel_e[0:i]))
    model = LinearRegression().fit(x, y)
    print('coefficient of determination:', model.score(x,y))
    print('slope:', model.coef_)