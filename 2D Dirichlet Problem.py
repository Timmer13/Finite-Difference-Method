import numpy                          #give mathamatical or matrix expressions like array
import math
from matplotlib import pyplot, cm           #plotting library that we will use to plot our results
from mpl_toolkits.mplot3d import Axes3D     #To plot a projected 3D result, make sure that you have added the Axes3D library

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
    g1 = math.sin(2*math.pi*x)*math.sin(2*math.pi*y)
    g2 = pow(math.sin(math.pi*x),2)*math.sin(math.pi*y)
    g3 = pow(x,2)*math.sin(math.pi*x)*math.sin(math.pi*y)
    if switch1 == str(1) : g=g1
    if switch1 == str(2) : g=g2
    if switch1 == str(3) : g=g3
    return g

#RHS of Dirichlet problem

def f(x,y):
    # the RHS f
    f1 = (8*pow(math.pi,2)+1)*math.sin(2*math.pi*x)*math.sin(2*math.pi*y)
    f2 = (3*pow(math.pi,2)+1)*pow(math.sin(math.pi*x),2)*math.sin(math.pi*y)-2*pow(math.pi,2)*pow(math.cos(math.pi*x),2)*math.sin(math.pi*y)
    f3 = math.sin(math.pi*y)*((2*pow(math.pi,2)*pow(x,2)-2+pow(x,2))*math.sin(math.pi*x)-4*math.pi*x*math.cos(math.pi*x))
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
  p = numpy.zeros((ny,nx))
  pd = numpy.zeros((ny,nx))
  b = numpy.zeros((ny,nx))              
  x = numpy.linspace(xmin,xmax,nx)
  y = numpy.linspace(ymin,ymax,ny)


  dx = (1) / (nx - 1)         
  dy = (1) / (ny - 1)
  actual = numpy.zeros((nx,ny))
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
    
          p[0,:] = 0
          p[ny-1,:] = 0
          p[:,0] = 0
          p[:,nx-1] = 0



        
#Compute l2 error
      error = 0
      denom = 0
      for i in range (nx):
          for j in range (ny):
              error += pow(actual[i,j] - p[i,j],2)
              denom +=  pow(actual[i,j],2)

      error = math.sqrt(error)
      rel_error = math.sqrt(error/denom)
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
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Relative l2error')
    pyplot.show()
elif switch == str(2):
    pyplot.plot(mbig,rel_e)
    pyplot.title('Plot for relative L2 error with different number of mesh points w/ iterations = ' + str(iteration[0]))
    pyplot.xlabel('Number of mesh points in x')
    pyplot.ylabel('Relative l2error')
    pyplot.show()
    pyplot.loglog(mbig,rel_e)
    pyplot.title('Log-log plot for relative L2 error with different number of mesh points w/ iterations = ' + str(iteration[0]))
    pyplot.xlabel('Number of mesh points in x')
    pyplot.ylabel('Relative l2error')
    pyplot.show()