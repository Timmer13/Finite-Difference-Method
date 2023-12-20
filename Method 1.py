import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import time

plt.rcParams['animation.html'] = 'html5'

'''
Compute first derivative using central differencing on f
'''
def ddx(f, dx):
    result = np.zeros_like(f)
    result[1:-1,1:-1] = (f[1:-1,2:] - f[1:-1,:-2])/2.0/dx
    return result

def ddy(f, dy):
    result = np.zeros_like(f)
    result[1:-1,1:-1] = (f[2:,1:-1] - f[:-2,1:-1])/2.0/dy
    return result
    
def laplacian(f, dx, dy):
    result = np.zeros_like(f)
    result[1:-1,1:-1] = (f[1:-1,2:] - 2.0*f[1:-1,1:-1] + f[1:-1,:-2])/dx/dx \
                      + (f[2:,1:-1] -2.0*f[1:-1,1:-1] + f[:-2,1:-1])/dy/dy
    return result

def div(u,v,dx,dy):
    return ddx(u,dx) + ddy(v,dy)

nx = 40
ny = 40
lx = 1.0
ly = 1.0
dx = lx/(nx-1)
dy = ly/(ny-1)

ν = 0.01
Ut = 1.0 # m/s

dt = 2.5*dx*dx
print('Re =', Ut*lx/ν)
u = np.zeros([ny,nx])
v = np.zeros([ny,nx])
uh = np.zeros([ny,nx])
vh = np.zeros([ny,nx])
p = np.zeros([ny,nx])

def uv2psi(u, v, ny, nx):
    psi = np.zeros([ny, nx])
    dx = 1.0/(nx-1)
    dy = dx
    for i in range(ny):
        if i == 0:
            psi[i, 1] = 0
        else:
            psi[i, 1] = psi[i-1,1] + 0.5 * (u[i-1, 1]+u[i,1]) * dy
        
        for j in range(nx):
            if j == 0:
                psi[i, j] = 0
            else:
                psi[i, j] = psi[i, j-1] + 0.5 * (v[i, j-1]+v[i,j]) * dx
    return psi

def uv2w(u, v, ny, nx):
    w = np.zeros([ny, nx])
    dx = 1.0/(nx-1)
    dy = dx
    w[1:-1,1:-1] = (v[1:-1, 2:] - v[1:-1,:-2])/2.0/dx - (u[2:,1:-1] - u[:-2,1:-1])/2.0/dy
    return w


start_time = time.time()

t = 0.0
tstep = 5000
tend = tstep*dt

while t < tend:
    # set boundary conditions
    # bottom wall
    u[0,:] = 0.0 
    v[0,:] = 0.0     
    # top wall    
    u[-1,:] = Ut
    v[-1,:] = 0.0
    # left wall
    u[:,0] = 0.0
    v[:,0] = 0.0
    # right wall
    u[:,-1] = 0.0
    v[:,-1] = 0.0
        
    # do the x-momentum RHS
    # u rhs: - d(uu)/dx - d(vu)/dy + ν d2(u)
    
    uRHS = - ddx(u*u,dx) - ddy(v*u,dy) + ν*laplacian(u,dx,dy)
    # v rhs: - d(uv)/dx - d(vv)/dy + ν d2(v)
    vRHS = - ddx(u*v,dx) - ddy(v*v,dy) + ν*laplacian(v,dx,dy)
    
    uh = u + dt*uRHS
    vh = v + dt*vRHS
    
    # next compute the pressure RHS: prhs = div(un)/dt + div( [urhs, vrhs])
    
    # finally compute the true velocities
    # u_{n+1} = uh - dt*dpdx
    p -= dt * div(u,v,dx,dy)
    
    u = uh - dt*ddx(p,dx)
    v = vh - dt*ddy(p,dx)  
    t += dt
    
print(time.time()-start_time)
psi = uv2psi(u, v, ny, nx)
p1 = plt.contour(psi) 
plt.clabel(p1, inline=10, fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Streamlines')
plt.text(10, -8, 'Re = %.2f' %(Ut*lx/ν)+'  h = 1/%i' %nx+' t = %i*dt' %tstep)
plt.text(35, -8, 'psi_max = %.4f' %psi.max())

plt.show()

w = uv2w(u, v, ny, nx)
p2 = plt.contour(w) 
plt.clabel(p2, inline=10, fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lines of isovorticity')
plt.text(10, -8, 'Re = %.2f' %(Ut*lx/ν)+'  h = 1/%i' %nx+' t = %i*dt' %tstep)

plt.show()

p3 = plt.contour(p) 
plt.clabel(p3, inline=10, fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Isobars')
plt.text(10, -8, 'Re = %.2f' %(Ut*lx/ν)+'  h = 1/%i' %nx+' t = %i*dt' %tstep)

plt.show()