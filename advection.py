import numpy as np
from scipy import linalg

import generatemats
from quadrules import quadrules
import matplotlib.pyplot as plt


p = 4
dom = [0., 1.]
nt = 100
nElems = 50
dt = 2e-3
gamma = 1.4
solptstype = 'gausslegendre'
fluxsplitmethod = ['centered', 'upwind'][1]
graphvisc = True
lbc = ['free', 'fixed'][1]
rbc = ['free', 'fixed'][0]
graphupwind = False
pscale = False
cadvec = 1.0


def initialConditions(sol, x):
	global gamma

	sol[:,0] = np.sin(2*np.pi*x)
	sol[:,1] = np.sin(2*np.pi*x)
	sol[:,2] = np.sin(2*np.pi*x)
	sol[:,0] = x
	sol[:,1] = x
	sol[:,2] = x


	return [sol, 0, 0]

def divF(sol):
	global p, nElems, detJ
	global interpLeft, interpRight, solGradMat, leftcorrGrad, rightcorrGrad

	stride = p+1
	divF = np.zeros_like(sol)
	for eidx in range(nElems):
		locsol = sol[stride*eidx:stride*(eidx+1), :]
		locf = calcFlux(locsol)

		divF_uncorr = np.matmul(solGradMat,locf)
		dfl = commonFlux('left', eidx) - calcFlux(np.matmul(interpLeft,locsol))
		dfr = commonFlux('right', eidx) - calcFlux(np.matmul(interpRight,locsol))
		divF_corr = divF_uncorr + np.outer(leftcorrGrad, dfl) + np.outer(rightcorrGrad, dfr)


		divF[stride*eidx:stride*(eidx+1),:] = divF_corr/detJ
		if graphvisc:
			graphsource = getGraphSource(eidx)
			divF[stride*eidx:stride*(eidx+1),:] += -graphsource
			#print(divF_corr/detJ, graphsource)
			#input()


	return divF

def calcFlux(sol):
	global gamma
	global cadvec

	if np.ndim(sol) == 1:
		sol = np.reshape(sol,(1,len(sol)))

	f = np.zeros_like(sol)

	f[:,0] = cadvec*sol[:,0]
	f[:,1] = cadvec*sol[:,1]
	f[:,2] = cadvec*sol[:,2]

	return f

def calcWaveSpeed(sol_left, sol_right):
	global cadvec
	return cadvec

def fluxSplit(sol_left, sol_right):
	global fluxsplitmethod, gamma, cadvec

	if fluxsplitmethod  == 'centered':
		return (0.5*calcFlux(sol_left) + 0.5*calcFlux(sol_right))
	elif fluxsplitmethod  == 'upwind':
		if cadvec > 0.0:
			return calcFlux(sol_left)
		else:
			return calcFlux(sol_right)


def commonSol(side, eidx):
	global sol, p
	global interpLeft, interpRight
	global lbc, rbc, lval, rval

	stride = p+1
	if side == 'left':
		# Get solution at basis points
		sol_right = sol[eidx    *stride:(eidx+1)*stride, :]
		sol_rint = np.matmul(interpLeft, sol_right)
		if eidx == 0:
			sol_left = sol[(eidx-1)*stride:, :]
			sol_lint = np.matmul(interpRight, sol_left)
		else:
			sol_left = sol[(eidx-1)*stride:eidx    *stride, :]
			sol_lint = np.matmul(interpRight, sol_left)
	else:
		# Get solution at basis points
		sol_left = sol[eidx*stride    :(eidx+1)*stride, :]
		sol_lint = np.matmul(interpRight, sol_left)
		if eidx == nElems - 1:
			sol_right = sol[:stride, :]
			sol_rint = np.matmul(interpLeft, sol_right)
		else:
			sol_right = sol[(eidx+1)*stride:(eidx+2)*stride, :]
			sol_rint = np.matmul(interpLeft, sol_right)

	return [sol_lint, sol_rint]

def commonFlux(side, eidx):
	[sol_lint, sol_rint] = commonSol(side, eidx)
	fcomm = fluxSplit(sol_lint, sol_rint)
	return fcomm

def interpFluxSplit(ul, ur):
	tol = 1e-5
	f = [calcFlux(ul)[0],calcFlux(ur)[0], fluxSplit(ul, ur)[0]]

	interps = np.zeros((nvars, 2))
	for j in range(nvars):
		if abs(f[2][j]-f[0][j]) < tol and abs(f[2][j]-f[1][j]) < tol:
			interps[j, :] = [0.5, 0.5]
		elif abs(f[2][j]-f[0][j]) < tol:
			interps[j, :] = [1., 0.]
		elif abs(f[2][j]-f[1][j]) < tol:
			interps[j, :] = [0., 1.]
		else:
			fint = (f[2][j] - f[0][j])/(f[1][j] - f[0][j] + tol)
			fint = min(max(0., fint), 1.)
			interps[j, :] = [ 1.-fint,  fint]
	return interps

def getGraphSource(eidx):
	global sol, nElems, p, detJ
	global solGradMat, leftcorrGrad, rightcorrGrad
	global basiswts, graphupwind

	stride = p+1
	u = sol[stride*eidx:stride*(eidx+1),:]
	graphsource = np.zeros_like(u)

	[ul_ext, ul_int]  = commonSol('left', eidx)
	[ur_int, ur_ext]  = commonSol('right', eidx)

	# u at left/right flux points
	ufl = np.vstack((ul_ext, ul_int))
	ufr = np.vstack((ur_int, ur_ext))

	if graphupwind:
		nl = interpFluxSplit(ul_ext, ul_int)
		nr = -interpFluxSplit(ur_int, ur_ext)
	else:
		nl = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
		nr = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])


	for i in range(p+1):
		for j in range(p+1):
			du = u[j,:] - u[i,:]
			lmax = calcWaveSpeed(u[j,:], u[i,:])
			cij = solGradMat[i,j]
			dij = np.abs(cij)*lmax/detJ
			if pscale:
				graphsource[i,:] += basiswts[i]*dij*du/(2*(p+1))
			else:
				graphsource[i,:] += dij*du

		# Left flux points
		cij = leftcorrGrad[i]
		for j in range(2):
			du = ufl[j,:] - u[i,:]
			lmax = calcWaveSpeed(ufl[j,:] , u[i,:])
			dij = np.abs(nl[:,j]*cij)*lmax/detJ
			if pscale:
				graphsource[i,:] += dij*du
			else:
				graphsource[i,:] += dij*du

		# Right flux points
		cij = rightcorrGrad[i]
		for j in range(2):
			du = ufr[j,:] - u[i,:]
			lmax = calcWaveSpeed(ufr[j,:] , u[i,:])
			dij = np.abs(nr[:,j]*cij)*lmax/detJ
			if pscale:
				graphsource[i,:] += dij*du
			else:
				graphsource[i,:] += dij*du
	return graphsource

def DEBUGgetGraphSource(eidx):
	global sol, nElems, p, detJ
	global solGradMat, leftcorrGrad, rightcorrGrad 
	global lambdamethod, basiswts, fluxsplitmethod
	fluxsplitmethod = 'centered'

	stride = p+1
	u = sol[stride*eidx:stride*(eidx+1),:]
	graphsource = np.zeros_like(u)

	[ul_ext, ul_int]  = commonSol('left', eidx)
	[ur_int, ur_ext]  = commonSol('right', eidx)

	# u at left/right flux points
	ufl = np.vstack((ul_ext, ul_int))
	ufr = np.vstack((ur_int, ur_ext))

	'''
	#CALCULATE CIJ = DIV FLUX TERMS
	for i in range(p+1):
		for j in range(p+1):			
			cij = solGradMat[i,j]
			dij = cij*(calcFlux(u[j,:])[0])/detJ
			graphsource[i,:] += dij
		# Left flux points
		cij = leftcorrGrad[i]
		n = [0.5, -0.5]
		for j in range(2):
			dij = cij*(calcFlux(ufl[j,:])[0])/detJ
			graphsource[i,:] += n[j]*dij

		# Right flux points
		cij = rightcorrGrad[i]
		n = [-0.5, 0.5]
		for j in range(2):
			dij = cij*(calcFlux(ufr[j,:])[0])/detJ
			graphsource[i,:] += n[j]*dij
	'''
	'''
	# CALCULATE SUM CIJ = 0
	for i in range(p+1):
		for j in range(p+1):
			cij = solGradMat[i,j]
			graphsource[i,:] += cij
		# Left flux points
		cij = leftcorrGrad[i]
		n = [0.5, -0.5]
		for j in range(2):
			graphsource[i,:] += n[j]*cij

		# Right flux points
		cij = rightcorrGrad[i]
		n = [-0.5, 0.5]
		for j in range(2):
			graphsource[i,:] += n[j]*cij
		#graphsource[i,:] *= basiswts[i]
	'''


	return graphsource

def step(sol,dt):
	k1 = -dt*divF(sol)
	k2 = -dt*divF(sol+k1/2.)
	k3 = -dt*divF(sol+k2/2.)
	k4 = -dt*divF(sol+k3)
	return sol + k1/6. + k2/3. + k3/3. + k4/6.

def getPointLocations(basispts, dom, nElems):
	ptsperelem = len(basispts)
	x = np.zeros(ptsperelem*nElems)
	for i in range(nElems):
		xi = dom[0] + i*(dom[1] - dom[0])/(nElems)
		xf = dom[0] + (i+1)*(dom[1] - dom[0])/(nElems)
		for j in range(ptsperelem):
			x[i*ptsperelem + j] = xi + (xf-xi)*0.5*(basispts[j]+1.)
	return x

def integrateDomain(sol):
	global basiswts
	global nElems
	global p

	domint = np.zeros(3)
	stride = p+1
	for i in range(nElems):
		domint[:] += np.dot(basiswts, sol[i*stride:(i+1)*stride,:])
	return domint



detJ = (dom[1] - dom[0])/(2.0*nElems)
nvars = 3
sol = np.ones((nElems*(p+1), nvars))

# Get basis points in computational space
[basispts, basiswts] = quadrules.getPtsAndWeights(p, solptstype) 
x = getPointLocations(basispts, dom, nElems)
[sol, lval, rval] = initialConditions(sol, x)

# Get interpolation and differentiation matrices
interpLeft = generatemats.interpVect(basispts, -1.)
interpRight = generatemats.interpVect(basispts, 1.)
solGradMat = generatemats.solGradMat(basispts)
leftcorrGrad = generatemats.corrGradVect(basispts, 'left')
rightcorrGrad = generatemats.corrGradVect(basispts, 'right')

for t in range(nt):
	sol = step(sol, dt)
	#print('T: ', t*dt, 'L_inf norm: ', np.max(sol, axis=0))
	#print('T: ', t*dt, 'L_2 norm: ', np.linalg.norm(sol, axis=0))
	print('T: ', t*dt, 'Integral: ', integrateDomain(sol))


plt.figure(0)
plt.plot(x, sol[:,0], '-')
#plt.plot(x, np.sin(2*np.pi*(x + cadvec*(-nt*dt))), '.')
plt.plot(x, (x + cadvec*(-nt*dt)), '.')
plt.xlabel('x')
plt.ylabel('U')
plt.title('P = {0}, DOF = {1}, t = {2} \n Flux {3}, GV-Upwind {4}, PScale {5}'.format(p, (p+1)*nElems, nt*dt, fluxsplitmethod, graphupwind, pscale))
plt.show()




