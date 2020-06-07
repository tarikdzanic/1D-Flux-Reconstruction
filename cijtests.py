import numpy as np
import generatemats
from quadrules import quadrules


solptstype = 'gausslegendre'

def getGraphSource(i):
    global solGradMat, leftcorrGrad, rightcorrGrad, leftsolGrad, rightsolGrad
    global basiswts

    Fs = 1.0

    cij = []
    #val = max(0.5*np.abs(leftcorrGrad[i])*Fs, 0.5*np.abs(leftsolGrad[i]))
    val = 0.5*np.abs(leftcorrGrad[i])*Fs
    #print(0.5*np.abs(leftcorrGrad[i])*Fs, np.abs(leftsolGrad[i]))
    cij.append(val)
    cij.append(val)

    for j in range(p+1):
        Ss = 1.
        val = max(np.abs(solGradMat[i,j]), np.abs(solGradMat[j,i]))
        cij.append(val*Ss)

    #val = max(0.5*np.abs(rightcorrGrad[i])*Fs, 0.5*np.abs(rightsolGrad[i]))
    val = 0.5 * np.abs(rightcorrGrad[i]) * Fs
    #print(0.5*np.abs(rightcorrGrad[i])*Fs, np.abs(rightsolGrad[i]))
    cij.append(val*Fs)
    cij.append(val*Fs)

    return cij

detJ = 1.0
p = 3
s = 3
dudx = 4
dt = 0.01

i = 1
# Get basis points in computational space
[basispts, basiswts] = quadrules.getPtsAndWeights(p, solptstype)

# Get interpolation and differentiation matrices
solGradMat = generatemats.solGradMat(basispts)
leftcorrGrad = generatemats.corrGradVect(basispts, 'left')
rightcorrGrad = generatemats.corrGradVect(basispts, 'right')
leftsolGrad = generatemats.solGradInterVect(basispts, -1.)
rightsolGrad = generatemats.solGradInterVect(basispts, 1.)

vel = lambda x: 1.0 + dudx*(x+1.0)/2.0

u = [vel(-1), vel(-1)]
for j in range(len(basispts)):
    u.append(vel(basispts[j]))
u.append(vel(1))
u.append(vel(1))
u = np.array(u)
'''
cij = getGraphSource(i)
conts = []
for j in range(len(cij)):
    conts.append(np.abs(cij[j])*(u[j]-u[i]))
print(u)
print(cij)
print(conts)
print(np.sum(conts))
'''
scont = []
for i in range(p+1):
    cij = getGraphSource(i)
    conts = []
    for j in range(len(cij)):
        fac = basiswts[i]
        conts.append(s*fac*np.abs(cij[j])*(u[j]-u[i+2]))
        #conts.append(np.abs(cij[j]))
    #print(conts)
    #print(np.sum(conts))
    scont.append(np.sum(conts))

print('dU n+1', np.ones_like(u[2:-2])*(-dt*s*dudx))

scont = np.array(scont)
AV = scont
#nscont = scont/min(scont)
#nscont = scont/np.sum(scont)
print('AV', scont)
avdiv = s*dudx-scont
print('dU w/ AV', -dt*(avdiv))


divmag = np.linalg.norm(s*dudx)
avmag = np.linalg.norm(scont)
divscale = divmag/(divmag + avmag)
corrdiv = divscale*(s*dudx-scont)
print(avdiv, corrdiv, divscale)

print('U n+1 corr', -dt*(corrdiv))
