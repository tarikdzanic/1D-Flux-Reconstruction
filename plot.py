import pickle
import csv
import matplotlib.pyplot as plt


path = []
'''
path.append('C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\data\\sod_p2_dof255_l2_FScentered_LMguermond_GVUFalse_DSFalse')
path.append('C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\data\\sod_p2_dof255_l2_FScentered_LMguermond_GVUFalse_DSTrue')
path.append('C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\data\\sod_p3_dof256_l2_FScentered_LMguermond_GVUFalse_DSFalse')
path.append('C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\data\\sod_p3_dof256_l2_FScentered_LMguermond_GVUFalse_DSTrue')
leg = ['Reference', 'P2', 'P2-scaled', 'P3', 'P3-scaled']
'''

'''
path.append('C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\data\\sod_p2_dof510_l2_FScentered_LMguermond_GVUFalse_DSFalse')
path.append('C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\data\\sod_p2_dof510_l2_FScentered_LMguermond_GVUFalse_DSTrue')
path.append('C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\data\\sod_p3_dof512_l2_FScentered_LMguermond_GVUFalse_DSFalse')
path.append('C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\data\\sod_p3_dof512_l2_FScentered_LMguermond_GVUFalse_DSTrue')
leg = ['Reference', 'P2', 'P2-scaled', 'P3', 'P3-scaled']
'''

path.append('C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\data\\sod_p3_dof512_l2_FScentered_LMguermond_GVUFalse_DSTrue')
path.append('C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\data\\sod_p3_dof512_l2_FSguermond_LMguermond_GVUFalse_DSTrue')
path.append('C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\data\\sod_p3_dof512_l2_FSguermond_LMguermond_GVUTrue_DSTrue')
leg = ['Reference', 'Centered', 'Rusanov-noGVU', 'Rusanov-GVU']




xref, rhoref, velref, pref = [], [], [], []

if 'sod' in path[0]:
	refpath = 'C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\sod_reference0p20.csv'
elif 'shu-osher' in path[0]:
	refpath = 'C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\shu-osher_reference.csv'
with open(refpath, 'r') as f:
	reader = csv.reader(f)
	next(reader)
	for row in reader:
		xref.append(float(row[0]))
		rhoref.append(float(row[1]))
		velref.append(float(row[2]))
		pref.append(float(row[3]))

gamma = 1.4
plt.figure(0)
plt.plot(xref, rhoref, 'k-')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid()
plt.figure(1)
plt.plot(xref, velref, 'k-')
plt.xlabel('x')
plt.ylabel('Velocity')
plt.grid()
plt.figure(2)
plt.plot(xref, pref, 'k-')
plt.xlabel('x')
plt.ylabel('Pressure')
plt.grid()


for ppath in path:
	with open(ppath, 'rb') as f:
		[meta, x, sol] = pickle.load(f)

	plt.figure(0)
	plt.plot(x, sol[:, 0], '-')
	plt.legend(leg)
	plt.figure(1)
	plt.plot(x, sol[:, 1]/sol[:,0], '-')
	plt.legend(leg)
	plt.figure(2)
	plt.plot(x, (gamma - 1.0) * (sol[:, 2] - 0.5 * (sol[:, 1] ** 2) / sol[:, 0]), '-')
	plt.legend(leg)

plt.show()