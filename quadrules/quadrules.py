import numpy as np

def getPtsAndWeights(p, solpts):	
	f = open('quadrules/line_' + solpts + '_p' + str(p) + '.txt', 'r')
	fl = f.readlines()

	fpts = np.zeros(p+1)
	w =np.zeros(p+1)

	for i in range(len(fl)):
		fpts[i] = float(fl[i].split()[0])
		w[i] = float(fl[i].split()[1])
	return([fpts, w])

