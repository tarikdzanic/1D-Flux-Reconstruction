import numpy as np
from poly import poly
from quadrules import quadrules

def getCorrector(corrfun, psol, pflux):
	if corrfun == 'DGFR':
		c0 = getLegendreCoeffs(psol)
		c1 = getLegendreCoeffs(psol+1)
		c0 = poly.scalePoly(c0, 0.5)
		c1 = poly.scalePoly(c1, 0.5)

		c0l = poly.evalPoly(-1., c0)
		c0r = poly.evalPoly(1., c0)

		c1l = poly.evalPoly(-1., c1)
		c1r = poly.evalPoly(1., c1)

		lsum = c0l + c1l
		rsum = c0r + c1r

		A = np.array([[c0l, c1l], [c0r, c1r]])
		bl = np.array([1,0])
		br = np.array([0,1])

		sl = np.linalg.solve(A,bl)
		sr = np.linalg.solve(A,br)
	
		corr_left = poly.addPoly(poly.scalePoly(c0, sl[0]),poly.scalePoly(c1, sl[1]))
		corr_right = poly.addPoly(poly.scalePoly(c0, sr[0]),poly.scalePoly(c1, sr[1]))



	if corrfun == 'SDFR':
		pl = quadrules.getPtsAndWeights(psol + 1, 'gausslegendrelobatto')[0]
		ul = np.zeros(len(pl))
		ul[0] = 1.0

		pr = quadrules.getPtsAndWeights(psol + 1, 'gausslegendrelobatto')[0]		
		ur = np.zeros(len(pr))
		ur[-1] = 1.0

		corr_left  = poly.makePoly(pl, [ul])[0]
		corr_right = poly.makePoly(pr, [ur])[0]

	if corrfun == 'None':
		corr_left = np.array([0])
		corr_right = np.array([0])

	def correctFlux(fp, fl, fr, nvars):
		fpout = np.zeros((nvars, max(len(fp[0]), len(corr_left))))
		for i in range(nvars):
			fpinleft = poly.evalPoly(-1., fp[i])
			dfleft = fl[i] - fpinleft
			cf_left = poly.scalePoly(corr_left, dfleft)
			fpout[i] = poly.addPoly(fp[i], cf_left)

			fpinright = poly.evalPoly(1., fpout[i])
			dfright = fr[i] - fpinright
			cf_right = poly.scalePoly(corr_right, dfright)
			fpout[i] = poly.addPoly(fpout[i], cf_right)

		return fpout

	return correctFlux


def getLegendreCoeffs(p):
	if p == 0:
		return [1]
	elif p == 1:
		return [0, 1]
	else:
		n = p - 1
		# (n+1)*P_(n+1) = (2*n + 1)*x*P_n - n*P_(n-1)
		Pn = getLegendreCoeffs(n)
		Pnm1 = getLegendreCoeffs(n-1)
		xPn = [0]
		for i in range(len(Pn)):
			xPn.append(Pn[i])

		f1, f2 = [], []
		for i in range(len(xPn)):
			f1.append(((2*n+1)/(n+1))*xPn[i])

		for i in range(len(Pnm1)):
			f2.append(-((n)/(n+1))*Pnm1[i])

		return poly.addPoly(f1,f2)
		