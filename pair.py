import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(ic, ti, p):
	m, k, req = p
	r0, v0, theta0, omega0, r1, v1, theta1, omega1 = ic

	sub = {M[0]:m[0], M[1]:m[1], K:k, Req:req, R[0]:r0, R[1]:r1, Rdot[0]:v0, Rdot[1]:v1, THETA[0]:theta0, THETA[1]:theta1, THETAdot[0]:omega0, THETAdot[1]:omega1}

	print(ti)

	return [v0,A[0].subs(sub),omega0,ALPHA[0].subs(sub),v1,A[1].subs(sub),omega1,ALPHA[1].subs(sub)]

t = sp.symbols('t')
M = sp.symbols('M0:2')
K, Req = sp.symbols('K Req')
R = dynamicsymbols('R0:2')
THETA = dynamicsymbols('THETA0:2')

Rdot = [R[i].diff(t, 1) for i in range(2)]
Rddot = [R[i].diff(t, 2) for i in range(2)]
THETAdot = [THETA[i].diff(t, 1) for i in range(2)]
THETAddot = [THETA[i].diff(t, 2) for i  in range(2)]

X = [R[i] * sp.cos(THETA[i]) for i in range(2)]
Y = [R[i] * sp.sin(THETA[i]) for i in range(2)]

dR = sp.sqrt((X[1] - X[0])**2 + (Y[1] - Y[0])**2) - Req

Xdot = np.asarray([X[i].diff(t, 1) for i in range(2)])
Ydot = np.asarray([Y[i].diff(t, 1) for i in range(2)])

T = sp.simplify(sp.Rational(1, 2) * sum(M * (Xdot**2 + Ydot**2)))

V = sp.simplify(sp.Rational(1, 2) * K * dR**2)

L = T - V

dLdR = [L.diff(i, 1) for i in R]
dLdRdot = [L.diff(i, 1) for i in Rdot]
ddtdLdRdot = [i.diff(t, 1) for i in dLdRdot]
dLR = np.asarray(ddtdLdRdot) - np.asarray(dLdR)

dLdTHETA = [L.diff(i, 1) for i in THETA]
dLdTHETAdot = [L.diff(i, 1) for i in THETAdot]
ddtdLdTHETAdot = [i.diff(t, 1) for i in dLdTHETAdot]
dLTHETA = np.asarray(ddtdLdTHETAdot) - np.asarray(dLdTHETA)

eqs = []
ddot = []
for i in range(2):
	eqs.append(dLR[i])
	eqs.append(dLTHETA[i])
	ddot.append(Rddot[i])
	ddot.append(THETAddot[i])

sol = sp.solve(eqs,ddot)

A = [sp.simplify(sol[i]) for i in Rddot]
ALPHA = [sp.simplify(sol[i]) for i in THETAddot]

#-------------------------------------------------------

m = np.asarray([1, 1])
k = 25 
req = 5
ro = np.asarray([4, 4])
vo = np.asarray([0 ,0])
thetao = np.asarray([180, 0]) * np.pi/180
omegao = np.asarray([30, 30]) * np.pi/180
mr = 0.25
tf = 30 

p = m,k,req

ic = ro[0], vo[0], thetao[0], omegao[0], ro[1], vo[1], thetao[1], omegao[1]

nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

rth = odeint(integrate, ic, ta, args=(p,))

x = np.asarray([[X[i].subs({R[i]:rth[j,4 * i], THETA[i]:rth[j,4 * i + 2]}) for j in range(nframes)] for i in range(2)],dtype=float)
y = np.asarray([[Y[i].subs({R[i]:rth[j,4 * i], THETA[i]:rth[j,4 * i + 2]}) for j in range(nframes)] for i in range(2)],dtype=float)

ke = np.asarray([T.subs({M[0]:m[0], M[1]:m[1], R[0]:rth[i,0], R[1]:rth[i,4], Rdot[0]:rth[i,1], Rdot[1]:rth[i,5],\
		THETA[0]:rth[i,2], THETA[1]:rth[i,6], THETAdot[0]:rth[i,3], THETAdot[1]:rth[i,7]}) for i in range(nframes)])
pe = np.asarray([V.subs({K:k, Req:req, R[0]:rth[i,0], R[1]:rth[i,4], Rdot[0]:rth[i,1], Rdot[1]:rth[i,5],\
		THETA[0]:rth[i,2], THETA[1]:rth[i,6], THETAdot[0]:rth[i,3], THETAdot[1]:rth[i,7]}) for i in range(nframes)])
E = ke + pe

#-------------------------------------------------------

xmax = x.max() + 2 * mr
xmin = x.min() - 2 * mr
ymax = y.max() + 2 * mr
ymin = y.min() - 2 * mr

dr = np.asarray(np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2))
drmax = max(dr)
theta = np.asarray([np.arccos((y[1] - y[0])/dr)])
nl = int(np.ceil(drmax / (2 * mr)))
l = np.asarray([(dr - 2 * mr)/nl])
h = np.sqrt(mr**2 - (0.5 * l)**2)
flipa = np.asarray([-1 if x[0][i]>x[1][i] and y[0][i]<y[1][i] else 1 for i in range(nframes)])
flipb = np.asarray([-1 if x[0][i]<x[1][i] and y[0][i]>y[1][i] else 1 for i in range(nframes)])
flipc = np.asarray([-1 if x[0][i]<x[1][i] else 1 for i in range(nframes)])
xlo = np.zeros(nframes)
ylo = np.zeros(nframes)
xlo[:] = x[0] + np.sign((y[1] - y[0]) * flipa * flipb) * mr * np.sin(theta)
ylo[:] = y[0] + mr * np.cos(theta)
xl = np.zeros((nl,nframes))
yl = np.zeros((nl,nframes))
xl[0] = xlo + np.sign((y[1]-y[0])*flipa*flipb) * 0.5 * l * np.sin(theta) - np.sign((y[1]-y[0])*flipa*flipb) * flipc * h * np.sin(np.pi/2 - theta)
yl[0] = ylo + 0.5 * l * np.cos(theta) + flipc * h * np.cos(np.pi/2 - theta)
for i in range(1,nl):
	xl[i] = xlo + np.sign((y[1]-y[0])*flipa*flipb) * (0.5 + i) * l * np.sin(theta) - np.sign((y[1]-y[0])*flipa*flipb) * flipc * (-1)**i * h * np.sin(np.pi/2 - theta)
	yl[i] = ylo + (0.5 + i) * l * np.cos(theta) + flipc * (-1)**i * h * np.cos(np.pi/2 - theta)
xlf = np.zeros(nframes)
ylf = np.zeros(nframes)
xlf[:] = x[1] - mr * np.sign((y[1]-y[0])*flipa*flipb) * np.sin(theta)
ylf[:] = y[1] - mr * np.cos(theta)

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	for i in range(2):
		circle=plt.Circle((x[i][frame],y[i][frame]),radius=mr,fc='xkcd:red')
		plt.gca().add_patch(circle)
	plt.plot([xlo[frame],xl[0][frame]],[ylo[frame],yl[0][frame]],'xkcd:cerulean')
	plt.plot([xl[nl-1][frame],xlf[frame]],[yl[nl-1][frame],ylf[frame]],'xkcd:cerulean')
	for i in range(nl-1):
		plt.plot([xl[i][frame],xl[i+1][frame]],[yl[i][frame],yl[i+1][frame]],'xkcd:cerulean')
	plt.title("Mass-Spring Pair")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([float(xmin),float(xmax)])
	plt.ylim([float(ymin),float(ymax)])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('mass_spring_pair.mp4', writer=writervideo)
plt.show()




