# Figure 2: sketch of two-phase training in two parameters.
# L0: non-convex, even, saddle at 0. J = L0 + mu/2|T|^2 + gamma/4|T|^4 - E v0'T is convex,
# v0 along the negative-curvature direction of L0 at 0.
# Phase 1: gradient descent on J to its minimiser; Phase 2: gradient descent on L0.
import os
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 9.5, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in"})
m = np.array([1.7, 0.9]); p = np.array([-0.9, 1.6])
mu, gamma = 1.0, 0.3
def gau(x, y, c, s): return np.exp(-((x-c[0])**2 + (y-c[1])**2)/(2*s*s))
def L0(x, y):
    return (1.0 - 0.95*(gau(x, y, m, .75) + gau(x, y, -m, .75))
            - 0.5*(gau(x, y, p, .45) + gau(x, y, -p, .45)) + 0.06*(x*x + y*y))
def hess0(F, h=1e-4):
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei = np.eye(2)[i]*h; ej = np.eye(2)[j]*h
            H[i, j] = (F(*(ei+ej)) - F(*(ei-ej)) - F(*(-ei+ej)) + F(*(-ei-ej)))/(4*h*h)
    return H
ev, V = np.linalg.eigh(hess0(L0))
v0 = V[:, 0]                       # negative-curvature direction of L0 at 0
if v0 @ (-m) < 0: v0 = -v0         # either sign works; pick one
E_r = 0.19
def J(x, y):
    t = x*x + y*y; return L0(x, y) + mu/2*t + gamma/4*t*t - E_r*(v0[0]*x + v0[1]*y)
def grad(F, w, h=1e-6):
    return np.array([(F(w[0]+h, w[1]) - F(w[0]-h, w[1]))/(2*h),
                     (F(w[0], w[1]+h) - F(w[0], w[1]-h))/(2*h)])
def gd(F, w, eta, n, tol=0):
    P = [w.copy()]
    for _ in range(n):
        g = grad(F, w); w = w - eta*g; P.append(w.copy())
        if tol and np.linalg.norm(g) < tol: break
    return np.array(P)
W0 = np.array([-2.35, 2.05])
P1 = gd(J, W0, 0.02, 40000, tol=1e-6)
Ws = P1[-1]
P2 = gd(L0, Ws, 0.05, 40000, tol=1e-7)
We = P2[-1]
lm = minimize(lambda w: L0(*w), p).x

xs = np.linspace(-3.0, 3.0, 500); ys = np.linspace(-2.4, 2.6, 500)
X, Y = np.meshgrid(xs, ys); Z = L0(X, Y); ZJ = J(X, Y)
fig, ax = plt.subplots(figsize=(3.3, 3.15))
ax.imshow(Z, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin='lower',
          cmap='Blues_r', alpha=0.35, interpolation='bilinear', aspect='auto')
ax.contour(X, Y, Z, levels=np.concatenate([np.linspace(0.12, 1.0, 12), np.linspace(1.1, 1.6, 4)]),
           cmap='Blues_r', linewidths=0.7)
ax.contour(X, Y, ZJ, levels=J(*Ws) + np.array([0.12, 0.6, 1.4, 2.6, 4.2, 6.5, 10.0]),
           colors='0.55', linestyles='--', linewidths=0.6)
ax.plot(P1[:, 0], P1[:, 1], color='#2ca02c', lw=1.8, solid_capstyle='round')
ax.plot(P2[:, 0], P2[:, 1], color='#d62728', lw=1.8, ls=(0, (3.5, 1.8)))
def arrow(P, f, c):
    sarc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    i = int(np.searchsorted(sarc, f*sarc[-1])); j = i + 1
    while np.linalg.norm(P[j]-P[i]) < 0.12 and j < len(P)-1: j += 1
    ax.annotate('', P[j], P[i], arrowprops=dict(arrowstyle='-|>', color=c, lw=1.4,
                mutation_scale=11, shrinkA=0, shrinkB=0))
arrow(P1, 0.05, '#2ca02c'); arrow(P2, 0.5, '#d62728')
box = dict(fc='white', ec='none', alpha=0.85, pad=1.2)
ax.plot(*W0, 'o', color='k', ms=4.5)
ax.text(W0[0]+0.12, W0[1]+0.1, r'$\Theta_0$', bbox=box)
ax.plot(0, 0, '+', color='k', ms=7, mew=1.1, zorder=6)
ax.text(0.14, 0.12, r'$0$', bbox=box, color='0.15')
ax.plot(*Ws, 's', color='#2ca02c', ms=5.5, mec='k', mew=0.6, zorder=5)
ax.text(Ws[0]+0.05, Ws[1]-0.5, r'$\Theta^\star=\arg\min\,J$', bbox=box, color='0.15')
ax.plot(*We, '*', color='#d62728', ms=10, mec='k', mew=0.6, zorder=5)
ax.text(We[0], We[1]-0.42, r'$\Theta_T$', ha='center', color='#d62728', bbox=box)
ax.plot(*(-We), '*', color='white', ms=10, mec='#d62728', mew=0.8, zorder=5)
ax.text(-We[0], -We[1]-0.42, r'$-\Theta_T$', ha='center', color='#d62728', bbox=box)
for q in (lm, -lm):
    ax.plot(*q, 'x', color='k', ms=5.5, mew=1.2)
ax.text(lm[0]+0.18, lm[1]+0.05, 'local min.', fontsize=8.5, bbox=box)
ax.text(-2.92, 0.42, 'Phase 1:\n' r'descent on $J$', color='#2ca02c', fontsize=9, bbox=box)
ax.text(1.05, -2.3, 'Phase 2:\n' r'descent on $L_0$', color='#d62728', fontsize=9, bbox=box)
hd = [Line2D([], [], color='#4a7fb5', lw=0.9, label=r'level sets of $L_0$'),
      Line2D([], [], color='0.55', ls='--', lw=0.8, label=r'level sets of $J$'),
      Line2D([], [], color='#2ca02c', lw=1.8, label=r'Phase 1: $\Theta_0\to\Theta^\star$'),
      Line2D([], [], color='#d62728', lw=1.8, ls=(0, (3.5, 1.8)), label=r'Phase 2: $\Theta^\star\to\Theta_T$')]
ax.legend(handles=hd, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
          frameon=False, fontsize=9, handlelength=1.8, columnspacing=0.8, handletextpad=0.4)
ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); ax.set_aspect('equal')
ax.set_xlabel(r'$\Theta(1)$', labelpad=1); ax.set_ylabel(r'$\Theta(2)$', labelpad=1)
fig.tight_layout(pad=0.3)
os.makedirs('results', exist_ok=True)
fig.savefig('results/fig_two_phase.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig('results/fig_two_phase.png', dpi=200, bbox_inches='tight', pad_inches=0.02)

