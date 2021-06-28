# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy import sparse 
from scipy.sparse import linalg
import numpy as np
import matplotlib.cm as cm

epsilon0 = 8.852e-12
kB = 1.38e-23
q = 1.602e-19      # C
me = 9.11e-31      # kg
hbar = 1.054e-34   # m2kg/s

def aa(Efn, Ec0, Efp, Ev0, Nc, Nv, T, solve_electron='True', solve_hole='True'):
  v = 0
  if solve_electron:
    v += -1*q**2*Nc*np.exp(Efn/kB/T)*np.exp(-Ec0/kB/T)/epsilon0
  if solve_hole:
    v += +1*q**2*Nv*np.exp(-Efp/kB/T)*np.exp(Ev0/kB/T)/epsilon0
  return v

def an(Ec, mu_n, T, Nc):
  return mu_n*kB*T*Nc*np.exp(-Ec/kB/T)

def ap(Ev, mu_p, T, Nv):
  return -1*mu_p*kB*T*Nv*np.exp(Ev/kB/T)

def an_half(i, j, Ec0, pos, mu_n, T, Nc):
  iL, iR = (int(i), round(i))
  jL, jR = (int(j), round(j))
  Ec_avg = (Ec0[pos[(iL, jL)][0]] + Ec0[pos[(iR, jR)][0]])/2
  return mu_n*kB*T*Nc*np.exp(-Ec_avg/kB/T)

def ap_half(i, j, Ev0, pos, mu_p, T, Nv):
  iL, iR = (int(i), round(i))
  jL, jR = (int(j), round(j))
  Ev_avg = (Ev0[pos[(iL, jL)][0]] + Ev0[pos[(iR, jR)][0]])/2
  return -1*mu_p*kB*T*Nv*np.exp(Ev_avg/kB/T)

def get_Ev(pos, Ec0, Egs):
  Ev0 = np.zeros(len(pos))
  for ij in pos:
    i, j = ij
    Ev0[pos[(i, j)][0]] = Ec0[pos[(i, j)][0]] - Egs(i, j)
  return Ev0

def PoissonSolver(pos, dx, dy, eps, Xis, Nds, Nc, Nv, T,  Efn, Ec0, Efp, Ev0, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, solve_electron, solve_hole):
  matrix_elements = []
  Q = []

  for ij in pos:
    i, j = ij
    cond_x1 = ((i+1, j) not in insul_bound)
    cond_x2 = ((i-1, j) not in insul_bound)
    cond_y1 = ((i, j+1) not in insul_bound)
    cond_y2 = ((i, j-1) not in insul_bound)

    # LHS
    #---------------------------
    # fill diagonal 
    # insul bound: laplace(Ec0) = Q = 0
    v = (  ( - eps(i+0.5, j)/dx/dx - eps(i-0.5, j)/dx/dx) * ( cond_x1 and cond_x2 )
          +( - eps(i, j+0.5)/dy/dy - eps(i, j-0.5)/dy/dy) * ( cond_y1 and cond_y2 ) )
    matrix_elements.append((pos[(i, j)][0], pos[(i, j)][0], v))
    
    # fill off-diag
    # insul bound: laplace(Ec0) = Q = 0
    try:
      v = eps(i+0.5, j)/dx/dx * ( cond_x1 and cond_x2 )
      matrix_elements.append((pos[(i, j)][0], pos[(i+1, j)][0], v))
    except:
      pass
    try:
      v = eps(i-0.5, j)/dx/dx * ( cond_x1 and cond_x2 )
      matrix_elements.append((pos[(i, j)][0], pos[(i-1, j)][0], v))
    except:
      pass
    try:
      v = eps(i, j+0.5)/dy/dy * ( cond_y1 and cond_y2 )
      matrix_elements.append((pos[(i, j)][0], pos[(i, j+1)][0], v))
    except:
      pass
    try:
      v = eps(i, j-0.5)/dy/dy * ( cond_y1 and cond_y2 )
      matrix_elements.append((pos[(i, j)][0], pos[(i, j-1)][0], v))
    except:
      pass

    # fill RHS
    # insul bound: laplace(Ec0) = Q = 0
    # contact bound: Ec0 = phiB - V
    v = (aa(Efn[pos[ij][0]], Ec0[pos[ij][0]], Efp[pos[ij][0]], Ev0[pos[ij][0]], Nc, Nv, T, solve_electron, solve_hole)
          + q**2*Nds(i, j)/epsilon0
          - q*(eps(i+0.5, j)*(Xis(i+1, j)-Xis(i, j))-eps(i-0.5, j)*(Xis(i, j)-Xis(i-1, j)))/dx/dx
          - q*(eps(i, j+0.5)*(Xis(i, j+1)-Xis(i, j))-eps(i, j-0.5)*(Xis(i, j)-Xis(i, j-1)))/dy/dy
          - q*eps(i+0.5, j)*(phiB1-V1)/dx/dx*( ((i+1, j) in contact1_bound) or ((i-1, j) in contact1_bound) )
          - q*eps(i, j+0.5)*(phiB1-V1)/dy/dy*( ((i, j+1) in contact1_bound) or ((i, j-1) in contact1_bound) )
          - q*eps(i-0.5, j)*(phiB2-V2)/dx/dx*( ((i+1, j) in contact2_bound) or ((i-1, j) in contact2_bound) )
          - q*eps(i, j-0.5)*(phiB2-V2)/dy/dy*( ((i, j+1) in contact2_bound) or ((i, j-1) in contact2_bound) )
            )
    Q.append(v)
  
  # finalize sparse matrix & vector
  row, col, data = list(zip(*matrix_elements))
  depdxdddx = sparse.csr_matrix((data, (row, col)))
  zero_order_term = sparse.diags(aa(Efn, Ec0, Efp, Ev0, Nc, Nv, T, solve_electron, solve_hole)/kB/T, 0, format='csc')
  LHS = depdxdddx + zero_order_term
  RHS = Q - depdxdddx*Ec0

  # plot sparse matrix
  #plt.scatter(row, col)
  #plt.axis('scaled')
  #plt.show()

  # solve Ec
  return sparse.linalg.spsolve(LHS, RHS)

def calc_shadow_Ec0(pos, shadow, eps, Ec0, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound):
  # include boundary/shadow points into Ec0
  shadow_Ec0 = []
  shadow_pos = {}
  ind=0
  for ij in shadow:
    i, j = ij
    shadow_pos[ij] = (ind+len(Ec0), mesh_data(i, j))
    ind += 1
    if ij in insul_bound:
      # insul bound: laplace(Ec0) = Q = 0
      # find nearest point that is a normal point
      near_pos = [a for a in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)] if a in pos][0]

      # determine vector pointing toward that normal point
      i_diff, j_diff = (near_pos[0] - i, near_pos[1] - j)
      eps_tmp = eps(i+i_diff/2, j+i_diff/2)/eps(i+3*i_diff/2, j+3*i_diff/2)

      # ep1*(Ec-Ec) = ep2*(Ec-Ec)
      Ec_tmp = (eps_tmp+1)*Ec0[pos[(i+i_diff, j+j_diff)][0]] - eps_tmp*Ec0[pos[(i+2*i_diff, j+2*j_diff)][0]]
      shadow_Ec0.append(Ec_tmp)
    elif ij in contact1_bound:
      # contact bound: Ec0 = phiB - V
      shadow_Ec0.append(q*(phiB1-V1))
    elif ij in contact2_bound:
      # contact bound: Ec0 = phiB - V
      shadow_Ec0.append(q*(phiB2-V2))
    else:
      print('unassigned boundary!')
  return shadow_Ec0, shadow_pos

def LLSolver(pos, shadow_pos, dx, dy, m_eff, extended_Ec0, shift_factor = 2):
  LL_elements = []
  LLRHS = []

  # for stability shift Ec by a factor, will shift back after obtain eff pot
  extended_Ec0_shift = np.array(extended_Ec0) + shift_factor*q*np.ones(len(extended_Ec0))

  for ij in pos:
    i, j = ij
    
    # LHS
    #---------------------------
    # fill diagonal 
    v = - hbar*hbar/2/m_eff*(-2)/dx/dx - hbar*hbar/2/m_eff*(-2)/dy/dy + extended_Ec0_shift[pos[(i, j)][0]]
    LL_elements.append((pos[(i, j)][0], pos[(i, j)][0], v))

    # fill off-diag
    try:
      v = -hbar*hbar/2/m_eff*(1)/dx/dx
      LL_elements.append((pos[(i, j)][0], pos[(i+1, j)][0], v))
    except:
      pass
    try:
      v = -hbar*hbar/2/m_eff*(1)/dx/dx
      LL_elements.append((pos[(i, j)][0], pos[(i-1, j)][0], v))
    except:
      pass
    try:
      v = -hbar*hbar/2/m_eff*(1)/dy/dy
      LL_elements.append((pos[(i, j)][0], pos[(i, j+1)][0], v))
    except:
      pass
    try:
      v = -hbar*hbar/2/m_eff*(1)/dy/dy
      LL_elements.append((pos[(i, j)][0], pos[(i, j-1)][0], v))
    except:
      pass

    # fill RHS
    #---------------------------  
    v = 1
    try:
      u0 = 1/extended_Ec0_shift[shadow_pos[(i+1, j)][0]]
      v += hbar*hbar/2/m_eff/dx/dx*u0
    except:
      pass
    try:
      u0 = 1/extended_Ec0_shift[shadow_pos[(i-1, j)][0]]
      v += hbar*hbar/2/m_eff/dx/dx*u0
    except:
      pass
    try:
      u0 = 1/extended_Ec0_shift[shadow_pos[(i, j+1)][0]]
      v += hbar*hbar/2/m_eff/dy/dy*u0
    except:
      pass
    try:
      u0 = 1/extended_Ec0_shift[shadow_pos[(i, j-1)][0]]
      v += hbar*hbar/2/m_eff/dy/dy*u0
    except:
      pass
    LLRHS.append(v)
  
  # generate LL matrix
  row, col, data = list(zip(*LL_elements))
  LL = sparse.csr_matrix((data, (row, col)))
  # solve Ec
  u = sparse.linalg.spsolve(LL, LLRHS)
  w = np.reciprocal(u) - shift_factor*q*np.ones(len(u))

  return w


def DDSolver(pos, extended_pos, dx, dy, extended_Ec0, an_half, mu_n, T, Nc, V1, V2, contact1_bound, contact2_bound, insul_bound):
  DD_elements = []
  DDRHS = []

  for ij in pos:
    i, j = ij
    
    # LHS
    #---------------------------
    # fill diagonal

    # BC: Jx=0 => d(slotboom)/dx = 0
    cond_x1 = ((i+1, j) not in insul_bound)
    cond_x2 = ((i-1, j) not in insul_bound)
    cond_y1 = ((i, j+1) not in insul_bound)
    cond_y2 = ((i, j-1) not in insul_bound)

    v = ( (-an_half(i+0.51, j, extended_Ec0, extended_pos, mu_n, T, Nc)/dx/dx
           -an_half(i-0.49, j, extended_Ec0, extended_pos, mu_n, T, Nc)/dx/dx) * (cond_x1 and cond_x2)
         +(-an_half(i, j+0.51, extended_Ec0, extended_pos, mu_n, T, Nc)/dy/dy   
           -an_half(i, j-0.49, extended_Ec0, extended_pos, mu_n, T, Nc)/dy/dy) * (cond_y1 and cond_y2) )
    DD_elements.append((pos[(i, j)][0], pos[(i, j)][0], v))

    
    # fill off-diag
    # insul bound: laplace(Ec0) = Q = 0
    try:
      v = an_half(i+0.51, j, extended_Ec0, extended_pos, mu_n, T, Nc)/dx/dx * ( cond_x1 and cond_x2 )
      DD_elements.append((pos[(i, j)][0], pos[(i+1, j)][0], v))
    except:
      pass
    try:
      v = an_half(i-0.49, j, extended_Ec0, extended_pos, mu_n, T, Nc)/dx/dx * ( cond_x1 and cond_x2 )
      DD_elements.append((pos[(i, j)][0], pos[(i-1, j)][0], v))
    except:
      pass
    try:
      v = an_half(i, j+0.51, extended_Ec0, extended_pos, mu_n, T, Nc)/dy/dy * ( cond_y1 and cond_y2 )
      DD_elements.append((pos[(i, j)][0], pos[(i, j+1)][0], v))
    except:
      pass
    try:
      v = an_half(i, j-0.49, extended_Ec0, extended_pos, mu_n, T, Nc)/dy/dy * ( cond_y1 and cond_y2 )
      DD_elements.append((pos[(i, j)][0], pos[(i, j-1)][0], v))
    except:
      pass
    
    # fill RHS
    # BC: metal contact at left & right
    v=0
    v =( -an_half(i+0.51, j, extended_Ec0, extended_pos, mu_n, T, Nc)*np.exp(-q*V1/kB/T)/dx/dx * ((i+1, j) in contact1_bound)
         -an_half(i-0.49, j, extended_Ec0, extended_pos, mu_n, T, Nc)*np.exp(-q*V1/kB/T)/dx/dx * ((i-1, j) in contact1_bound)
         -an_half(i, j+0.51, extended_Ec0, extended_pos, mu_n, T, Nc)*np.exp(-q*V1/kB/T)/dy/dy * ((i, j+1) in contact1_bound)
         -an_half(i, j-0.49, extended_Ec0, extended_pos, mu_n, T, Nc)*np.exp(-q*V1/kB/T)/dy/dy * ((i, j-1) in contact1_bound)
         -an_half(i+0.51, j, extended_Ec0, extended_pos, mu_n, T, Nc)*np.exp(-q*V2/kB/T)/dx/dx * ((i+1, j) in contact2_bound)
         -an_half(i-0.49, j, extended_Ec0, extended_pos, mu_n, T, Nc)*np.exp(-q*V2/kB/T)/dx/dx * ((i-1, j) in contact2_bound)
         -an_half(i, j+0.51, extended_Ec0, extended_pos, mu_n, T, Nc)*np.exp(-q*V2/kB/T)/dy/dy * ((i, j+1) in contact2_bound)
         -an_half(i, j-0.49, extended_Ec0, extended_pos, mu_n, T, Nc)*np.exp(-q*V2/kB/T)/dy/dy * ((i, j-1) in contact2_bound) )
    DDRHS.append(v)
  
  # generate sparse matrix
  row, col, data = list(zip(*DD_elements))
  DD = sparse.csr_matrix((data, (row, col)))

  # solve Slotboom
  return sparse.linalg.spsolve(DD, DDRHS)


def PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn, Ec0, Efp, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=q, tolerance=0.01*q, max_tries=1000, calc_LL=True, LL_shift_factor_n=5, LL_shift_factor_p=7):
  Ev0 = get_Ev(pos, Ec0, Egs)
  error = init_error
  tries = 0
  while error > tolerance and tries < max_tries:
    tries += 1

    #------------------------------------------------------------#
    # Solve Poisson
    #------------------------------------------------------------#
    dEc = PoissonSolver(pos, dx, dy, eps, Xis, Nds, Nc, Nv, T, Efn, Ec0, Efp, Ev0, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, True, True)

    # update Ec0
    error = max(abs(dEc))
    Ec0 += dEc

    #------------------------------------------------------------#
    # data management
    #------------------------------------------------------------#
    # setup Ev
    Ev0 = get_Ev(pos, Ec0, Egs)

    # include boundary/shadow points into Ec0
    shadow_Ec0, shadow_pos = calc_shadow_Ec0(pos, shadow, eps, Ec0, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound)
    extended_pos = {**pos, **shadow_pos}
    extended_Ec0 = np.concatenate((Ec0, shadow_Ec0))
    extended_Ev0 = get_Ev(extended_pos, extended_Ec0, Egs)
    
    #------------------------------------------------------------#
    # construct LL solver for Ec & Ev
    #------------------------------------------------------------#
    if calc_LL == True:
      print('LL turned on')
      w_n = LLSolver(pos, shadow_pos, dx, dy, m_eff, extended_Ec0, shift_factor = LL_shift_factor_n)
      w_p = LLSolver(pos, shadow_pos, dx, dy, m_hff, extended_Ev0, shift_factor = LL_shift_factor_p)
      # update extended Ec0 & Ev0 with effective potential
      for i in range(len(w_n)):
        extended_Ec0[i] = w_n[i]
        extended_Ev0[i] = w_p[i]
    else:
      w_n = 'None'
      w_p = 'None'
    #------------------------------------------------------------#
    # construct DD eq for electron (use Slotboom)
    #------------------------------------------------------------#

    slotboom_n = DDSolver(pos, extended_pos, dx, dy, extended_Ec0, an_half, mu_n, T, Nc, V1, V2, contact1_bound, contact2_bound, insul_bound)

    # if slotboom have negative value, reset error to restart the loop
    if any(i<0 for i in slotboom_n):    
      error = q
      slotboom_n = np.abs(slotboom_n)
    Efn = kB*T*np.log(slotboom_n)
    try:
      n = Nc*np.exp((Efn-w_n)/kB/T)
    except:
      n = Nc*np.exp((Efn-Ec0)/kB/T)  
    
    #------------------------------------------------------------#
    # construct DD eq for hole (use Slotboom)
    #------------------------------------------------------------#

    slotboom_p = DDSolver(pos, extended_pos, dx, dy, extended_Ev0, ap_half, mu_p, T, Nv, -V1, -V2, contact1_bound, contact2_bound, insul_bound)
    
    # if slotboom have negative value, reset error to restart the loop
    if any(i<0 for i in slotboom_p):    
      error = q
      slotboom_p = np.abs(slotboom_p)
    Efp = -kB*T*np.log(slotboom_p)
    try:
      p = Nv*np.exp((w_p-Efp)/kB/T)
    except:
      p = Nv*np.exp((Ev0-Efp)/kB/T)

  return Ec0, Ev0, w_n, w_p, Efn, Efp, n, p, slotboom_n, slotboom_p, tries

def get_current_density(pos, Ec0, Ev0, mu_n, mu_p, T, Nc, Nv, slotboom_n, slotboom_p):
  # calc current density
  Jn=[]
  Jp=[]
  Jtotal=[]
  for ij in pos:
    i, j = ij
    try:   
      Jnx = an_half(i+0.51, j, Ec0, pos, mu_n, T, Nc)/dx*(slotboom_n[pos[(i+1,j)][0]]-slotboom_n[pos[(i,j)][0]])
      Jny = an_half(i, j+0.51, Ec0, pos, mu_n, T, Nc)/dy*(slotboom_n[pos[(i,j+1)][0]]-slotboom_n[pos[(i,j)][0]])
      Jn.append((i+0.51, j+0.51, Jnx, Jny))

      Jpx = ap_half(i+0.51, j, Ev0, pos, mu_p, T, Nv)/dx*(slotboom_p[pos[(i+1,j)][0]]-slotboom_p[pos[(i,j)][0]])
      Jpy = ap_half(i, j+0.51, Ev0, pos, mu_p, T, Nv)/dy*(slotboom_p[pos[(i,j+1)][0]]-slotboom_p[pos[(i,j)][0]])
      Jp.append((i+0.51, j+0.51, Jpx, Jpy))

      Jtotal.append((i+0.51, j+0.51, Jpx+Jnx, Jpy+Jny))
    except:
      pass

  return Jn, Jp, Jtotal

#=================================================#
#================= program test ======================#
#=================================================#

# 1
#------------------------------------------------------------#
# construct model
#------------------------------------------------------------#
dx = 1.0e-10        # m
dy = 1.0e-10        # m
l = 8.0e-10          # m
w = 10.0e-10          # m

def mesh_data(i,j):
  return 'space holder'

# rectangle shape
pos = {}
nl = int(l/dx)+1
nw = int(w/dx)+1
for i in range(nl):
  for j in range(nw):
    pos[(i,j)] = 'space holder'

# extrude shape
for i in range(0, 2):
  for j in range(-3, 0):
    pos[(i, j)] = 'space holder'

# extrude shape
for i in range(7, 9):
  for j in range(-3, 0):
    pos[(i, j)] = 'space holder'

# extrude shape
for i in range(-2, 0):
  for j in range(2, 8):
    pos[(i, j)] = 'space holder'

# remove rectangle
for ij in list(pos.keys()):
  i, j = ij
  if i>1 and i<8 and j>4 and j<10:
    del pos[ij]

# assign index
for ind, ij in enumerate(pos):
  pos[ij] = (ind, mesh_data(ij, ij))

# find nearest neighbors
neighbor_pair = []
shadow = {}
for ij in pos:
  i, j = ij
  for near in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
    if near not in pos:
      shadow[near] = 'boundary'


contact1_bound = shadow
contact2_bound = []
insul_bound = [ij for ij in shadow if ((ij not in contact1_bound) and (ij not in contact2_bound))]

#------------------------------------------------------------#
# plot model
#------------------------------------------------------------#
fig, ax = plt.subplots(figsize=(8, 8))

# plot neighbor
line_segments = LineCollection(neighbor_pair)
ax.add_collection(line_segments)

# plot mesh points
x, y = list(zip(*pos)) 
ax.scatter(x, y, c='r')

# plot shadow points
x, y = list(zip(*shadow)) 
ax.scatter(x, y, c='k')

# write indeces
for i in pos:
  ax.text(i[0]+0.1, i[1]+0.1, pos[i][0], c='r')

plt.axis('scaled')
plt.show()


# 2
#------------------------------------------------------------#
# construct model: tank
#------------------------------------------------------------#
dx = 1.0e-10        # m
dy = 1.0e-10        # m
l = 4.0e-9        # m
w = 1.8e-9        # m

def mesh_data(i,j):
  return 'space holder'

# rectangle shape
pos = {}
nl = int(l/dx)+1
nw = int(w/dx)+1
for i in range(nl):
  for j in range(nw):
    pos[(i,j)] = 'space holder'

# extrude shape
for i in range(10, 30):
  for j in range(18, 30):
    pos[(i, j)] = 'space holder'

# extrude shape
for i in range(30, 37):
  for j in range(22, 28):
    pos[(i, j)] = 'space holder'

# assign index
for ind, ij in enumerate(pos):
  pos[ij] = (ind, mesh_data(ij, ij))

# find nearest neighbors
neighbor_pair = []
shadow = {}
for ij in pos:
  i, j = ij
  for near in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
    if near not in pos:
      shadow[near] = 'boundary'


contact1_bound = shadow
contact2_bound = []
insul_bound = [ij for ij in shadow if ((ij not in contact1_bound) and (ij not in contact2_bound))]

#------------------------------------------------------------#
# plot model
#------------------------------------------------------------#
fig, ax = plt.subplots(figsize=(8, 8))

# plot neighbor
#line_segments = LineCollection(neighbor_pair)
#ax.add_collection(line_segments)

# plot mesh points
x, y = list(zip(*pos)) 
ax.scatter(x, y, c='r')

# plot shadow points
x, y = list(zip(*shadow)) 
ax.scatter(x, y, c='k')

# write indeces
#for i in pos:
#  ax.text(i[0]+0.1, i[1]+0.1, pos[i][0])

plt.axis('scaled')
plt.show()

T = 300            # K
ep = 11.7          # relative
Xi1 = 0          # eV
Xi2 = 0          # eV
Nd = 5.0e+23       # 1/m3
Nc = 1.0e+10       # 1/m3
Nv = 0       # 1/m3
Eg1 = 0          # eV
Eg2 = 0          # eV
phiB1 = 0        # eV
phiB2 = 0        # eV
V1 = 0          # eV  
V2 = 0          # eV

def eps(i, j):
  return ep

def Xis(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return Xi1*(not cond) + Xi2*(cond)

def Egs(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return q*Eg1*(not cond) + q*Eg2*(cond)

def Nds(i, j):
  return Nd*np.exp(-((i-6)**2+(j-6)**2)/2**2) + Nd*np.exp(-((i-20)**2+(j-5)**2)/2**2) + Nd*np.exp(-((i-34)**2+(j-6)**2)/2**2)

Efn = np.zeros(len(pos))
Efp = np.zeros(len(pos))
Ec0 = q*np.ones(len(pos))
Ev0 = np.zeros(len(pos))
error = q
tries = 0
while error > 0.001*q:
  tries += 1
  dEc = PoissonSolver(pos, dx, dy, eps, Xis, Nds, Nc, Nv, T, Efn, Ec0, Efp, Ev0, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, solve_electron=True, solve_hole=False)
  Ec0 += dEc
  error = max(abs(dEc))

x, y = list(zip(*pos)) 

print(tries)
plt.scatter(x, y, c=Ec0/q, s=30)
plt.colorbar()
plt.title('Ec0')
plt.axis('scaled')
plt.show()



# 3
#------------------------------------------------------------#
# construct model
#------------------------------------------------------------#
dx = 1.0e-10        # m
dy = 1.0e-10        # m
l = 5.0e-9        # m
w = 3.0e-9        # m

def mesh_data(i,j):
  return 'space holder'

# rectangle shape
pos = {}
nl = int(l/dx)+1
nw = int(w/dx)+1
for i in range(nl):
  for j in range(nw):
    pos[(i,j)] = 'space holder'

# assign index
for ind, ij in enumerate(pos):
  pos[ij] = (ind, mesh_data(ij, ij))

# find nearest neighbors
neighbor_pair = []
shadow = {}
for ij in pos:
  i, j = ij
  for near in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
    if near not in pos:
      shadow[near] = 'boundary'

contact1_bound = [ij for ij in shadow if ij[0] == -1]
contact2_bound = [ij for ij in shadow if ij[0] == nl]
insul_bound = [ij for ij in shadow if ((ij not in contact1_bound) and (ij not in contact2_bound))]

#------------------------------------------------------------#
# plot model
#------------------------------------------------------------#
fig, ax = plt.subplots(figsize=(8, 8))

# plot neighbor
#line_segments = LineCollection(neighbor_pair)
#ax.add_collection(line_segments)

# plot mesh points
x, y = list(zip(*pos)) 
ax.scatter(x, y, c='r')

# plot shadow points
x, y = list(zip(*shadow)) 
ax.scatter(x, y, c='k')

# write indeces
#for i in pos:
#  ax.text(i[0]+0.1, i[1]+0.1, pos[i][0])

plt.axis('scaled')
plt.show()


# 3-1
#------------------------------------------------------------#

T = 300            # K
ep = 11.7          # relative
Xi1 = 3.5          # eV
Xi2 = 3.5          # eV
Nd = 5.0e+24       # 1/m3
Nc = 1.0e+22       # 1/m3
Nv = 1.0e+22       # 1/m3
mu_n = 1000*0.0001 # m2/Vs
mu_p = 500*0.0001  # m2/Vs
m_eff = 1.08*me    
m_hff = 1.15*me    
Eg1 = 3          # eV
Eg2 = 3          # eV
phiB1 = 1.5        # eV
phiB2 = 1.5        # eV
V1 = 1          # eV  
V2 = 0          # eV

def eps(i, j):
  return ep

def Xis(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return Xi1

def Egs(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return q*Eg1

def Nds(i, j):
  cond = (i*dx>=0.2*l and i*dx<=0.7*l)
  return Nd*(cond)-Nd*(not cond)

# initialize
Efn = np.zeros(len(pos))
Efp = np.zeros(len(pos))
Ec0 = q*np.ones(len(pos))
Ev0 = get_Ev(pos, Ec0, Egs)
error = q
tolerance = q*0.01 
tries = 0

Ec0, Ev0, w_n, w_p, Efn, Efp, n, p, slotboom_n, slotboom_p, tries = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn, Ec0, Efp, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = True, LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
Jn, Jp, Jtotal = get_current_density(pos, Ec0, Ev0, mu_n, mu_p, T, Nc, Nv, slotboom_n, slotboom_p)

#------------------------------------------------------------#
# plot result 
#------------------------------------------------------------#
print('total tries =', tries)
x, y = list(zip(*pos)) 

x_slice = []
Ec_slice = []
Ev_slice = []
eff_pot_n_slice = []
eff_pot_p_slice = []
Efn_slice = []
Efp_slice = []
slice_y = 15
for i in range(len(x)):
  if y[i] == slice_y:
    x_slice.append(x[i])
    Ec_slice.append(Ec0[i]/q)
    Ev_slice.append(Ev0[i]/q)
    Efn_slice.append(Efn[i]/q)
    Efp_slice.append(Efp[i]/q)
    eff_pot_n_slice.append(w_n[i]/q)
    eff_pot_p_slice.append(w_p[i]/q)
plt.plot(x_slice, Ec_slice, label='Ec')
plt.plot(x_slice, Ev_slice, label='Ev')
plt.plot(x_slice, Efn_slice, label='Efn')
plt.plot(x_slice, Efp_slice, label='Efp')
plt.plot(x_slice, eff_pot_n_slice, label='eff e- pot')
plt.plot(x_slice, eff_pot_p_slice, label='eff h+ pot')
plt.legend()
plt.title('E along y= ' + str(slice_y))
plt.ylabel('E (eV)')
plt.show()

plt.scatter(x, y, c=n, s=30)
plt.colorbar()
plt.title('n')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=p, s=30)
plt.colorbar()
plt.title('p')
plt.axis('scaled')
plt.show()

x_slice = []
n_slice = []
p_slice = []
slice_y = 15
for i in range(len(x)):
  if y[i] == slice_y:
    x_slice.append(x[i])
    n_slice.append(n[i])
    p_slice.append(p[i])
plt.plot(x_slice, n_slice, label='n')
plt.plot(x_slice, p_slice, label='p')
plt.legend()
plt.title('n/p along y= ' + str(slice_y))
plt.ylabel('1/m3')
plt.show()
print()


# 3-2
#------------------------------------------------------------#
T = 300            # K
ep = 11.7          # relative
Xi1 = 3.5          # eV
Xi2 = 4.5          # eV
Nd = 5.0e+23       # 1/m3
Nc = 1.0e+24       # 1/m3
Nv = 1.0e+24       # 1/m3
mu_n = 1000*0.0001 # m2/Vs
mu_p = 500*0.0001  # m2/Vs
m_eff = 1.08*me    
m_hff = 1.15*me    
Eg1 = 3.0          # eV
Eg2 = 1.5          # eV
phiB1 = 1.5        # eV
phiB2 = 1.5        # eV
V1 = 1          # eV  
V2 = 0          # eV

def eps(i, j):
  return ep

def Xis(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return Xi1*(not cond) + Xi2*(cond)

def Egs(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return q*Eg1*(not cond) + q*Eg2*(cond)

def Nds(i, j):
  return Nd

# initialize
Efn = np.zeros(len(pos))
Efp = np.zeros(len(pos))
Ec0 = q*np.ones(len(pos))
Ev0 = get_Ev(pos, Ec0, Egs)
error = q
tolerance = q*0.01 
tries = 0

Ec0, Ev0, w_n, w_p, Efn, Efp, n, p, slotboom_n, slotboom_p, tries = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn, Ec0, Efp, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = True, LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
Jn, Jp, Jtotal = get_current_density(pos, Ec0, Ev0, mu_n, mu_p, T, Nc, Nv, slotboom_n, slotboom_p)

#------------------------------------------------------------#
# plot result 
#------------------------------------------------------------#
print('total tries =', tries)
x, y = list(zip(*pos)) 

plt.scatter(x, y, c=Ec0/q, s=30)
plt.colorbar()
plt.title('Ec0')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=Ev0/q, s=30)
plt.colorbar()
plt.title('Ev0')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=w_n/q, s=30)
plt.colorbar()
plt.title('effective electron pot from LL')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=w_p/q, s=30)
plt.colorbar()
plt.title('effective hole pot from LL')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=Efn/q, s=30)
plt.colorbar()
plt.title('Efn')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=Efp/q, s=30)
plt.colorbar()
plt.title('Efp')
plt.axis('scaled')
plt.show()

x_slice = []
Ec_slice = []
Ev_slice = []
eff_pot_n_slice = []
eff_pot_p_slice = []
Efn_slice = []
Efp_slice = []
slice_y = 15
for i in range(len(x)):
  if y[i] == slice_y:
    x_slice.append(x[i])
    Ec_slice.append(Ec0[i]/q)
    Ev_slice.append(Ev0[i]/q)
    Efn_slice.append(Efn[i]/q)
    Efp_slice.append(Efp[i]/q)
    eff_pot_n_slice.append(w_n[i]/q)
    eff_pot_p_slice.append(w_p[i]/q)
plt.plot(x_slice, Ec_slice, label='Ec')
plt.plot(x_slice, Ev_slice, label='Ev')
plt.plot(x_slice, Efn_slice, label='Efn')
plt.plot(x_slice, Efp_slice, label='Efp')
plt.plot(x_slice, eff_pot_n_slice, label='eff e- pot')
plt.plot(x_slice, eff_pot_p_slice, label='eff h+ pot')
plt.legend()
plt.title('E along y= ' + str(slice_y))
plt.ylabel('E (eV)')
plt.show()

fig, ax = plt.subplots(figsize=(24, 24))
xj, yj, Jtotalx, Jtotaly = list(zip(*Jtotal))
_, _,  Jnx, Jny = list(zip(*Jn))
_, _,  Jpx, Jpy = list(zip(*Jp))
max_Jtotal = max(np.power(np.power(Jtotalx, 2)+np.power(Jtotaly, 2), 0.5))
plt.axis('scaled')
plt.xlim(-1, nl+0)
plt.ylim(-1, nw+0)
for i in range(len(xj)):
  J_len = np.power(Jtotalx[i]**2+Jtotaly[i]**2, 0.5)
  plt.arrow(xj[i], yj[i], Jtotalx[i]/J_len, Jtotaly[i]/J_len, width=0.1, length_includes_head=True, head_width=0.6, head_length=0.5, facecolor='k', edgecolor='k')
  J_len = np.power(Jnx[i]**2+Jny[i]**2, 0.5)
  plt.arrow(xj[i], yj[i], Jnx[i]/J_len, Jny[i]/J_len, width=0.1, length_includes_head=True, head_width=0.6, head_length=0.5, facecolor='none', edgecolor='r')
  J_len = np.power(Jpx[i]**2+Jpy[i]**2, 0.5)
  plt.arrow(xj[i], yj[i], Jpx[i]/J_len, Jpy[i]/J_len, width=0.1, length_includes_head=True, head_width=0.6, head_length=0.5, facecolor='none', edgecolor='b')
plt.title('current density (longest arrow = ' +str(max_Jtotal)+')')
plt.show()

plt.scatter(x, y, c=n, s=30)
plt.colorbar()
plt.title('n')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=p, s=30)
plt.colorbar()
plt.title('p')
plt.axis('scaled')
plt.show()

x_slice = []
n_slice = []
p_slice = []
slice_y = 15
for i in range(len(x)):
  if y[i] == slice_y:
    x_slice.append(x[i])
    n_slice.append(n[i])
    p_slice.append(p[i])
plt.plot(x_slice, n_slice, label='n')
plt.plot(x_slice, p_slice, label='p')
plt.legend()
plt.title('n/p along y= ' + str(slice_y))
plt.ylabel('1/m3')
plt.show()
print()


# 3-3-1
#------------------------------------------------------------#
T = 300            # K
ep = 11.7          # relative
Xi1 = 4.0         # eV
Xi2 = 4.0          # eV
Nd = 5.0e+23       # 1/m3
Nc = 1.0e+24       # 1/m3
Nv = 1.0e+24       # 1/m3
mu_n = 1000*0.0001 # m2/Vs
mu_p = 500*0.0001  # m2/Vs
m_eff = 1.08*me    
m_hff = 1.15*me    
Eg1 = 1          # eV
Eg2 = 1          # eV
phiB1 = 0.75        # eV
phiB2 = 0       # eV
V1 = 0          # eV  
V2 = 0          # eV

def eps(i, j):
  return ep

def Xis(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return Xi1*(not cond) + Xi2*(cond)

def Egs(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return q*Eg1*(not cond) + q*Eg2*(cond)

def Nds(i, j):
  return Nd

# initialize
Efn = np.zeros(len(pos))
Efp = np.zeros(len(pos))
Ec0 = q*np.ones(len(pos))
Ev0 = get_Ev(pos, Ec0, Egs)
error = q
tolerance = q*0.01 
tries = 0

Ec0, Ev0, w_n, w_p, Efn, Efp, n, p, slotboom_n, slotboom_p, tries = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn, Ec0, Efp, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = True, LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
Jn, Jp, Jtotal = get_current_density(pos, Ec0, Ev0, mu_n, mu_p, T, Nc, Nv, slotboom_n, slotboom_p)

#------------------------------------------------------------#
# plot result
#------------------------------------------------------------#
print('total tries =', tries)
x, y = list(zip(*pos)) 



x_slice = []
Ec_slice = []
Ev_slice = []
eff_pot_n_slice = []
eff_pot_p_slice = []
Efn_slice = []
Efp_slice = []
slice_y = 15
for i in range(len(x)):
  if y[i] == slice_y:
    x_slice.append(x[i])
    Ec_slice.append(Ec0[i]/q)
    Ev_slice.append(Ev0[i]/q)
    Efn_slice.append(Efn[i]/q)
    Efp_slice.append(Efp[i]/q)
plt.plot(x_slice, Ec_slice, label='Ec')
plt.plot(x_slice, Ev_slice, label='Ev')
plt.plot(x_slice, Efn_slice, label='Efn')
plt.plot(x_slice, Efp_slice, label='Efp')
plt.legend()
plt.title('E (phiB1=0.75eV) along y= ' + str(slice_y))
plt.ylabel('E (eV)')
plt.show()

phiB1 = 1.0
phiB2 = 0
def Xis(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return 4.0
def Egs(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return q*1

Js = []
V1s = np.linspace(-1, 0.5, 11)
for v in V1s:
  Efn1 = np.zeros(len(pos))
  Efp1 = np.zeros(len(pos))
  Ec01 = q*np.ones(len(pos))

  Ec01, Ev01, w_n1, w_p1, Efn1, Efp1, n1, p1, slotboom_n1, slotboom_p1, tries1 = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn1, Ec01, Efp1, v, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = True, LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
  Jn1, Jp1, Jtotal1 = get_current_density(pos, Ec01, Ev01, mu_n, mu_p, T, Nc, Nv, slotboom_n1, slotboom_p1)

  J_left = 0
  J_right = 0
  for jt in Jtotal1:
    if jt[0] == 0.51:
      J_left += np.sqrt(jt[2]**2+jt[3]**2)
    elif jt[0] == 49.51:
      J_right += np.sqrt(jt[2]**2+jt[3]**2)
  Js.append(J_left)

V2s = np.linspace(0.5, 1, 21)
for v in V2s:
  Efn1 = np.zeros(len(pos))
  Efp1 = np.zeros(len(pos))
  Ec01 = q*np.ones(len(pos))

  Ec01, Ev01, w_n1, w_p1, Efn1, Efp1, n1, p1, slotboom_n1, slotboom_p1, tries1 = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn1, Ec01, Efp1, v, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = True, LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
  Jn1, Jp1, Jtotal1 = get_current_density(pos, Ec01, Ev01, mu_n, mu_p, T, Nc, Nv, slotboom_n1, slotboom_p1)

  J_left = 0
  J_right = 0
  for jt in Jtotal1:
    if jt[0] == 0.51:
      J_left += np.sqrt(jt[2]**2+jt[3]**2)
    elif jt[0] == 49.51:
      J_right += np.sqrt(jt[2]**2+jt[3]**2)
  Js.append(J_left)

plt.plot(np.concatenate((V1s,V2s)), Js)  # magnitude of current
plt.show()


# 3-3-2
#------------------------------------------------------------#

phiB1 = 0.5
phiB2 = 0
def Xis(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return 4.0 #Xi1*(not cond) + Xi2*(cond)
def Egs(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return q*1 #q*Eg1*(not cond) + q*Eg2*(cond)
  
Js2 = []
V1s = np.linspace(-1, 0, 11)
for v in V1s:
  Efn1 = np.zeros(len(pos))
  Efp1 = np.zeros(len(pos))
  Ec01 = q*np.ones(len(pos))

  Ec01, Ev01, w_n1, w_p1, Efn1, Efp1, n1, p1, slotboom_n1, slotboom_p1, tries1 = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn1, Ec01, Efp1, v, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = True, LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
  Jn1, Jp1, Jtotal1 = get_current_density(pos, Ec01, Ev01, mu_n, mu_p, T, Nc, Nv, slotboom_n1, slotboom_p1)

  J_left = 0
  J_right = 0
  for jt in Jtotal1:
    if jt[0] == 0.51:
      J_left += np.sqrt(jt[2]**2+jt[3]**2)
    elif jt[0] == 49.51:
      J_right += np.sqrt(jt[2]**2+jt[3]**2)
  Js2.append(J_left)

V2s = np.linspace(0, 1, 31)

for v in V2s:
  Efn1 = np.zeros(len(pos))
  Efp1 = np.zeros(len(pos))
  Ec01 = q*np.ones(len(pos))

  Ec01, Ev01, w_n1, w_p1, Efn1, Efp1, n1, p1, slotboom_n1, slotboom_p1, tries1 = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn1, Ec01, Efp1, v, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = True, LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
  Jn1, Jp1, Jtotal1 = get_current_density(pos, Ec01, Ev01, mu_n, mu_p, T, Nc, Nv, slotboom_n1, slotboom_p1)


  J_left = 0
  J_right = 0
  for jt in Jtotal1:
    if jt[0] == 0.51:
      J_left += np.sqrt(jt[2]**2+jt[3]**2)
    elif jt[0] == 49.51:
      J_right += np.sqrt(jt[2]**2+jt[3]**2)
  Js2.append(J_left)
plt.plot(np.concatenate((V1s,V2s)), Js2)  # magnitude of current
plt.show()

phiB1 = 0.75
phiB2 = 0
def Xis(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return 4.0 #Xi1*(not cond) + Xi2*(cond)
def Egs(i, j):
  cond = ((i*dx>=0.3*l and i*dx<0.6*l) and (j*dy<=0.6*w and j*dy>0.3*w))
  return q*1 #q*Eg1*(not cond) + q*Eg2*(cond)
  
Js3 = []
V1s = np.linspace(-1, 0, 11)
for v in V1s:
  Efn1 = np.zeros(len(pos))
  Efp1 = np.zeros(len(pos))
  Ec01 = q*np.ones(len(pos))

  Ec01, Ev01, w_n1, w_p1, Efn1, Efp1, n1, p1, slotboom_n1, slotboom_p1, tries1 = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn1, Ec01, Efp1, v, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = True, LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
  Jn1, Jp1, Jtotal1 = get_current_density(pos, Ec01, Ev01, mu_n, mu_p, T, Nc, Nv, slotboom_n1, slotboom_p1)

  J_left = 0
  J_right = 0
  for jt in Jtotal1:
    if jt[0] == 0.51:
      J_left += np.sqrt(jt[2]**2+jt[3]**2)
    elif jt[0] == 49.51:
      J_right += np.sqrt(jt[2]**2+jt[3]**2)
  Js3.append(J_left)

V2s = np.linspace(0, 1, 31)

for v in V2s:
  Efn1 = np.zeros(len(pos))
  Efp1 = np.zeros(len(pos))
  Ec01 = q*np.ones(len(pos))

  Ec01, Ev01, w_n1, w_p1, Efn1, Efp1, n1, p1, slotboom_n1, slotboom_p1, tries1 = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn1, Ec01, Efp1, v, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = True, LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
  Jn1, Jp1, Jtotal1 = get_current_density(pos, Ec01, Ev01, mu_n, mu_p, T, Nc, Nv, slotboom_n1, slotboom_p1)


  J_left = 0
  J_right = 0
  for jt in Jtotal1:
    if jt[0] == 0.51:
      J_left += np.sqrt(jt[2]**2+jt[3]**2)
    elif jt[0] == 49.51:
      J_right += np.sqrt(jt[2]**2+jt[3]**2)
  Js3.append(J_left)
plt.plot(np.concatenate((V1s,V2s)), Js3)  # magnitude of current
plt.show()

V1s = np.linspace(-1, 0.5, 11)      
V2s = np.linspace(0.5, 1, 21)
plt.plot(np.concatenate((V1s,V2s)), Js, label='phiB1=1.0eV')
V1s = np.linspace(-1, 0, 11)
V2s = np.linspace(0, 1, 31)
plt.plot(np.concatenate((V1s,V2s)), Js3, label='phiB1=0.75eV')
V1s = np.linspace(-1, 0, 11)
V2s = np.linspace(0, 1, 31)
plt.title('IV')
plt.plot(np.concatenate((V1s,V2s)), Js2, label='phiB1=0.5eV')
plt.legend()
plt.show()


# 3-4
#------------------------------------------------------------#

T = 300            # K
ep = 11.7          # relative
Xi1 = 3.5          # eV
Xi2 = 4.5          # eV
Nd = 5.0e+23       # 1/m3
Nc = 1.0e+24       # 1/m3
Nv = 1.0e+24       # 1/m3
mu_n = 1000*0.0001 # m2/Vs
mu_p = 500*0.0001  # m2/Vs
m_eff = 1.08*me    
m_hff = 1.15*me    
Eg1 = 3.0          # eV
Eg2 = 1.2          # eV
phiB1 = 1.5        # eV
phiB2 = 1.5        # eV
V1 = 0.5          # eV  
V2 = 0          # eV

def eps(i, j):
  return ep

def Xis(i, j):
  cond = ((i>=12 and i<15) or (i>=17 and i<20) or (i>=22 and i<25) or (i>=27 and i<30) or (i>=32 and i<35) or (i>=37 and i<40)) 
  return Xi1*(not cond) + Xi2*(cond)

def Egs(i, j):
  cond = ((i>=12 and i<15) or (i>=17 and i<20) or (i>=22 and i<25) or (i>=27 and i<30) or (i>=32 and i<35) or (i>=37 and i<40)) 
  return q*Eg1*(not cond) + q*Eg2*(cond)

def Nds(i, j):
  return Nd

# initialize
Efn = np.zeros(len(pos))
Efp = np.zeros(len(pos))
Ec0 = q*np.ones(len(pos))
Ev0 = get_Ev(pos, Ec0, Egs)
error = q
tolerance = q*0.0001 
tries = 0
calc_LL = True
Ec0, Ev0, w_n, w_p, Efn, Efp, n, p, slotboom_n, slotboom_p, tries = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn, Ec0, Efp, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = calc_LL,  LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
Jn, Jp, Jtotal = get_current_density(pos, Ec0, Ev0, mu_n, mu_p, T, Nc, Nv, slotboom_n, slotboom_p)
#------------------------------------------------------------#
# plot result 
#------------------------------------------------------------#
print('total tries =', tries)
x, y = list(zip(*pos)) 

plt.scatter(x, y, c=Ec0/q, s=30)
plt.colorbar()
plt.title('Ec0')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=Ev0/q, s=30)
plt.colorbar()
plt.title('Ev0')
plt.axis('scaled')
plt.show()

'''plt.scatter(x, y, c=w_n/q, s=30)
plt.colorbar()
plt.title('effective electron pot from LL')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=w_p/q, s=30)
plt.colorbar()
plt.title('effective hole pot from LL')
plt.axis('scaled')
plt.show()'''

plt.scatter(x, y, c=Efn/q, s=30)
plt.colorbar()
plt.title('Efn')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=Efp/q, s=30)
plt.colorbar()
plt.title('Efp')
plt.axis('scaled')
plt.show()

x_slice = []
Ec_slice = []
Ev_slice = []
eff_pot_n_slice = []
eff_pot_p_slice = []
Efn_slice = []
Efp_slice = []
slice_y = 15
for i in range(len(x)):
  if y[i] == slice_y:
    x_slice.append(x[i])
    Ec_slice.append(Ec0[i]/q)
    Ev_slice.append(Ev0[i]/q)
    Efn_slice.append(Efn[i]/q)
    Efp_slice.append(Efp[i]/q)
    try:
      eff_pot_n_slice.append(w_n[i]/q)
      eff_pot_p_slice.append(w_p[i]/q)
    except:
      pass
plt.plot(x_slice, Ec_slice, label='Ec')
plt.plot(x_slice, Ev_slice, label='Ev')
plt.plot(x_slice, Efn_slice, label='Efn')
plt.plot(x_slice, Efp_slice, label='Efp')
try:
  plt.plot(x_slice, eff_pot_n_slice, label='eff e- pot')
  plt.plot(x_slice, eff_pot_p_slice, label='eff h+ pot')
except:
  pass
plt.legend()
plt.title('E along y= ' + str(slice_y))
plt.ylabel('E (eV)')
plt.show()

plt.scatter(x, y, c=n, s=30)
plt.colorbar()
plt.title('n')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=p, s=30)
plt.colorbar()
plt.title('p')
plt.axis('scaled')
plt.show()

x_slice = []
n_slice = []
p_slice = []
slice_y = 15
for i in range(len(x)):
  if y[i] == slice_y:
    x_slice.append(x[i])
    n_slice.append(n[i])
    p_slice.append(p[i])
plt.plot(x_slice, n_slice, label='n')
plt.plot(x_slice, p_slice, label='p')
plt.legend()
plt.title('n/p along y= ' + str(slice_y))
plt.ylabel('1/m3')
plt.show()

# 3-5
#------------------------------------------------------------#
def Xis(i, j):
  cond = ((i>=12 and i<15) or (i>=17 and i<20) or (i>=22 and i<25) or (i>=27 and i<30) or (i>=32 and i<35) or (i>=37 and i<40)) 
  return Xi1*(not cond) + Xi2*(cond)

def Egs(i, j):
  cond = ((i>=12 and i<15) or (i>=17 and i<20) or (i>=22 and i<25) or (i>=27 and i<30) or (i>=32 and i<35) or (i>=37 and i<40)) 
  return q*Eg1*(not cond) + q*Eg2*(cond)

Js_yes = []
V1_yes = np.linspace(-1, 1, 13)
for v in V1_yes:
  print('yes', v)
  Efn = np.zeros(len(pos))
  Efp = np.zeros(len(pos))
  Ec0 = q*np.ones(len(pos))

  Ec0_yes, Ev0_yes, w_n_yes, w_p_yes, Efn_yes, Efp_yes, n_yes, p_yes, slotboom_n_yes, slotboom_p_yes, tries = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn, Ec0, Efp, v, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.0001*q, max_tries=100, calc_LL = True, LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
  Jn_yes, Jp_yes, Jtotal_yes = get_current_density(pos, Ec0_yes, Ev0_yes, mu_n, mu_p, T, Nc, Nv, slotboom_n_yes, slotboom_p_yes)

  J_left_yes = 0
  J_right_yes = 0
  for jt in Jtotal_yes:
    if jt[0] == 0.51:
      J_left_yes += np.sqrt(jt[2]**2+jt[3]**2)
    elif jt[0] == 49.51:
      J_right_yes += np.sqrt(jt[2]**2+jt[3]**2)
  Js_yes.append(J_left_yes)
plt.plot(V1_yes, Js_yes)  # magnitude of current

Js_no = []
V1_no = np.linspace(-1, 1, 13)
for v in V1_no:
  print('no', v)
  Efn = np.zeros(len(pos))
  Efp = np.zeros(len(pos))
  Ec0 = q*np.ones(len(pos))

  Ec0_no, Ev0_no, w_n_no, w_p_no, Efn_no, Efp_no, n_no, p_no, slotboom_n_no, slotboom_p_no, tries = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn, Ec0, Efp, v, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.0001*q, max_tries=100, calc_LL = False, LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
  Jn_no, Jp_no, Jtotal_no = get_current_density(pos, Ec0_no, Ev0_no, mu_n, mu_p, T, Nc, Nv, slotboom_n_no, slotboom_p_no)

  J_left_no = 0
  J_right_no = 0
  for jt in Jtotal_no:
    if jt[0] == 0.51:
      J_left_no += np.sqrt(jt[2]**2+jt[3]**2)
    elif jt[0] == 49.51:
      J_right_no += np.sqrt(jt[2]**2+jt[3]**2)
  Js_no.append(J_left_no)


plt.plot(V1_no, Js_no)  # magnitude of current
plt.show()

# 4
#------------------------------------------------------------#
# construct model
#------------------------------------------------------------#
dx = 1.0e-11        # m
dy = 1.0e-11        # m
l = 4.0e-10        # m
w = 4.0e-10        # m

def mesh_data(i,j):
  return 'space holder'

# rectangle shape
pos = {}
nl = int(l/dx)+1
nw = int(w/dx)+1
for i in range(nl):
  for j in range(nw):
    pos[(i,j)] = 'space holder'

# assign index
for ind, ij in enumerate(pos):
  pos[ij] = (ind, mesh_data(ij, ij))

# find nearest neighbors
neighbor_pair = []
shadow = {}
for ij in pos:
  i, j = ij
  for near in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
    if near not in pos:
      shadow[near] = 'boundary'


contact1_bound = shadow
contact2_bound = []
insul_bound = [ij for ij in shadow if ((ij not in contact1_bound) and (ij not in contact2_bound))]

#------------------------------------------------------------#
# plot model
#------------------------------------------------------------#
fig, ax = plt.subplots(figsize=(8, 8))

# plot neighbor
#line_segments = LineCollection(neighbor_pair)
#ax.add_collection(line_segments)

# plot mesh points
x, y = list(zip(*pos)) 
ax.scatter(x, y, c='r')

# plot shadow points
x, y = list(zip(*shadow)) 
ax.scatter(x, y, c='k')

# write indeces
#for i in pos:
#  ax.text(i[0]+0.1, i[1]+0.1, pos[i][0])

plt.axis('scaled')
plt.show()

# 4-1
#------------------------------------------------------------#
T = 300            # K
ep = 11.7          # relative
Xi1 = 3.5          # eV
Xi2 = 4.5          # eV
Nd = 5.0e+23       # 1/m3
Nc = 1.0e+24       # 1/m3
Nv = 1.0e+24       # 1/m3
mu_n = 1000*0.0001 # m2/Vs
mu_p = 500*0.0001  # m2/Vs
m_eff = 1.08*me    
m_hff = 1.15*me    
Eg1 = 3.0          # eV
Eg2 = 1.0          # eV
phiB1 = 1.5        # eV
phiB2 = 1.5        # eV
V1 = 0          # eV  
V2 = 0          # eV

def eps(i, j):
  return ep

def Xis(i, j):
  cond = ((i-20)**2+(j-20)**2 < 25)
  return Xi1*(not cond) + Xi2*(cond)

def Egs(i, j):
  cond = ((i-20)**2+(j-20)**2 < 25)
  return q*Eg1*(not cond) + q*Eg2*(cond)

def Nds(i, j):
  return Nd

# initialize
Efn = np.zeros(len(pos))
Efp = np.zeros(len(pos))
Ec0 = q*np.ones(len(pos))
Ev0 = get_Ev(pos, Ec0, Egs)
tries = 0
calc_LL = True
Ec0, Ev0, w_n, w_p, Efn, Efp, n, p, slotboom_n, slotboom_p, tries = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn, Ec0, Efp, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = calc_LL,  LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
Jn, Jp, Jtotal = get_current_density(pos, Ec0, Ev0, mu_n, mu_p, T, Nc, Nv, slotboom_n, slotboom_p)
#------------------------------------------------------------#
# plot result (last col will look wierd b.c. the fig is form by large filled circles)
#------------------------------------------------------------#
print('total tries =', tries)
x, y = list(zip(*pos)) 

plt.scatter(x, y, c=Ec0/q, s=30)
plt.colorbar()
plt.title('Ec0')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=Ev0/q, s=30)
plt.colorbar()
plt.title('Ev0')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=w_n/q, s=30)
plt.colorbar()
plt.title('effective electron pot from LL')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=w_p/q, s=30)
plt.colorbar()
plt.title('effective hole pot from LL')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=Efn/q, s=30)
plt.colorbar()
plt.title('Efn')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=Efp/q, s=30)
plt.colorbar()
plt.title('Efp')
plt.axis('scaled')
plt.show()

x_slice = []
Ec_slice = []
Ev_slice = []
eff_pot_n_slice = []
eff_pot_p_slice = []
Efn_slice = []
Efp_slice = []
slice_y = 20
for i in range(len(x)):
  if y[i] == slice_y:
    x_slice.append(x[i])
    Ec_slice.append(Ec0[i]/q)
    Ev_slice.append(Ev0[i]/q)
    Efn_slice.append(Efn[i]/q)
    Efp_slice.append(Efp[i]/q)
    try:
      eff_pot_n_slice.append(w_n[i]/q)
      eff_pot_p_slice.append(w_p[i]/q)
    except:
      pass
plt.plot(x_slice, Ec_slice, label='Ec')
plt.plot(x_slice, Ev_slice, label='Ev')
plt.plot(x_slice, Efn_slice, label='Efn')
plt.plot(x_slice, Efp_slice, label='Efp')
try:
  plt.plot(x_slice, eff_pot_n_slice, label='eff e- pot')
  plt.plot(x_slice, eff_pot_p_slice, label='eff h+ pot')
except:
  pass
plt.legend()
plt.title('E along y= ' + str(slice_y))
plt.ylabel('E (eV)')
plt.show()

plt.scatter(x, y, c=n, s=30)
plt.colorbar()
plt.title('n')
plt.axis('scaled')
plt.show()

plt.scatter(x, y, c=p, s=30)
plt.colorbar()
plt.title('p')
plt.axis('scaled')
plt.show()

x_slice = []
n_slice = []
p_slice = []
slice_y = 20
for i in range(len(x)):
  if y[i] == slice_y:
    x_slice.append(x[i])
    n_slice.append(np.log10(n[i]))
    p_slice.append(np.log10(p[i]))
plt.plot(x_slice, n_slice, label='n w/ LL')
plt.plot(x_slice, p_slice, label='p w/ LL')
plt.legend()
plt.title('n/p (LL) along y= ' + str(slice_y))
plt.ylabel('1/m3 log scale')

calc_LL = False
Ec01, Ev01, w_n1, w_p1, Efn1, Efp1, n1, p1, slotboom_n1, slotboom_p1, tries1 = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn, Ec0, Efp, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = calc_LL,  LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
x1_slice = []
n1_slice = []
p1_slice = []
slice_y = 20
for i in range(len(x)):
  if y[i] == slice_y:
    x1_slice.append(x[i])
    n1_slice.append(np.log10(n1[i]))
    p1_slice.append(np.log10(p1[i]))
plt.plot(x1_slice, n1_slice, label='n w/o LL')
plt.plot(x1_slice, p1_slice, label='p w/o LL')
plt.legend()
plt.title('n/p along y= ' + str(slice_y))
plt.ylabel('1/m3 log scale')
plt.show()

# 4-2
#------------------------------------------------------------#
# construct model
#------------------------------------------------------------#
dx = 1.0e-11        # m
dy = 1.0e-11        # m
l = 4.0e-10        # m
w = 4.0e-10        # m

def mesh_data(i,j):
  return 'space holder'

# rectangle shape
pos = {}
nl = int(l/dx)+1
nw = int(w/dx)+1
for i in range(nl):
  for j in range(nw):
    pos[(i,j)] = 'space holder'

# assign index
for ind, ij in enumerate(pos):
  pos[ij] = (ind, mesh_data(ij, ij))

# find nearest neighbors
neighbor_pair = []
shadow = {}
for ij in pos:
  i, j = ij
  for near in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
    if near not in pos:
      shadow[near] = 'boundary'


contact1_bound = shadow
contact2_bound = []
insul_bound = [ij for ij in shadow if ((ij not in contact1_bound) and (ij not in contact2_bound))]

T = 300            # K
ep = 11.7          # relative
Xi1 = 3.5          # eV
Xi2 = 4.5          # eV
Nd = 5.0e+23       # 1/m3
Nc = 1.0e+24       # 1/m3
Nv = 1.0e+24       # 1/m3
mu_n = 1000*0.0001 # m2/Vs
mu_p = 500*0.0001  # m2/Vs
m_eff = 1.08*me    
m_hff = 1.15*me    
Eg1 = 3.0          # eV
Eg2 = 1.0          # eV
phiB1 = 1.5        # eV
phiB2 = 1.5        # eV
V1 = 0          # eV  
V2 = 0          # eV
radius = 5

for radius in range(4, 10, 1):
  def eps(i, j):
    return ep

  def Xis(i, j):
    cond = ((i-20)**2+(j-20)**2 < radius**2)
    return Xi1*(not cond) + Xi2*(cond)

  def Egs(i, j):
    cond = ((i-20)**2+(j-20)**2 < radius**2)
    return q*Eg1*(not cond) + q*Eg2*(cond)

  def Nds(i, j):
    return Nd

  # initialize
  Efn = np.zeros(len(pos))
  Efp = np.zeros(len(pos))
  Ec0 = q*np.ones(len(pos))
  Ev0 = get_Ev(pos, Ec0, Egs)
  tries = 0
  calc_LL = True
  Ec0, Ev0, w_n, w_p, Efn, Efp, n, p, slotboom_n, slotboom_p, tries = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn, Ec0, Efp, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = calc_LL,  LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
  Jn, Jp, Jtotal = get_current_density(pos, Ec0, Ev0, mu_n, mu_p, T, Nc, Nv, slotboom_n, slotboom_p)
  #------------------------------------------------------------#
  # plot result (last col will look wierd b.c. the fig is form by large filled circles)
  #------------------------------------------------------------#
  print('total tries =', tries)
  x, y = list(zip(*pos)) 

  x_slice = []
  n_slice = []
  p_slice = []
  slice_y = 20
  for i in range(len(x)):
    if y[i] == slice_y:
      x_slice.append(x[i])
      n_slice.append(np.log10(n[i]))
      p_slice.append(np.log10(p[i]))
  plt.plot(x_slice, n_slice, label='n w/ LL')
  plt.plot(x_slice, p_slice, label='p w/ LL')

  calc_LL = False
  Ec01, Ev01, w_n1, w_p1, Efn1, Efp1, n1, p1, slotboom_n1, slotboom_p1, tries1 = PoissonLLDDSolver(pos, shadow, dx, dy, eps, Egs, Xis, Nds, Nc, Nv, mu_n, mu_p, m_eff, m_hff, T, Efn, Ec0, Efp, V1, phiB1, V2, phiB2, contact1_bound, contact2_bound, insul_bound, init_error=1.2*q, tolerance=0.001*q, max_tries=100, calc_LL = calc_LL,  LL_shift_factor_n=5.1, LL_shift_factor_p=7.1)
  x1_slice = []
  n1_slice = []
  p1_slice = []
  slice_y = 20
  for i in range(len(x)):
    if y[i] == slice_y:
      x1_slice.append(x[i])
      n1_slice.append(np.log10(n1[i]))
      p1_slice.append(np.log10(p1[i]))
  plt.plot(x1_slice, n1_slice, label='n w/o LL')
  plt.plot(x1_slice, p1_slice, label='p w/o LL')
  plt.legend()
  plt.title('n/p along y= ' + str(slice_y))
  plt.ylabel('1/m3 log scale')
  plt.show()