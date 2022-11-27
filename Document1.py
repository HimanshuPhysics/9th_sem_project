import numpy as np
from  scipy.optimize import  fsolve, minimize
from numba import njit
import plotly.graph_objects as go

def Interaction_Matrix(J_int, D_int, K_int):
    fo = open('input_matrix_JDK_hamiltonian_pyrochlore.dat', "r")
    ls_eq = fo.read().split('\n')
    fo.close()
    Matrix = np.zeros(3888).reshape(4,3,3,3,4,3,3)
    i = 0
    for row in ls_eq:
        i += 1
        data = [float(j) for j in row.split()]
        if(i in range(73)):
            Matrix[int(data[0]-1), int(data[1]+1), int(data[2]+1), int(data[3]+1), int(data[4]-1), int(data[5]-1), int(data[6]-1)] = J_int*data[-1]/2
        elif(i in range(73,217)):
            Matrix[int(data[0]-1), int(data[1]+1), int(data[2]+1), int(data[3]+1), int(data[4]-1), int(data[5]-1), int(data[6]-1)] = D_int*data[-1]/2
        else:
            Matrix[int(data[0]-1), int(data[1]+1), int(data[2]+1), int(data[3]+1), int(data[4]-1), int(data[5]-1), int(data[6]-1)] = K_int*data[-1]
    #del row, ls_eq, data
    return Matrix

@njit
def e_trans(m,Lambda,theta,phi):
    #m::local coordinate index
    #lambda::fixed coordinate index
    if(m==1):
        if(Lambda==1):
            return np.cos(theta)*np.cos(phi)
        elif(Lambda==2):
            return np.cos(theta)*np.sin(phi)
        elif(Lambda==3):
            return -np.sin(theta)
    elif(m==2):
        if(Lambda==1):
            return -np.sin(phi)
        elif(Lambda==2):
            return np.cos(phi)
        elif(Lambda==3):
            return 0
    elif(m==3):
        if(Lambda==1):
            return np.sin(theta)*np.cos(phi)
        elif(Lambda==2):
            return np.sin(theta)*np.sin(phi)
        elif(Lambda==3):
            return np.cos(theta)

@njit
def J_curly(J_Matrix,alpha, n_1, n_2, n_3, beta, Lambda, Mu):
    return J_Matrix[(alpha)-1,(n_1)+1,(n_2)+1,(n_3)+1,(beta)-1,(Lambda)-1,(Mu)-1]

@njit
def D(J_Matrix, alpha, n_1, n_2, n_3, beta, m, n, theta_alpha, phi_alpha, theta_beta, phi_beta):
    #m,n::local coordinate
    #alpha,beta:sublattice sites
    #i,j :lattice sites
    sum = 0
    for Lambda in [1,2,3]:
        for Mu in [1,2,3]:
            sum += J_curly(J_Matrix,alpha, n_1, n_2, n_3, beta, Lambda, Mu)*e_trans(m, Lambda, theta_alpha, phi_alpha)*e_trans(n, Mu, theta_beta, phi_beta)
    return sum

@njit
def B(m, theta, phi, B_ext):
    sum = 0
    for Lambda in range(1,4):
        sum += B_ext[Lambda-1]*e_trans(m, Lambda, theta, phi)
    return sum

@njit
def Linear_Terms(x, B_ext, J_Matrix):
    S = 1
    F = np.zeros(8)
    for alpha in [1,2,3,4]:
        sum1, sum2 = 0, 0
        for n_1 in [-1,0,1]:
            for n_2 in [-1,0,1]:
                for n_3 in [-1,0,1]:
                    for beta in [1,2,3,4]:
                        sum1 += (S)*(D(J_Matrix, alpha, -n_1, -n_2, -n_3, beta, 1, 3, x[alpha-1], x[alpha+3], x[beta-1], x[beta+3]))
                        sum2 += (S)*(D(J_Matrix, alpha, -n_1, -n_2, -n_3, beta, 2, 3, x[alpha-1], x[alpha+3], x[beta-1], x[beta+3]))
        sum1 -= B(1, x[alpha-1], x[alpha+3], B_ext)/2
        sum2 -= B(2, x[alpha-1], x[alpha+3], B_ext)/2
        F[alpha-1] = sum1
        F[alpha+3] = sum2
    return F

@njit
def S_Lambda(Lambda, theta_alpha, phi_alpha):
    S = 1
    if(Lambda==3):
        return S*np.cos(theta_alpha)
    elif(Lambda==1):
        return S*np.sin(theta_alpha)*np.cos(phi_alpha)
    elif(Lambda==2):
        return S*np.sin(theta_alpha)*np.sin(phi_alpha)
    else:
        print('Error')

@njit
def Classical_Energy_at(Angles, B_ext, J_Matrix):
    sum = 0
    for n1 in [-1,0,1]:
        for n2 in [-1,0,1]:
            for n3 in [-1,0,1]:
                for alpha in range(1,5):
                    for beta in range(1,5):
                        for Lambda in range(1,4):
                            for Mu in range(1,4):
                                sum += J_curly(J_Matrix,alpha, n1, n2, n3, beta, Lambda, Mu)*S_Lambda(Lambda, Angles[alpha-1], Angles[alpha+3])*S_Lambda(Mu, Angles[beta-1], Angles[beta+3]) 
    sum2 =0
    for alpha in range(1,5):
        for Lambda in range(1,4):
            sum2 -= B_ext[Lambda-1]*S_Lambda(Lambda, Angles[alpha-1], Angles[alpha+3])
    return sum/4 + sum2/4

def Energy_minimization(guess, B_ext, J_Matrix):
    res = minimize(Classical_Energy_at, guess, args=(B_ext, J_Matrix), bounds=((0.0,np.pi), (0.0,np.pi), (0.0,np.pi), (0.0,np.pi), (0.0,2.0*np.pi), (0.0,2.0*np.pi), (0.0,2.0*np.pi), (0.0,2.0*np.pi)))
    return res

############### For Spectrum ################
@njit
def J_pp(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/4)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0.0 + 1.0j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0.0 + 1.0j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) -D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 2, theta_alpha, phi_alpha, theta_beta, phi_beta))

@njit
def J_nn(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/4)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0.0 + 1.0j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0.0 + 1.0j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) -D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 2, theta_alpha, phi_alpha, theta_beta, phi_beta))    

@njit
def J_pn(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/4)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0.0 + 1.0j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0.0 + 1.0j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) +D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 2, theta_alpha, phi_alpha, theta_beta, phi_beta))

@njit
def J_np(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/4)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0.0 + 1.0j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0.0 + 1.0j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) +D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 2, theta_alpha, phi_alpha, theta_beta, phi_beta))

@njit
def J_p3(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/2)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 3, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0.0 + 1.0j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 3, theta_alpha, phi_alpha, theta_beta, phi_beta) )

@njit
def J_n3(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/2)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 3, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0.0 + 1.0j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 3, theta_alpha, phi_alpha, theta_beta, phi_beta) )

@njit
def J_3p(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/2)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 3, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0.0 + 1.0j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 3, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) )

@njit
def J_3n(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/2)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 3, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0.0 + 1.0j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 3, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) )

@njit
def J_33(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return D(J_Matrix, alpha, n_1, n_2, n_3, beta, 3, 3, theta_alpha, phi_alpha, theta_beta, phi_beta)

## 1j = +    and     -1j = -
@njit
def J_mn(J_Matrix, mu, nu, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    if(mu==1j):
        if(nu==1j):
            return J_pp(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==-1j):
            return J_pn(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==3):
            return J_p3(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        else:
            print('Error')
    elif(mu==-1j):
        if(nu==1j):
            return J_np(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==-1j):
            return J_nn(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==3):
            return J_n3(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        else:
            print('Error')
    elif(mu==3):
        if(nu==3):
            return D(J_Matrix, alpha, n_1, n_2, n_3, beta, 3, 3, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==1j):
            return J_3p(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==-1j):
            return J_3n(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        else:
            print('Error')
    else:
            print('Error')

@njit
def J_q(J_Matrix, q, mu, nu, alpha, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    sum = 0.0+0.0j
    for n1 in [-1,0,1]:
        for n2 in [-1,0,1]:
            for n3 in [-1,0,1]:
                kernal = np.exp((0.0-1.0j)*(n1*np.dot(q,np.array([0.5,0.5,0.0])) + n2*np.dot(q,np.array([0.0,0.5,0.5])) + n3*np.dot(q,np.array([0.5,0.0,0.5]))))
                sum += J_mn(J_Matrix, mu, nu, alpha, n1, n2, n3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)*kernal
    return sum

@njit
def Elements_A_q(J_Matrix, q, alpha, beta, Theta_s, Phi_s):
    sum = 0.0+0.0j
    if(alpha==beta):
        for gamma in range(1,5):
            sum += J_q(J_Matrix, np.array([0.0,0.0,0.0]), 3, 3, alpha, gamma, Theta_s[alpha-1], Phi_s[alpha-1], Theta_s[gamma-1], Phi_s[gamma-1]) + J_q(J_Matrix, np.array([0.0,0.0,0.0]), 3, 3, gamma, alpha, Theta_s[gamma-1], Phi_s[gamma-1], Theta_s[alpha-1], Phi_s[alpha-1])
    sum = -sum + 2.0*(J_q(J_Matrix, -q, -1j, 1j, alpha, beta, Theta_s[alpha-1], Phi_s[alpha-1], Theta_s[beta-1], Phi_s[beta-1]) + J_q(J_Matrix, q, 1j, -1j, beta, alpha, Theta_s[beta-1], Phi_s[beta-1], Theta_s[alpha-1], Phi_s[alpha-1]))
    return sum

@njit
def Elements_B_q(J_Matrix, q, alpha, beta, Theta_s, Phi_s):
    return 2.0*(J_q(J_Matrix, -q, -1j, -1j, alpha, beta, Theta_s[alpha-1], Phi_s[alpha-1], Theta_s[beta-1], Phi_s[beta-1]) + J_q(J_Matrix, q, -1j, -1j, beta, alpha, Theta_s[beta-1], Phi_s[beta-1], Theta_s[alpha-1], Phi_s[alpha-1]))


@njit
def Eigen_Value_finder(q, B_ext, J_Matrix, Theta_s, Phi_s):
    A_q = np.zeros((4,4), dtype=np.csingle)
    B_q = np.zeros((4,4), dtype=np.csingle)
    A_nq= np.zeros((4,4), dtype=np.csingle)

    for alpha in range(4):
        for beta in range(4):
            A_q[alpha, beta] = Elements_A_q(J_Matrix, q, alpha+1, beta+1, Theta_s, Phi_s)
            B_q[alpha, beta] = Elements_B_q(J_Matrix, q, alpha+1, beta+1, Theta_s, Phi_s)
            A_nq[alpha, beta] = Elements_A_q(J_Matrix, -q, alpha+1, beta+1, Theta_s, Phi_s)

    B_q_H = np.conjugate(B_q).T
    A_nq_T = A_nq.T

    D_q = np.zeros((8,8), dtype=np.csingle)
    for i in range(4):
        for j in range(4):
            if(i==j):
                D_q[i, j] = A_q[i,j] + B(3, Theta_s[i], Phi_s[i], B_ext)
                D_q[i+4, j+4] = -A_nq_T[i, j] - B(3, Theta_s[i], Phi_s[i], B_ext)
            else:
                D_q[i, j] = A_q[i,j]
                D_q[i+4, j+4] = -A_nq_T[i, j] 
            D_q[i, 4+j] = B_q[i,j]
            D_q[i+4, j] = -B_q_H[i, j]
    Eigen_Values = np.linalg.eigvals(D_q)
    if(Eigen_Values.imag.max()>1e-4): 
        print('Complex Eigen Value Error')
    return np.real(Eigen_Values)
	
	
def GS(B_ext, J_Matrix, guess):
    Energies = []
    Angles = []

    res = fsolve(Linear_Terms, guess, args=(B_ext, J_Matrix))
    Energies += [Classical_Energy_at(res, B_ext, J_Matrix)]
    Angles += [res]
    res = Energy_minimization(guess, B_ext, J_Matrix)
    Energies += [res.fun]
    Angles += [res.x]
        
    Energies = np.array(Energies)
    y=Angles[Energies.argmin()]
    
    root=fsolve(Linear_Terms,y,args=(B_ext,J_Matrix))
    return root

#####################################################################################


J_int = 1
D_int = 0.3
K_int = 0.12
B_val = 0.1
B_dir = [1,1,1]
B_ext = B_val*np.array(B_dir)/np.sqrt(3)

#AIAO guess
guess = np.array([0.9553166505837463,0.9433973089951246,2.1802789744445,2.180278973417027,0.7853981228724557,3.926990823237409,5.510376296045982,2.343605293321518])

Name = f"J = 1, D = 0.3, K = {K_int}, B = {B_val} along {B_dir}"

J_Matrix=Interaction_Matrix(J_int, D_int, K_int)

######################
Path = ['G','X','W','G','L','W','U','X','K','G']
G = np.array([0,0,0],dtype=np.float64)
X = np.array([2*np.pi,0,0],dtype=np.float64)
W = np.array([2*np.pi,np.pi,0],dtype=np.float64)
K = np.array([3*np.pi/2,3*np.pi/2,0],dtype=np.float64)
L = np.array([np.pi, np.pi, np.pi],dtype=np.float64)
U = np.array([2*np.pi,np.pi/2, np.pi/2],dtype=np.float64)
No_points = len(Path) 
x = []
y = [[] for i in range(8)]
x_lebal = [[],[]]
q = np.array([0,0,0], dtype=np.float64)
Q = []
x_l = 0
for p in range(No_points-1):
    Points_in_1_path = int(np.linalg.norm((locals()[Path[p+1]]-locals()[Path[p]])))*4
    for i in range(Points_in_1_path):
        q = locals()[Path[p]] + (locals()[Path[p+1]]-locals()[Path[p]])*i/Points_in_1_path
        Q += [q]
        x += [i+x_l]
        if(i==0): 
            x_lebal[0] += [x[-1]]
            if(Path[p]=='G'): x_lebal[1] += ['\u0393']
            else: x_lebal[1] += [Path[p]]
    x_l = x[-1]+1
q = locals()[Path[No_points-1]]
Q += [q]
x += [x_l]
x_lebal[0] += [x_l]
if(Path[No_points-1]=='G'): x_lebal[1] += ['\u0393']
else: x_lebal[1] += [Path[No_points-1]]
######################

root = GS(B_ext, J_Matrix, guess)
CE = Classical_Energy_at(root, B_ext, J_Matrix)
Max_LT = Linear_Terms(root, B_ext, J_Matrix).max()

Theta_s = np.array([root[0], root[1], root[2], root[3]], dtype=np.float64)
Phi_s =  np.array([root[4], root[5], root[6], root[7]], dtype=np.float64)

EV = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}

for q in Q:
    ev = np.sort(abs(Eigen_Value_finder(q, B_ext, J_Matrix, Theta_s, Phi_s)))
    for j in range(8):
        EV[j] += [ev[j]]

############################################
fig = go.Figure()
colors = ['#000000', '#000000', '#FF0000', '#FF0000', '#008000', '#008000', '#0000FF', '#0000FF']

for band_index in range(8):
    fig.add_trace(
        go.Scatter(
            line=dict(color=colors[band_index]),
            x= x,
            y= EV[band_index]))
fig.update(layout_showlegend=False)
fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = x_lebal[0],
        ticktext = x_lebal[1]),
    title = Name,
    font=dict(
        size=20,
        color="Black")
)
fig.show()