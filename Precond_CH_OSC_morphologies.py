
import sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, minres
from dolfinx import log
from dolfinx.fem import Function, FunctionSpace, assemble_matrix, assemble_matrix_block, assemble_vector_block, form
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, create_box, locate_entities, MeshTags
from ufl import Measure, FiniteElement, TrialFunction, TestFunction, diff, dx, grad, inner, variable
from mpi4py import MPI
from petsc4py import PETSc
import pyamg


### Reference
# [1] K. Bergermann, C. Deibel, R. Herzog, R. C. I. MacKenzie, J.-F. Pietschmann and M. Stoll. Preconditioning for a phase-field model with application to morphology evolution in organic semiconductors. arXiv:2204.03575. 2022.


# Save all logging to file
log.set_output_file('log.txt')
postfix = ''

# Define counter for MINRES iterations
class minres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i' % self.niter)



########################
### Define functions ###
########################

def setup_2d_mesh(L, W, nx, ny):
    mesh = create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([L, W])], [nx, ny], CellType.triangle)
    return mesh
    
def setup_3d_mesh(L, W, H, nx, ny, nz):
    mesh = create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, H])], [nx, ny, nz], CellType.tetrahedron)
    return mesh


def setup_boudary_measures(mesh, substrate_patterning=False):

    if substrate_patterning:
    # Define boundaries (1: (bottom) substrate boundary with nfa preference, 2: (bottom) substrate boundary with polymer preference, 3: (top) surface boundary at which solvent evaporates)
        boundaries = [(1, lambda x: np.logical_and(np.isclose(x[1], 0), np.logical_or(np.isclose(x[0], (3*L)/12, rtol=0, atol=L/12), np.logical_or(np.isclose(x[0], (7*L)/12, rtol=0, atol=L/12), np.isclose(x[0], (11*L)/12, rtol=0, atol=L/12))))),
              (2, lambda x: np.logical_and(np.isclose(x[1], 0), np.logical_or(np.isclose(x[0], L/12, rtol=0, atol=L/12), np.logical_or(np.isclose(x[0], (5*L)/12, rtol=0, atol=L/12), np.isclose(x[0], (9*L)/12, rtol=0, atol=L/12))))),
              (3, lambda x: np.isclose(x[1], W))]
    
        # Identify boundary elements
        facet_indices, facet_markers = [], []
        fdim = mesh.topology.dim - 1
        for (marker, locator) in boundaries:
            facets = locate_entities(mesh, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))
        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = MeshTags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    
        # Setup three boundary measures
        ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)
    
    else:
        # Define boundaries (1: (bottom) substrate boundary, 2: (top) surface boundary at which solvent evaporates)
        boundaries = [(1, lambda x: np.isclose(x[1], 0)),
              (2, lambda x: np.isclose(x[1], W))]
    
        # Identify boundary elements
        facet_indices, facet_markers = [], []
        fdim = mesh.topology.dim - 1
        for (marker, locator) in boundaries:
            facets = locate_entities(mesh, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))
        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = MeshTags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    
        # Setup three boundary measures
        ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)
        
    return ds


def setup_function_space(mesh):

    # Build separate function spaces for concentrations and chemical potentials
    P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    MEphi, MEmu = FunctionSpace(mesh, P1), FunctionSpace(mesh, P1)
    
    return MEphi, MEmu


def setup_functions(MEphi, MEmu):

    # Trial functions
    dphi1, dphi2 = TrialFunction(MEphi), TrialFunction(MEphi)
    dmu1, dmu2 = TrialFunction(MEmu), TrialFunction(MEmu)
    
    # Test functions
    q1, q2 = TestFunction(MEphi), TestFunction(MEphi)
    v1, v2 = TestFunction(MEmu), TestFunction(MEmu)
    
    # Solutions of concentrations and chemical potential for current and previous time step
    phi1, phi2 = Function(MEphi), Function(MEphi)
    mu1, mu2 = Function(MEmu), Function(MEmu)
    phi01, phi02 = Function(MEphi), Function(MEphi)
    mu01, mu02 = Function(MEmu), Function(MEmu)
    
    # Functions used for writing the solution to xdmf
    u1, u2 = Function(MEphi), Function(MEmu)
    
    return q1, v1, q2, v2, dphi1, dmu1, dphi2, dmu2, phi1, mu1, phi2, mu2, phi01, mu01, phi02, mu02, u1, u2


def setup_initial_conditions(pol_init, pol_init_fluct, nfa_init, nfa_init_fluct):

    # Zero initial conditions for chemical potentials
    mu1.x.array[:] = 0.0
    mu2.x.array[:] = 0.0

    # Interpolate initial conditions for concentrations (polymer and nfa)
    phi1.interpolate(lambda x: pol_init + 2*pol_init_fluct * (0.5 - np.random.rand(x.shape[1])))
    phi1.x.scatter_forward()
    phi2.interpolate(lambda x: nfa_init + 2*nfa_init_fluct * (0.5 - np.random.rand(x.shape[1])))
    phi2.x.scatter_forward()
    
    return phi1, phi2, mu1, mu2


def setup_potentials(phi01, phi02, RT_V, N_p, N_nfa, N_s, chi_p_nfa, chi_p_s, chi_nfa_s):

    # Compute the chemical potential df/dc
    # TODO: implement logarithmic potential
    
    # Polynomial approximation of the logarithmic potential term
    phi01 = variable(phi01)
    phi02 = variable(phi02)
        
    f = RT_V*(3.5*phi01**2*phi02**2+0.1*(1-phi01-phi02)**2)
    
    # Partial derivatives of bulk potential (w.r.t: 1: polymer, 2: nfa)
    dfdphi01 = diff(f, phi01)
    dfdphi02 = diff(f, phi02)

    # Define substrate potential df_sub/dc
    f_sub01 = -0.01*phi01-0.0*phi01**2
    f_sub02 = -0.01*phi02-0.0*phi02**2
    df_sub01dc = diff(f_sub01,phi01)
    df_sub02dc = diff(f_sub02,phi02)

    # Define flux terms for Neumann boundary conditions
    k = 5e-03
    g01 = 0
    h01 = - k*phi01*(1-phi01-phi02)
    g02 = 0
    h02 = - k*phi02*(1-phi01-phi02)

    return dfdphi01, dfdphi02, df_sub01dc, df_sub02dc, g01, h01, g02, h02


def setup_Krylov_solver():
    # Separate solvers for the two (polymer and nfa) equations
    ksp1, ksp2 = PETSc.KSP(), PETSc.KSP()
    ksp1.create(PETSc.COMM_WORLD)
    ksp2.create(PETSc.COMM_WORLD)
    return ksp1, ksp2
    

def IMEX_time_step(tau, ds, dphi1, q1, dmu1, dphi2, q2, dmu2, v1, v2, phi01, h01, phi02, h02, dfdphi01, g01, df_sub01dc, dfdphi02, g02, df_sub02dc, ksp1, ksp2, substrate_patterning=False):

    # Assemble block matrices
    Lone_block = form([[inner(dphi1, q1)*dx, tau*inner(grad(dmu1),grad(q1))*dx], 
        [tau*inner(grad(dphi1),grad(v1))*dx, (-tau/epsilon)*inner(dmu1, v1)*dx]])
    Ltwo_block = form([[inner(dphi2, q2)*dx, tau*inner(grad(dmu2),grad(q2))*dx], 
        [tau*inner(grad(dphi2),grad(v2))*dx, (-tau/epsilon)*inner(dmu2, v2)*dx]])
    
    A1_block = assemble_matrix_block(Lone_block)
    A1_block.assemble()
    A2_block = assemble_matrix_block(Ltwo_block)
    A2_block.assemble()
    
    # Assemble block vectors
    if substrate_patterning:
        fone_block = form([inner(phi01, q1)*dx + tau*inner(h01, q1)*ds(3), (-tau/epsilon)*(inner(dfdphi01, v1)*dx - inner(g01, v1)*ds(3) + inner(df_sub01dc, v1)*ds(2))])
        ftwo_block = form([inner(phi02, q2)*dx + tau*inner(h02, q2)*ds(3), (-tau/epsilon)*(inner(dfdphi02, v2)*dx - inner(g02, v2)*ds(3) + inner(df_sub02dc, v2)*ds(1))])
    else:
        fone_block = form([inner(phi01, q1)*dx + tau*inner(h01, q1)*ds(2), (-tau/epsilon)*(inner(dfdphi01, v1)*dx - inner(g01, v1)*ds(2) + inner(df_sub01dc, v1)*ds(1))])
        ftwo_block = form([inner(phi02, q2)*dx + tau*inner(h02, q2)*ds(2), (-tau/epsilon)*(inner(dfdphi02, v2)*dx - inner(g02, v2)*ds(2) + inner(df_sub02dc, v2)*ds(1))])
    
    b1 = assemble_vector_block(fone_block, Lone_block)
    b2 = assemble_vector_block(ftwo_block, Ltwo_block)
    
    # Convert block matrices to scipy-sparse csr format
    A1i, A1j, A1v = A1_block.getValuesCSR()
    A1_block_sp = sp.csr_matrix((A1v, A1j, A1i))
    A2i, A2j, A2v = A2_block.getValuesCSR()
    A2_block_sp = sp.csr_matrix((A2v, A2j, A2i))
    
    # Assemble preconditioner matrices and set up AMG Ruge-Stüben solvers
    M = assemble_matrix(form(inner(dphi1, q1)*dx))
    M.assemble()
    Mi, Mj, Mv = M.getValuesCSR()
    M_sp = sp.csr_matrix((Mv, Mj, Mi))

    P22_mat = assemble_matrix(form(tau*inner(grad(dphi1),grad(q1))*dx + np.sqrt(tau/epsilon)*inner(dphi1, q1)*dx))
    P22_mat.assemble()
    P22_mati, P22_matj, P22_matv = P22_mat.getValuesCSR()
    P22_mat_sp = sp.csr_matrix((P22_matv, P22_matj, P22_mati))

    M_amg = pyamg.ruge_stuben_solver(M_sp)
    P22_mat_amg = pyamg.ruge_stuben_solver(P22_mat_sp)

    
    # Preconditioner from [1, Algorithm 1]
    def precond_lin_op(w, M_amg=M_amg, P22_mat_amg=P22_mat_amg):
    
        # Decompose block vector
        n, n = M_sp.get_shape()
        w1, w2 = w[:n], w[n:2*n]
        
        # Approximately solve linear systems using AMG
        v1 = M_amg.solve(w1, tol=1e-04)
        v2__ = P22_mat_amg.solve(w2, tol=1e-04)
        v2_ = M_sp.dot(v2__)
        v2 = P22_mat_amg.solve(v2_, tol=1e-04)
        
        # Concatenate vectors into block form again
        v = np.concatenate((v1, v2))
                
        return v
    
    
    # Construct LinearOperator from function handle precond_lin_op
    nn, nn = A1_block_sp.get_shape()
    P_lin_op = LinearOperator((nn, nn), matvec=precond_lin_op)
    
    # Solve linear systems with preconditioner MINRES (flag 0 means convergence)
    counter = minres_counter()
    start_time = time.time()
    x1, x1_convergence = minres(A1_block_sp, b1, tol=1e-07, M=P_lin_op, callback=counter)
    print("minres pol took %s seconds" % (time.time() - start_time))
    print('pol minres convergence flag', x1_convergence)
    start_time = time.time()
    x2, x2_convergence = minres(A2_block_sp, b2, tol=1e-07, M=P_lin_op, callback=counter)
    print("minres nfa took %s seconds" % (time.time() - start_time))
    print('nfa minres convergence flag', x1_convergence)
           
    return x1, x2


############################
### Set model parameters ###
############################

# Specify domain ('2d' or '3d')
domain='3d'

# Domain parameters
if domain=='2d':
    L, W = 10.0, 2.5
    nx, ny = 200, 100
    postfix += domain
elif domain=='3d':
    L, W, H = 10.0, 2.5, 10.0
    nx, ny, nz = 80, 40, 80
    postfix += domain
else:
    print('Error, domain not defined.')
    sys.exit(0)

# Cahn-Hilliard interface parameter
epsilon = 1.0e-03

# Time step size
tau = 1e-04

# Intial and terminal time
t = tau
T = 1e-01

# Initial concentrations (polymer + nfa) and magnitude of initial random fluctuations
# (equally distributed, s.t., e.g., u1 \in [pol_init-pol_init_fluct, nfa_init+nfa_init_fluct])
pol_init, pol_init_fluct = 0.35, 0.01
nfa_init, nfa_init_fluct = 0.35, 0.01

# Constant in bulk potential term
RT_V = 1

# Degree of polymerization (of polymer, nfa, and solvent)
N_p, N_nfa, N_s = 20, 20, 1

# Flory-Huggins interaction parameters (polymer-nfa, polymer-solvent, nfa-solvent)
chi_p_nfa, chi_p_s, chi_nfa_s = 1, 0.3, 0.3

# Substrate patterning switch
substrate_patterning = False
if substrate_patterning:
    postfix += '_substrate_patterning'

# Output file
filename1 = 'results/Precond_CH_OSC_polymer_%s.xdmf' % postfix
filename2 = 'results/Precond_CH_OSC_nfa_%s.xdmf' % postfix
file1, file2 = XDMFFile(MPI.COMM_WORLD, filename1, 'w'), XDMFFile(MPI.COMM_WORLD, filename2, 'w')
    

###################
### Main script ###
###################

# Setup problem
if domain=='2d':
    mesh = setup_2d_mesh(L, W, nx, ny)
elif domain=='3d':
    mesh = setup_3d_mesh(L, W, H, nx, ny, nz)
else:
    print('Error, domain not defined.')
    sys.exit(0)
    
file1.write_mesh(mesh)
file2.write_mesh(mesh)
    
ds = setup_boudary_measures(mesh, substrate_patterning)

MEphi, MEmu = setup_function_space(mesh)

q1, v1, q2, v2, dphi1, dmu1, dphi2, dmu2, phi1, mu1, phi2, mu2, phi01, mu01, phi02, mu02, u1, u2 = setup_functions(MEphi, MEmu)

# Degrees of freedom per function space
n = len(phi1.vector.array)

phi1, phi2, mu1, mu2 = setup_initial_conditions(pol_init, pol_init_fluct, nfa_init, nfa_init_fluct)

dfdphi01, dfdphi02, df_sub01dc, df_sub02dc, g01, h01, g02, h02 = setup_potentials(phi01, phi02, RT_V, N_p, N_nfa, N_s, chi_p_nfa, chi_p_s, chi_nfa_s)

ksp1, ksp2 = setup_Krylov_solver()

# Setup iteration counter
it = 0

# Loop over time
while t < T:
    # Move current solution to previous time step
    phi01.x.array[:] = phi1.x.array
    phi02.x.array[:] = phi2.x.array
    mu01.x.array[:] = mu1.x.array
    mu02.x.array[:] = mu2.x.array
    
    start_time = time.time()
    # Solve problem with time step size tau
    x1, x2 = IMEX_time_step(tau, ds, dphi1, q1, dmu1, dphi2, q2, dmu2, v1, v2, phi01, h01, phi02, h02, dfdphi01, g01, df_sub01dc, dfdphi02, g02, df_sub02dc, ksp1, ksp2, substrate_patterning)
    print("IMEX_time_step took %s seconds" % (time.time() - start_time))
    print("Norm between polymer iterates:", np.linalg.norm(phi1.x.array[:]-x1[:n]))
    print("Norm between nfa iterates:", np.linalg.norm(phi2.x.array[:]-x2[:n]))
    
    mu1.x.array[:] = x1[n:2*n]
    phi1.x.array[:] = x1[:n]
    u1.x.array[:n] = x1[:n]
    
    mu2.x.array[:] = x2[n:2*n]
    phi2.x.array[:] = x2[:n]
    u2.x.array[:n] = x2[:n]
    
    # Print current time and time step size
    print(f'Time {t: .5f}, time step size {tau: .7f}')
    
    # Write only the solution of every it_write'th time step
    it_write = 20
    
    if (it%it_write)==0:
        # Write solution to xdmf file
        file1.write_function(u1, t)
        file2.write_function(u2, t)
    
    # Advance time
    t += tau
    it += 1


file1.close()
file2.close()

print('Simulation done. Look for files\n%s\n%s' % (filename1, filename2))

