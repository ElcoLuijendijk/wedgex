"""
Python code for modelling advective-coductive heat transport using FiPy

This needs FiPY and GMSH:
* https://www.ctcms.nist.gov/fipy/
* https://gmsh.info/
        
Recommended installation method:
    
conda create --name fipy --channel conda-forge python=3.8.8 fipy
    
conda activate fipy

pip install gmsh

"""


import numpy as np
import scipy.interpolate

try:
    import fipy
except:
    raise ImportError('error, could not import FiPY module. check here on how to install this: https://www.ctcms.nist.gov/fipy/')

    
def create_rectangle_mesh_with_fault(Lx, Ly, alpha, beta, Lxmin, cellsize_wedge_top, cellsize_wedge_bottom, cellsize_footwall):
    
    xw0, yw0 = 0, 0
    xw1, yw1 = Lx, alpha * Lx
    xw2, yw2 = Lx, beta * Lx
    
    x_ul, y_ul = -Lxmin, 0
    x_ll, y_ll = -Lxmin, -Ly
    x_lm, y_lm = 0, -Ly
    x_lr, y_lr = Lx, beta * Lx -Ly
    

    geo = f'''

    // A mesh consisting of two wedges

    // define the corners of the wedge / hanging wall
    Point(1) = {{ {xw0}, {yw0}, 0, {cellsize_wedge_top} }};
    Point(2) = {{ {xw1}, {yw1}, 0, {cellsize_wedge_top} }};
    Point(3) = {{ {xw2}, {yw2}, 0, {cellsize_wedge_bottom} }};
    
    // corners of the footwall
    Point(4) = {{ {x_ul}, {y_ul}, 0, {cellsize_footwall} }};
    Point(5) = {{ {x_ll}, {y_ll}, 0, {cellsize_footwall} }};
    Point(6) = {{ {x_lm}, {y_lm}, 0, {cellsize_footwall} }};
    Point(7) = {{ {x_lr}, {y_lr}, 0, {cellsize_footwall} }};

    // define the hanging wedge
    Line(1) = {{1, 2}};
    Line(2) = {{2, 3}};
    Line(3) = {{3, 1}};
    
    // define the hanging wedge
    Line(4) = {{1, 4}};
    Line(5) = {{4, 5}};
    Line(6) = {{5, 6}};
    Line(7) = {{6, 7}};
    Line(8) = {{7, 3}};
    //+Line(3)
    
    // define the boundaries
    Line Loop(1) = {{1, 2, 3}};
    Line Loop(2) = {{4, 5, 6, 7, 8, 3}};

    // define the domain
    Plane Surface(1) = {{1}};
    Plane Surface(2) = {{2}};
    
    Physical Surface("wedge") = {1};
    Physical Surface("footwall") = {2};
    Physical Line("surface") = {1, 4};
    Physical Line("bottom") = {6, 7};

    Physical Volumes("cells") = {{1, 2}}

    '''

    mesh = fipy.Gmsh2D(geo)

    return mesh


def horizontal_compression_velocity(x, vc, L):
    
    return x * vc / L


def vertical_compression_velocity(x, y, vc, alpha, beta, L):
    
    vn = vc / L
    eta = (-2 - (beta)/ (alpha - beta))
    zeta = 2 * beta + (alpha * beta)/ (alpha - beta)
    
    vyc = vn * (eta * y + zeta * x)
    
    return vyc


def interpolate_data(xyz_array, data, dx, dy, limit_number_of_nodes=True, max_nodes=1e6):

    xi = np.arange(xyz_array[:, 0].min(), xyz_array[:, 0].max() + dx, dx)
    yi = np.arange(xyz_array[:, 1].min(), xyz_array[:, 1].max() + dy, dy)

    if (len(xi) * len(yi)) > max_nodes:
        print('warning, interpolating data on raster with >1e6 nodes')
        if limit_number_of_nodes is True:
            xi = np.linspace(xyz_array[:, 0].min(), xyz_array[:, 0].max(), int(np.sqrt(max_nodes)))
            yi = np.linspace(xyz_array[:, 1].min(), xyz_array[:, 1].max(), int(np.sqrt(max_nodes)))

    xg, yg = np.meshgrid(xi, yi)
    xgf, ygf = xg.flatten(), yg.flatten()
    zgf = scipy.interpolate.griddata(xyz_array, data, np.vstack((xgf, ygf)).T,
                                     method='linear')
    zg = np.resize(zgf, xg.shape)

    #zg = matplotlib.mlab.griddata(xyz_array[:, 0], xyz_array[:, 1], Ti,
    #                              xi, yi,
    #                              interp='linear')

    return xg, yg, zg




def model_heat_transport(Lx, Ly, alpha, beta, Lxmin, cellsize_wedge_top, cellsize_wedge_bottom, cellsize_footwall, 
                         vd, vc, vxa, vya, v_downgoing, 
                        sea_lvl_temp, lapse_rate, lab_temp, K, rho, c, H0, e_folding_depth):
    
    # create mesh
    mesh = create_rectangle_mesh_with_fault(Lx, Ly, alpha, beta, Lxmin, cellsize_wedge_top, cellsize_wedge_bottom, cellsize_footwall)

    # get mesh coordinates
    xyc = mesh.cellCenters()
    xyf = mesh.faceCenters()

    # calculate elevation and depth
    surface_elev= xyc[0] * alpha
    surface_elev_f = xyf[0] * alpha

    a = surface_elev < 0
    surface_elev[a] = 0.0

    b = surface_elev_f < 0
    surface_elev_f[b] = 0.0

    depth = surface_elev - xyc[1]
    depthf = surface_elev_f - xyf[1]

    wedge = xyc[1] >= (xyc[0] * beta)
    non_wedge = xyc[1]  < (xyc[0] * beta)
    below_wedge = non_wedge * xyc[0] >= 0
    outside_wedge = xyc[0] < 0

    wedge_f = xyf[1] >= (xyf[0] * beta)
    non_wedge_f = wedge_f == False
    below_wedge_f = non_wedge_f * (xyf[0] >= 0)
    outside_wedge_f = xyf[0] < 0

    # calculate heat production
    hp_exp = 1.0 / e_folding_depth
    HP = H0 * np.exp(-hp_exp * depth)

    # set up variable for advection
    q = fipy.FaceVariable(mesh=mesh, rank=1)

    # calcuate advection rates in wedge
    vxc = horizontal_compression_velocity(xyf[0], vc, Lx)
    vyc = vertical_compression_velocity(xyf[0], xyf[1], vc, alpha, beta, Lx)
    vxt = vd + vxa
    vyt = beta * vd + vya

    vx = vxc + vxt
    vy = vyc + vyt

    q[0] = wedge_f * vx
    q[1] = wedge_f * vy

    # calculate advection rates below wedge
    wedge_angle = np.arctan(beta)

    vx_downgoing = v_downgoing * np.cos(wedge_angle)
    vy_downgoing = v_downgoing * np.sin(wedge_angle)

    # set velocity in front of wedge
    q[0] = q[0] + outside_wedge_f * v_downgoing

    # set velocity below wedge
    q[0] = q[0] + below_wedge_f * vx_downgoing
    q[1] = q[1] + below_wedge_f * vy_downgoing

    # set up temperature variable
    T = fipy.CellVariable(mesh=mesh, name='T')

    # boudnary conditions
    surface_nodes = depthf <= 1.0
    bottom_nodes = ((xyf[0] <= 0) & (np.abs(depthf - Ly) < 1.0)) | ((xyf[0] > 0) & (np.abs(xyf[1] - (beta*xyf[0] - Ly)) < 1.0))

    surface_temp_variable = sea_lvl_temp + xyf[1] * lapse_rate

    #T.constrain(surface_temp_variable, mesh.physicalFaces['surface'])
    T.constrain(surface_temp_variable, surface_nodes)
    #T.constrain(lab_temp, mesh.physicalFaces['bottom'])
    T.constrain(lab_temp, bottom_nodes)

    # constrain flux at lateral boundaries. for some reason zero-flux not automatically enforced by fipy in this case.
    T.faceGrad.constrain([0.0], mesh.facesRight)
    T.faceGrad.constrain([0.0], mesh.facesLeft)

    # coefficients
    k = K / (rho * c)
    HP_adj = HP / (rho * c)

    # set up equation
    diffTerm = fipy.DiffusionTerm(coeff=k) 
    sourceTerm  =  fipy.CellVariable(mesh=mesh,  value=HP_adj)
    #sourceTerm_v2  =  fipy.ImplicitSourceTerm(HP_adj)

    #convTerm = fipy.ExponentialConvectionTerm(coeff=q)
    convTerm = fipy.PowerLawConvectionTerm(coeff=q)
    #convTerm = fipy.UpwindConvectionTerm(coeff=q)

    eq = (diffTerm == convTerm - sourceTerm)

    # solve eequation
    solver = fipy.solvers.LinearGMRESSolver(tolerance=1e-20, iterations=10000)
    eq.solve(var=T, solver=solver)

    # convert result to arrays
    T_array = T(mesh.cellCenters.globalValue)
    x, y = mesh.cellCenters.globalValue
    
    #q_array = q(mesh.cellCenters.globalValue)

    return x, y, T_array, q, mesh