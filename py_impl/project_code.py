import trimesh
import numpy as np
from scipy.sparse import bsr_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import bmat
from scipy.sparse import identity
# from scipy.sparse import dia_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
import plotly.graph_objs as go
import plotly
import matplotlib.cm as cm
from math import *

def get_mesh(name):
    # return the E, F, V data of the mesh
    mesh = trimesh.load_mesh(name)
    return mesh.edges, mesh.faces, mesh.vertices

def get_unique_edge_list(edge):
    # default edge gives 3 edges for every face
    # which contains both the pos and neg direction of every edge
    edge_sets = set([ frozenset(x) for x in edge ])
    return [list(x) for x in edge_sets]

def get_edge_face_index(edge_list, face):
    # the 'edge'th row has the indices of its corresponding faces,
    # and which edge this is in the faces
    ne = len(edge_list)
    nf = face.shape[0]
    edge_face_index = np.zeros(shape=(ne, 4), dtype=int)
    #               (neg_face, neg_index, pos_face, pos_index)
    for i in range(nf):
        for j in range(3):
            pos = [face[i][j-1], face[i][j]]
            neg = [face[i][j], face[i][j-1]]
            if pos in edge_list:
                index = edge_list.index(pos)
                edge_face_index[index][2] = i
                edge_face_index[index][3] = j-1
            else:
                index = edge_list.index(neg)
                edge_face_index[index][0] = i
                edge_face_index[index][1] = j-1    
    return edge_face_index

def get_face_norm(vertex, face):
    # compute the unit normal vector for every face
    face_ji = (vertex[face[:, 0]] - vertex[face[:, 1]])
    face_jk = (vertex[face[:, 2]] - vertex[face[:, 1]])
    face_norm = np.cross(face_ji, face_jk, axis=1)
    face_norm = face_norm / np.linalg.norm(face_norm, axis=1)[:, np.newaxis]
    return face_norm

def get_frame(face_ji, face_norm):
    # compute the frames (3*3 matrix) for every face
    f_1 = -face_ji / np.linalg.norm(face_ji, axis=1)[:, np.newaxis]
    f_2 = np.cross(face_norm, f_1)
    # f_3 = face_norm
    f = np.column_stack((f_1, f_2, face_norm)).reshape(nf, 3, 3)
    return f

def get_edge_angle(edge_vec, face_norm, edge_face_index):
    neg_norm = face_norm[edge_face_index[:, 0]]
    pos_norm = face_norm[edge_face_index[:, 2]]
    cross_norm = np.cross(neg_norm, pos_norm, axis=1)
    edge_cos = np.sum(neg_norm*pos_norm, axis=1)
    edge_sign = np.sign(np.sum(cross_norm*edge_vec, axis=1))
    edge_sin = np.linalg.norm(cross_norm, axis=1) * edge_sign
    return edge_cos, edge_sin

def get_mesh_data(name):
    # return the first, and second foundamental forms of mesh
    # and also the connectivity data
    edge, face, vertex = get_mesh(name)
    edge_list = get_unique_edge_list(edge)
    edge = np.array(edge_list)
    
    edge_face_index = get_edge_face_index(edge_list, face)
    edge_vec = vertex[edge[:, 0]] - vertex[edge[:, 1]]
    edge_len = np.linalg.norm(edge_vec, axis=1)
    face_norm = get_face_norm(vertex, face)
    edge_cos, edge_sin = get_edge_angle(edge_vec, face_norm, edge_face_index)
    return edge, face, edge_face_index, edge_len, edge_cos, edge_sin

def get_face_angles(edge_len, edge_face_index, nf):
    # return cot_angle: the 3 opposite angles to each edge
    # return rot_angle: the amount needs to be rotated to align frames
    face_edge_len = np.zeros(shape=(nf, 3))
    # note that we shouldn't be suing the vertex data anymore
    # since the only stored data for the mesh are the connectivity
    # data, edge length, and transition angles
    for i, row in enumerate(edge_face_index):
        face_edge_len[row[0]][row[1]] = edge_len[i]
        face_edge_len[row[2]][row[3]] = edge_len[i]
    face_edge_len_2 = face_edge_len**2
    # the jth angle is the angle opposite to the second edge
    face_angle_j = np.arccos((face_edge_len_2[:, 2]+face_edge_len_2[:, 0]-face_edge_len_2[:, 1]) /\
                             (2*face_edge_len[:, 2]*face_edge_len[:, 0]))
    face_angle_k = np.arccos((face_edge_len_2[:, 0]+face_edge_len_2[:, 1]-face_edge_len_2[:, 2]) /\
                             (2*face_edge_len[:, 0]*face_edge_len[:, 1]))
    face_cot_angle = np.column_stack((np.pi - face_angle_j - face_angle_k, face_angle_j, face_angle_k))
    face_rot_angle = np.column_stack((np.zeros(nf), np.pi - face_angle_k, face_angle_j - np.pi))
    return face_cot_angle, face_rot_angle

def get_edge_face_angles(edge_face_index, face_rot_angle, ne):
    edge_angle_neg = face_rot_angle[edge_face_index[:, 0], edge_face_index[:, 1]]
    edge_angle_pos = face_rot_angle[edge_face_index[:, 2], edge_face_index[:, 3]]
    neg_sin = np.sin(np.pi + edge_angle_neg) # negative edge, flip direction
    neg_cos = np.cos(np.pi + edge_angle_neg)
    pos_sin = np.sin(edge_angle_pos)
    pos_cos = np.cos(edge_angle_pos)
    return neg_sin, neg_cos, pos_sin, pos_cos

def get_transition_matrices(neg_sin, neg_cos, pos_sin, pos_cos, edge_sin, edge_cos, ne):
    # angle (sin, cos) in the negative side face, the positive side, between the 2 faces
    # Assuming a local frame with ei, b, N.
    Rie = np.zeros(shape=(ne, 3, 3)) # rotate around z axis
    Rie[:, 2, 2] = 1
    Rie[:, 0, 0] = Rie[:, 1, 1] = neg_cos
    Rie[:, 0, 1] = neg_sin
    Rie[:, 1, 0] = -neg_sin

    Re = np.zeros(shape=(ne, 3, 3)) # rotate around x axis
    Re[:, 0, 0] = 1
    Re[:, 1, 1] = Re[:, 2, 2] = edge_cos
    Re[:, 1, 2] = edge_sin
    Re[:, 2, 1] = -edge_sin

    Rej = np.zeros(shape=(ne, 3, 3)) # rotate around z axis
    Rej[:, 2, 2] = 1
    Rej[:, 0, 0] = Rej[:, 1, 1] = pos_cos # pos_cos_back = pos_cos 
    Rej[:, 0, 1] = -pos_sin # pos_sin_back = -pos_sin
    Rej[:, 1, 0] = pos_sin # -pos_sin_back = pos_sin

    Riee = np.einsum('...jk, ...km->...jm', Rie, Re)
    Rij = np.einsum('...jk, ...km->...jm', Riee, Rej)
    return Rij

def get_edge_cot_angles(face_cot_angle, edge_face_index):
    edge_angle_alpha = face_cot_angle[edge_face_index[:, 0], edge_face_index[:, 1]]
    edge_angle_beta = face_cot_angle[edge_face_index[:, 2], edge_face_index[:, 3]]
    return edge_angle_alpha, edge_angle_beta

def get_dEf(edge_face_index, Rij, edge_angle_alpha, edge_angle_beta, ne, material_weight=None):
    # Ef = 1/2 * SUM_edges (1/we |fj - fi*Rij|^2)
    # D(Ef)/ Df = R3'*Wf*R3*f where
    # R3: each row jth entry is Id, ith entry is -RijT
    # Wf: each diagnal entry is 1/(cot(alpha_ij) + cot(beta_ij))
    R3_a_indptr = np.array(range(ne+1))
    R3_a_indices = edge_face_index[:, 0]
    R3_a = bsr_matrix((np.transpose(Rij, axes=(0,2,1)), \
                       R3_a_indices, R3_a_indptr))
    R3_b_indices = edge_face_index[:, 2]
    R3_b = bsr_matrix((np.tile(identity(3).toarray(), (ne, 1, 1)),\
                       R3_b_indices, R3_a_indptr))
    R3 = -R3_a + R3_b
    # sanity check: build f from given mesh
    # computed in the first section
    # R3*f.reshape(nf*3, 3), should be all 0's
    edge_weight = 1/np.tan(edge_angle_alpha) + 1/np.tan(edge_angle_beta)
    Wf_indptr = np.array(range(ne+1))
    Wf_indices = np.array(range(ne))
    Wf_data = np.tile(identity(3).toarray(), (ne, 1, 1))*\
              (( 1/edge_weight * material_weight )[:, np.newaxis, np.newaxis])
    Wf = bsr_matrix((Wf_data, Wf_indices, Wf_indptr))    
    return R3, Wf

def get_dEx(edge, edge_face_index, edge_coord, edge_angle_alpha, edge_angle_beta, ne, material_weight=None):
    # Ex1 = 1/2 * SUM_edges (w1e |xn - xm  - f1 * (a11, a12, 0)'|^2)
    # Ex2 = 1/2 * SUM_edges (w2e |xn - xm  - f2 * (a21, a22, 0)'|^2)
    # D(Ex1)/ Df = -R12'W1xR12*f + R12'*W1xR1*x
    # D(Ex2)/ Df = -R22'W2xR22*f + R22'*W2xR1*x
    # D(Ex1)/ Dx = -R1'W1xR12*f + R1'*W1xR1*x
    # D(Ex2)/ Dx = -R1'W2xR22*f + R1'*W2xR1*x where
    # R1: each row: nth entry is Id, mth entry, -Id
    # R12: each row: f1th entry is (a11, a12, 0)
    # R22: each row: f2th entry is (a21, a22, 0)
    R1_indptr = np.array(range(ne+1))*2
    R1_indices = edge.flatten()
    R1_data = np.tile(np.array([-1, 1]), ne)
    R1 = csr_matrix((R1_data, R1_indices, R1_indptr))

    R12_indptr = np.array(range(ne+1))
    R12_indices = edge_face_index[:, 0] 
    R12_data = np.column_stack((edge_coord[:, [0, 1]], np.array([0]*ne)))
    R12 = bsr_matrix((R12_data[:,np.newaxis, :], R12_indices, R12_indptr))

    R22_indices = edge_face_index[:, 2] 
    R22_data = np.column_stack((edge_coord[:, [2, 3]], np.array([0]*ne)))
    R22 = bsr_matrix((R22_data[:,np.newaxis, :], R22_indices, R12_indptr))
    # sanity check: use vertex from given mesh
    # R12*f.reshape(nf*3, 3)-R1*vertex*2+R22*f.reshape(nf*3, 3), should be all 0's
    W1x_indptr = np.array(range(ne+1))
    W1x_indices = np.array(range(ne))
    W1x_data = 1/np.tan(edge_angle_alpha) * material_weight
    W1x = csr_matrix((W1x_data, W1x_indices, W1x_indptr))

    W2x_data = 1/np.tan(edge_angle_beta) * material_weight
    W2x = csr_matrix((W2x_data, W1x_indices, W1x_indptr))
    return R1, R12, R22, W1x, W2x

def get_M(R3, Wf, R1, R12, R22, W1x, W2x, w):
    # Now, we should have:
    #                  M1                              M2
    # (w*R3'WfR3 - R12'W1xR12 - R22'W2xR22)*f + (R12'W1xR1 + R22'W2xR1)*x = 0
    #                  M3                              M4
    # (-R1'W1xR12 -R1'W2xR22)*f + (R1'W1xR1 + R1'W2xR1)*x = 0
    # M1*f + M2*x = 0
    # M3*f + M4*x = 0
    # ( M1 M2 ) * ( f ) = ( 0 )
    # ( M3 M4 )   ( x )   ( 0 )
    M1 = w*R3.transpose()*Wf*R3 - R12.transpose()*W1x*R12 - R22.transpose()*W2x*R22
    M2 = R12.transpose()*W1x*R1 + R22.transpose()*W2x*R1
    M3 = -R1.transpose()*W1x*R12 -R1.transpose()*W2x*R22
    M4 = R1.transpose()*W1x*R1 + R1.transpose()*W2x*R1
    M = bmat([[M1, M2], [M3, M4]], format='csc') # need columns
    # sanity check:
    # M*np.row_stack((f.reshape(nf*3, 3), vertex)) should be all 0's
    return M

def find_index(x_range, y_range, z_range, vertices):
    return np.where(np.logical_and(np.logical_and(vertices[:, 2] > z_range[0], vertices[:, 2] < z_range[1]),
                                   np.logical_and(np.logical_and(vertices[:, 0] > x_range[0], vertices[:, 0] < x_range[1]),\
                                                  np.logical_and(vertices[:, 1] > y_range[0], vertices[:, 1] < y_range[1]))))[0]

def find_edge(vertices, edges):
    return np.where(np.logical_and( np.in1d(edges[:, 0], vertices), np.in1d(edges[:, 1], vertices)))[0]

def plotly_trisurf(x, y, z, colors, simplices, colormap=cm.RdBu, plot_edges=False):
    points3D=np.vstack((x,y,z)).T
    tri_vertices=map(lambda index: points3D[index], simplices)
    
    ncolors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))
    
    I = simplices[:, 0]; J = simplices[:, 1]; K = simplices[:, 2]
    triangles=go.Mesh3d(x=x,y=y,z=z,
                     intensity=ncolors,
                     colorscale='Viridis',
                     i=I,j=J,k=K,name='',
                     showscale=True,
                     colorbar=go.ColorBar(tickmode='array',
                                       tickvals=[np.min(z), np.max(z)],
                                       ticktext=['{:.3f}'.format(np.min(colors)),
                                                 '{:.3f}'.format(np.max(colors))]))
    
    if plot_edges is False: # the triangle sides are not plotted
        return Data([triangles])
    else:
        lists_coord=[[[T[k%3][c] for k in range(4)]+[ None] for T in tri_vertices] for c in range(3)]
        Xe, Ye, Ze=[reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]
        lines=go.Scatter3d(x=Xe,y=Ye,z=Ze,mode='lines',line=go.Line(color='rgb(50,50,50)', width=1.5))
        return go.Data([triangles, lines])


#### from mesh data compute first, second foundamental forms ####
edge, face, edge_face_index, edge_len, edge_cos, edge_sin = get_mesh_data('moomoo.off')
nf = face.shape[0]; ne = edge.shape[0]
# from edge length compute face angles
face_cot_angle, face_rot_angle = get_face_angles(edge_len, edge_face_index, nf)
edge_angle_alpha, edge_angle_beta = get_edge_cot_angles(face_cot_angle, edge_face_index)
neg_sin, neg_cos, pos_sin, pos_cos = get_edge_face_angles(edge_face_index, face_rot_angle, ne)

#### Compute linear system for minimization ####
E, F, V = get_mesh('moomoo.off') ######## TODO HERE: add loop, try more ########
E_list = get_unique_edge_list(E) ######## shapes, different parameters  ########
E = np.array(E_list)
nv = V.shape[0]
# f_index = np.array([]) # final inpute format
v_handle = find_index([5, 12], [-2.5, 5.5], [7, 13], V)
v_fix = np.delete(np.array(range(nv)), find_index([5, 20], [-3, 12], [-10, 20], V), axis=0)

v_soft_large = find_index([6, 19], [-2, 9], [-3, 15], V)
v_soft_mid = np.union1d(find_index([8, 19], [-2, 9], [-1, 6], V), find_index([15, 19], [-2, 9], [-2, 8], V))
v_soft_small = np.union1d(find_index([8, 19], [-2, 9], [1, 4], V), find_index([15, 19], [-2, 9], [-1, 6], V))


bending_weight = np.array([0.0]*ne)+1; streching_weight = np.array([0.0]*ne)+1
bending_weight[find_edge(v_soft_large, E)] = 2
bending_weight[find_edge(v_soft_mid, E)] = 1.7
bending_weight[find_edge(v_soft_small, E)] = 1.3
streching_weight[find_edge(v_soft_large, E)] =  0.9
streching_weight[find_edge(v_soft_mid, E)] =  0.7
streching_weight[find_edge(v_soft_small, E)] =  0.5


# From face angle compute R_ij's
Rij = get_transition_matrices(neg_sin, neg_cos, pos_sin, pos_cos, edge_sin, edge_cos, ne)
# From face_angle and edge_len compute a1T, a2T
# use negative sin instead of positive sin for b (y axis)
edge_coord = np.column_stack((edge_len * neg_cos, -edge_len * neg_sin, \
                              edge_len * pos_cos, -edge_len * pos_sin))
R3, Wf = get_dEf(edge_face_index, Rij, edge_angle_alpha, edge_angle_beta, ne, bending_weight)
R1, R12, R22, W1x, W2x = get_dEx(edge, edge_face_index, edge_coord, edge_angle_alpha, edge_angle_beta, ne, streching_weight)
M = get_M(R3, Wf, R1, R12, R22, W1x, W2x, 1000)

#### demo: include constraint: 1 frame and 1 vertex: can rotate/ translate ####

v_color = np.array([0.0]*nv) + 2
v_color[v_handle] -=0.5
v_color[v_fix] -=0.5
v_color[v_soft_large] += 0.5
v_color[v_soft_mid] += 0.5
v_color[v_soft_small] += 0.5

to_delete = np.concatenate((nf*3+v_handle, nf*3+v_fix)) # f_index*3, f_index*3+1, f_index*3+2, 
RR = identity(M.shape[0], format='lil')
RR.rows = np.delete(RR.rows, to_delete)
RR.data = np.delete(RR.data, to_delete)
RR._shape = (RR._shape[0]-to_delete.shape[0], RR._shape[1])
RR.tocsr()

RC = RR.transpose()
M_cols = M[:, to_delete]

handle_rows = V[v_handle, :]

###########
ankle = np.array([14, 4, 3])
axis = np.array([0, 1, 0])
angle = (-50. / 180) * np.pi
diff_vec = handle_rows - ankle
new_vec = diff_vec * cos(angle) + sin(angle)*np.cross(axis, diff_vec) + (1-cos(angle))*np.dot(diff_vec, axis)[:, np.newaxis]*axis
##########
handle_rows = new_vec + ankle
# handle_rows[:, 1] += 10

fix_rows = V[v_fix, :]
fx_rows = np.row_stack((handle_rows, fix_rows))
b = -RR*M_cols*fx_rows
M = RR*M*RC
x = (RR.transpose()*spsolve(M, b))[nf*3:, :]
x[v_handle] = handle_rows
x[v_fix] = fix_rows



data1 = plotly_trisurf(x[:, 0], x[:,1], x[:,2], v_color, F, colormap=cm.RdBu, plot_edges=True)
# data1 = plotly_trisurf(V[:, 0], V[:,1], V[:,2], v_color, F, colormap=cm.RdBu, plot_edges=True)
# triangels = go.Mesh3d(x=x[:,0], y=x[:,1], z=x[:,2], i=face[:,0], j=face[:,1], k=face[:,2], opacity=0.4)
plotly.offline.plot(data1, filename='temp-plot_rigit_strech.html', auto_open=True)
