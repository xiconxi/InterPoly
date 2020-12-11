'''
Author: Pupa
LastEditTime: 2020-12-11 17:50:16
'''
import sys 
sys.path.append("../")

import scipy.sparse as sparse
import InterPoly 
import scipy.io as sio 
import numpy as np 
import openmesh as om 


def LabeledScalpSmoothing(V, VT, F, VL, color_map, n_steps = 20, prev_smooth_file="", smoothed_file=""):
    inter_polyer = InterPoly.InterPolyMesh(V, F, VL)
    # _F == inter_polyer.face_vertex_indices()
    # it wouldn't change during the remaining process
    _, _F = inter_polyer.extract_region_graph_boundary()
    _FC, _VC, _FL, _VL  = inter_polyer.coloring(color_map)
    if prev_smooth_file is not "":
        om.write_mesh(prev_smooth_file, inter_polyer.m, face_color=True, vertex_color=True)

    inter_polyer.quadric_smoothing(n_step=n_steps)

    if smoothed_file is not "":
        om.write_mesh(smoothed_file, inter_polyer.m, face_color=True, vertex_color=True)


    interpoly_matrix = inter_polyer.interpoly_matrix

    _VT = interpoly_matrix.dot(VT)
    _VT[:, 0], _VT[:, 1] = _VT[:, 0]*2048,  (1-_VT[:, 1])*2048

    polygons, polygons_label = inter_polyer.extract_inter_polygons()
    tba_xmls = ''
    for i in range(len(polygons)):
        label, polygon = polygons_label[i], polygons[i]
        color = color_map[label].astype(np.int)
        poly_str = '<polygon points="'+''.join(["{:.3f},{:.3f} ".format(_VT[v][0], _VT[v][1]) for v in polygon])
        poly_str += '" fill="'+'#%02x%02x%02x' % (color[0], color[1], color[2])+ '" stroke="white" stroke-width="8"/>\n'
        tba_xmls += poly_str

    return _F, _FL, interpoly_matrix, tba_xmls, inter_polyer.m


def GenerateTBATexture(V, F, FL, VT):
    pass 
    V, F = cpc_mesh.points(), cpc_mesh.face_vertex_indices()
    mat = sio.loadmat("/Users/hotpot/Code/TBA/TPenPy3/Data/TBA/Atlas.mat")
    for tba_name in  ["lpba", "aal", "ba"]:
        VL = np.ones(cpc_mesh.n_vertices(), dtype=np.int) * -2
        VL[:9801], color_map = mat[tba_name+"_label"].reshape(-1), mat[tba_name+"_color"]

        smoother = Smoother(V, F, VL)
        smoother.subdivision_defined_border(0)
        NV_sub_seg, NF = smoother.sub_segmentation()

        np.savez("./output/"+tba_name+".F", NF)

        FC, FL = smoother.perface_color(color_map)

        np.savez("./output/"+tba_name+".FL", FL)

        om.write_mesh("./output/"+tba_name+".x.off", smoother.m, face_color=True, vertex_color=True)

        smoother.quadric_smooth(n_step=n_steps)
        sparse.save_npz("./output/"+tba_name+".x", smoother.sub_seg_matrix.tocsr())
        
        om.write_mesh("./output/"+tba_name+".smoothed.off", smoother.m, face_color=True, vertex_color=True)
        
if __name__ == "__main__":
    def gen_cpc_coord(inner_sample=99):
        n = inner_sample
        VCPC = np.zeros((n+2,n,2), dtype=np.float32)
        linspace_n = np.linspace(0, 1, n+2)[1:-1].reshape((-1,1))
        VCPC[:n,:n,1], VCPC[:n,:n,0] = np.meshgrid(linspace_n, linspace_n)
        VCPC[-2,:,0], VCPC[-1,:,0] = 1, 0
        VCPC[-2,:,1] = VCPC[-1,:,1] = linspace_n.reshape(-1)
        VCPC = np.vstack([VCPC.reshape((-1,2)), [0.5, 1-1e-3], [0.5, 1e-3]])
        return VCPC

    def convert_to_uv(cpc):
        # (u, v)
        # v = (std::sin(u*3.1415926))*(v-0.5)+0.5;
        # m.set_texcoord2D(vh, TriMesh::TexCoord2D(1-u, v));
        uv = np.zeros(cpc.shape, dtype=np.float)
        for i, row in enumerate(cpc):
            uv[i, 1] = np.sin(row[1]*np.pi)*(row[0]-0.5)+0.5
            uv[i, 0] = 1-row[1]
        return uv 

    n_steps = 1
    
    cpc_mesh = om.read_trimesh("./data/lsmesh-cpc.obj")
    V, F = cpc_mesh.points(), cpc_mesh.face_vertex_indices()
    VT = convert_to_uv(gen_cpc_coord(99))
    mat = sio.loadmat("./data/Atlas.mat")
    for tba_name in  ["lpba", "aal", "ba"]:
        VL = np.ones(cpc_mesh.n_vertices(), dtype=np.int) * -2
        print(mat[tba_name+"_label"].shape, mat[tba_name+"_color"].shape)

        VL[:9801], color_map = mat[tba_name+"_label"].reshape(-1), mat[tba_name+"_color"]
        mesh1  = "../runtime/tba/"+tba_name+".prev.off"
        mesh2  = "../runtime/tba/"+tba_name+".smoothed.off"
        _F, _FL, x, poly_xmls, interpoly_Mesh = LabeledScalpSmoothing(V, VT, F, VL, color_map, n_steps, mesh1, mesh2)

        np.savez("../runtime/tba/"+tba_name+".F", _F)
        np.savez("../runtime/tba/"+tba_name+".FL", _FL)
        sparse.save_npz("../runtime/tba/"+tba_name+".x", x.tocsr())
        print(poly_xmls, file=open("../runtime/tba/"+tba_name+".xml", "w"))    

