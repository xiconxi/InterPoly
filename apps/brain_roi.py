'''
Author: Pupa
LastEditTime: 2020-12-03 17:49:01
'''

import sys 
sys.path.append("../")

import InterPoly 
import scipy.io as sio 
import numpy as np 
import openmesh as om 

if __name__ == "__main__":
    color_map = sio.loadmat("./data/colorMapLPBA.mat")['ColormapLPBA']
    
    # load from local file
    V, F = sio.loadmat("./data/Adult_ICBM152_GMSurfaceMesh_LPBA.mat")["Adult_ICBM152_GMSurfaceMesh_LPBA"][0][0]
    V, VL, F = V[:,:3], V[:, 3].astype(np.uint8)-1, F-1 

    inter_polyer = InterPoly.InterPolyMesh(V, F, VL)
    interpoly_Mat, interpoly_F = inter_polyer.extract_region_graph_boundary()
    interpoly_C, interpoly_L = inter_polyer.perface_coloring(color_map)
    om.write_mesh("../runtime/interpoly_brain.off", inter_polyer.m, face_color=True, vertex_color=True)
    
    # np.savez("./output/"+tba_name+".F", NF)
    # np.savez("./output/"+tba_name+".FL", FL)

    inter_polyer.quadric_smoothing(n_step=2)

    # sparse.save_npz("./output/"+tba_name+".x", smoother.sub_seg_matrix.tocsr())

    om.write_mesh("../runtime/interpoly_smoothed_brain.off", smoother.m, face_color=True, vertex_color=True)