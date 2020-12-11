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
        color = (color_map[label]*255).astype(np.int)
        poly_str = '<polygon points="'+''.join(["{:.3f},{:.3f} ".format(_VT[v][0], _VT[v][1]) for v in polygon])
        poly_str += '" fill="'+'#%02x%02x%02x' % (color[0], color[1], color[2])+ '" stroke="white" stroke-width="8"/>\n'
        tba_xmls += poly_str

    return _F, _FL, interpoly_matrix, tba_xmls, inter_polyer.m


# tool functions
def gen_texture_png(axis, l1020, mark, texture_name, atlas_svg_str=""):
    import cairosvg
    from PIL import Image
    import os

    svg_header = '<svg width="2048" height="2048" viewBox="0 0 2048 2048" xmlns="http://www.w3.org/2000/svg" >'
    style_header = '<style type="text/css" >'
    l1020_style = '''text.mark { fill: #ffffffff; font-size: 50px; stroke: #000000ff; stroke-width: 1.5;}
                    circle.landmark {r: 20; fill: #ff0000ff; stroke: black; stroke-width: 2;}
                    circle.mark { r: 15;fill: #e5e599ff;stroke: black;stroke-width: 2;}'''
    axis_style = '''polyline.line {fill: None;stroke: #f2f2f2ff;stroke-width: 3;}
                    polyline.m_mark {fill: None;stroke: #333333ff;stroke-width: 5;}
                    polyline.outer_mark {fill: #ffccb2ff;stroke: #333333ff;stroke-width: 10;}
                    polyline.axis {fill: None;stroke: #333333ff;stroke-width: 6;}'''

    if atlas_svg_str != "":
        axis_style = axis_style.replace("#ffccb2ff", "None")  

    svg_string = svg_header + style_header + l1020_style + axis_style + "</style>\n" + atlas_svg_str
    svg_string += open("./data/tba_xml/axis_uv.xml").read() * axis
    svg_string += open("./data/tba_xml/1020.xml").read() * l1020
    svg_string += open("./data/tba_xml/mark.xml").read() * mark
    svg_string += "</svg>"  

    outpng = "../runtime/tba/png/"+ texture_name + (".1020." if l1020 else '.') + ("axis." if axis else '')+ ("text." if mark else '') + "png"
    
    cairosvg.svg2png(bytestring=svg_string, write_to=open(outpng, 'wb')) 
    Image.open(outpng).convert("RGB").save(outpng.replace("png", "jpg"), quality=100, format='JPEG')
    print(svg_string, file=open(outpng.replace("png", "svg"), "w"))
    # return  outpng


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

    n_steps = 20
    
    cpc_mesh = om.read_trimesh("./data/lsmesh-cpc.obj")
    V, F = cpc_mesh.points(), cpc_mesh.face_vertex_indices()
    VT = convert_to_uv(gen_cpc_coord(99))
    mat = sio.loadmat("./data/Atlas.mat")
    for tba_name in  ["lpba", "aal", "ba"]:
        VL = np.ones(cpc_mesh.n_vertices(), dtype=np.int) * -2
        print(mat[tba_name+"_label"].shape, mat[tba_name+"_color"].shape)

        VL[:9801], color_map = mat[tba_name+"_label"].reshape(-1), mat[tba_name+"_color"]
        mesh1  = "../runtime/tba/off/"+tba_name+".prev.off"
        mesh2  = "../runtime/tba/off/"+tba_name+".smoothed.off"
        _F, _FL, x, poly_xmls, interpoly_Mesh = LabeledScalpSmoothing(V, VT, F, VL, color_map, n_steps, mesh1, mesh2)

        np.savez("../runtime/tba/"+tba_name+".F", _F)
        np.savez("../runtime/tba/"+tba_name+".FL", _FL)
        sparse.save_npz("../runtime/tba/"+tba_name+".x", x.tocsr())
        print(poly_xmls, file=open("../runtime/tba/xml/"+tba_name+".xml", "w"))    

        for i in (4+2+1, 4+2, 4+1, 4):
            gen_texture_png(i&4, i&2, i&1, tba_name, poly_xmls)

