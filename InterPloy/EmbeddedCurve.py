import numpy as np
import openmesh as om 
import queue
import scipy.sparse as sparse 
import scipy.sparse.linalg as sparse_linalg


# Helper class for the query about the embedding info
class HostMesh(om.PolyMesh):
    def __init__(self, V, F):
        super().__init__()
        list(map(self.add_vertex, V))
        for e in F:
            self.add_face(list(map(self.vertex_handle, e)))
    
    def set_embedd_triple(self, host_h: om.HalfedgeHandle, em_v: om.VertexHandle, w: float): 
        self.set_halfedge_property("h:em_v", host_h, em_v )
        self.set_halfedge_property("h:em_w", host_h, w )

    def embedded_triple(self, host_h: om.HalfedgeHandle):
        host_vf = self.from_vertex_handle(host_h)
        host_vt = self.to_vertex_handle(host_h)
        return host_vf, host_vt, self.vertex_property("h:em_w", host_h)
    
    def normalize(self, host_h: om.HalfedgeHandle, w: float):
        if host_h.idx() & 1:
            host_h = self.opposite_halfedge_handle(host_h)
            w = 1 - w
        return host_h, w, self.halfedge_property("h:em_v", host_h)

class EmbeddedCurveMesh(om.PolyMesh):
    def __init__(self, mesh: HostMesh):
        super().__init__()
        self.host_mesh = mesh
        list(map(self.add_vertex, mesh.points()))
        for e in mesh.face_vertex_indices():
            self.add_face(list(map(self.vertex_handle, e)))
        self._curves = []

    def is_vertex_embedded(self, v: om.VertexHandle):
        return self.vertex_property("v:host_h", v) is not None
    
    def is_halfedge_embedded(self, h: om.HalfedgeHandle):
        return self.is_vertex_embedded(self.from_vertex_handle(h)) and \
               self.is_vertex_embedded(self.to_vertex_handle(h)) 
    
    def next_em_halfedge(self, h: om.HalfedgeHandle):
        assert(self.is_halfedge_embedded(h))
        opposite_h = self.opposite_halfedge_handle(h)
        for voh in self.voh(self.to_vertex_handle(h)):
            if self.is_halfedge_embedded(voh) and voh != opposite_h:
                return voh
        return om.HalfedgeHandle(-1)

    def prev_em_halfedge(self, h: om.HalfedgeHandle):
        assert(self.is_halfedge_embedded(h))
        opposite_h = self.opposite_halfedge_handle(h)
        for vih in self.vih(self.from_vertex_handle(h)):
            if self.is_halfedge_embedded(vih) and vih != opposite_h:
                return vih
        return om.HalfedgeHandle(-1)

    def __add_embedd_vertex(self, h: om.HalfedgeHandle, w: float):
        h, w, em_vertex = self.host_mesh.normalize(h, w)
        if em_vertex: return em_vertex
        vf, vt = self.from_vertex_handle(h), self.to_vertex_handle(h)
        em_v = self.add_vertex(self.point(vf)*w + self.point(vt)*(1-w))
        self.split_edge(self.edge_handle(h), em_v)
        self.host_mesh.set_embedd_triple(h, em_v, w)
        self.set_vertex_property("v:host_h", em_v, h)
        return em_v
    
    def add_embedd_edge(self, host_h1: om.HalfedgeHandle, w1: float, 
                              host_h2: om.HalfedgeHandle, w2: float):
        sector_v = self.host_mesh.to_vertex_handle(host_h1)
        em_v1 = self.__add_embedd_vertex(host_h1, w1)
        em_v2 = self.__add_embedd_vertex(host_h2, w2)
        em_h1 = self.find_halfedge(em_v1, sector_v)
        em_h2 = self.find_halfedge(sector_v, em_v2)
        return self.insert_edge(em_h2, em_h1)

    def extract_em_curves(self):
        _curves = []
        for h in self.halfedges():
            if self.is_halfedge_embedded(h) and self.halfedge_property("h:visited", h) is None:
                self.set_halfedge_property("h:visited", h, True)
                p_h, n_h = self.prev_em_halfedge(h), self.next_em_halfedge(h)

                while n_h.is_valid() and self.halfedge_property("h:visited", n_h) is None:
                    self.set_halfedge_property("h:visited", n_h, True)
                    n_h = self.next_em_halfedge(n_h)
                while p_h.is_valid() and self.halfedge_property("h:visited", p_h) is None:
                    self.set_halfedge_property("h:visited", p_h, True)
                    p_h = self.prev_em_halfedge(p_h)
                    h = p_h
                _curves.append(h)
        self.remove_halfedge_property("h:visited")
        return _curves

    def curve_halfedges(self, curve_begin: om.HalfedgeHandle):
        assert(self.is_halfedge_embedded(curve_begin))
        yield curve_begin
        h = self.next_em_halfedge(curve_begin)
        while h.is_valid() and h != curve_begin:
            yield h
            h = self.next_em_halfedge(h)

    def em_vertices(self):
        for v in self.vertices():
            if self.is_embedded(v): yield v
        
    def host_v(self, v: om.VertexHandle):
        host_h = self.vertex_property("v:host_h", v) 
        assert(host_h is not None)
        return self.host_mesh.from_vertex_handle(host_h), self.host_mesh.to_vertex_handle(host_h)
    
    def host_w(self, v: om.VertexHandle):
        host_h = self.vertex_property("v:host_h", v) 
        assert(host_h is not None)
        return self.host_mesh.halfedge_property("h:em_w", host_h)

    def set_host_w(self, v: om.VertexHandle, w:float):
        host_h = self.vertex_property("v:host_h", v) 
        assert(host_h is not None)
        vf, vt = self.host_mesh.from_vertex_handle(host_h), self.host_mesh.to_vertex_handle(host_h)
        self.host_mesh.set_halfedge_property("h:em_w", host_h, w)
        self.set_point(v, self.em_point(v, w))

    def em_point(self, v: om.VertexHandle, x: float):
        assert(self.vertex_property("v:host_h", v) is not None)
        p_f, p_t = map(curve_mesh.point, curve_mesh.host_v(v))
        return (p_f+p_t)/2 + (p_t - p_f) * x

    def calc_curve_sector_angle(self, h: om.HalfedgeHandle, t_x:float = 1.0):
        assert(self.is_halfedge_embedded(h))
        if t_x == 1.0: t_x = self.host_w(self.to_vertex_handle(h))
        pf, pt = self.point(self.from_vertex_handle(h)), self.em_point(self.to_vertex_handle(h), t_x)
        if self.next_em_halfedge(h).is_valid() is False: return 0
        ptt = self.point(self.to_vertex_handle(self.next_em_halfedge(h)))
        length = np.linalg.norm(pt-pf)*np.linalg.norm(ptt-pt)
        angle = np.arccos(np.clip(np.dot(pt-pf, ptt-pt)/length, -1.0, 1.0))
        if np.dot(np.cross(pt-pf, ptt-pt), self.calc_halfedge_normal(h)) < 0: return -angle
        else: return angle
    
    def calc_curve_vertex_local_area(self, h: om.HalfedgeHandle, t_x:float = 1.0):
        assert(self.is_halfedge_embedded(h))
        if t_x == 1.0: t_x = self.host_w(self.to_vertex_handle(h))
        pf, pt = self.point(self.from_vertex_handle(h)), self.em_point(self.to_vertex_handle(h), t_x)
        ptt = self.point(self.to_vertex_handle(self.next_em_halfedge(h)))
        return (np.linalg.norm(pf-pt)+ np.linalg.norm(pt-ptt))/2
    
    def calc_curve_sector_curvature(self, h: om.HalfedgeHandle, t_x:float = 1.0):
        '''curvature variation of vt - vf'''
        curr_angle = self.calc_curve_sector_angle(h, t_x)
        curr_A = self.calc_curve_vertex_local_area(h, t_x)
        return curr_angle/curr_A
    

def extract_embedded_matrix(curve_mesh: EmbeddedCurveMesh):
    n_old = curve_mesh.host_mesh.n_vertices()
    n_new = curve_mesh.n_vertices() - n_old
    EM = sparse.eye(n_old+n_new, n=n_old, format="dok")
    for v in curve_mesh.vertices():
        if curve_mesh.is_vertex_embedded(v) is False: continue
        vf, vt = curve_mesh.host_v(v)
        w = curve_mesh.host_w(v)
        # v = (vf+vt)/2 + w*(vt-vf)
        EM[v.idx(). vf.idx()] = 0.5 - w
        EM[v.idx(). vt.idx()] = 0.5 + w
    return EM, curve_mesh.face_vertex_indices()
    #sparse.save_npz("./output/"+tba_name+".x", smoother.sub_seg_matrix.tocsr())


# for closed curve
def curve_length_minimize(curve_mesh: EmbeddedCurveMesh, curve_begin: om.HalfedgeHandle, b: float = 0.45):
    import cvxpy as cp 
    X, Vs = [], []
    for c_h in curve_mesh.curve_halfedges(curve_begin):
        Vs.append(curve_mesh.to_vertex_handle(c_h))
        curve_mesh.set_vertex_property("v:idx", Vs[-1], len(X))
        X.append(cp.Variable())
        X[-1].value = 0

    X_constraints = [x >= -b for x in X]+[x <= b for x in X]
    for x in X: x.value = 0
    Energy = 0
    for i, c_h in enumerate(curve_mesh.curve_halfedges(curve_begin)):
        vf, vt = curve_mesh.from_vertex_handle(c_h), curve_mesh.to_vertex_handle(c_h)
        vtt = curve_mesh.to_vertex_handle(curve_mesh.next_em_halfedge(c_h))

        pf = curve_mesh.em_point(vf, X[curve_mesh.vertex_property("v:idx", vf)])
        pt = curve_mesh.em_point(vt, X[curve_mesh.vertex_property("v:idx", vt)])
        ptt = curve_mesh.em_point(vtt, X[curve_mesh.vertex_property("v:idx", vtt)])

        Energy += cp.norm(pt-pf)+cp.norm(ptt-pt)
    problem = cp.Problem(cp.Minimize(Energy), X_constraints)
    assert(problem.is_dqcp())
    try: problem.solve("ECOS")
    except Exception as e: print(e)
    for i, em_vertex in enumerate(Vs):
        curve_mesh.set_host_w(em_vertex , X[i].value)


def curvature_variation_minimize(curve_mesh: EmbeddedCurveMesh, curve_begin: om.HalfedgeHandle):
    total_angle = np.sum([curve_mesh.calc_curve_sector_angle(c_h) for c_h in curve_mesh.curve_halfedges(curve_begin)])
    def total_length():
        return np.sum([curve_mesh.calc_curve_vertex_local_area(c_h) for c_h in curve_mesh.curve_halfedges(curve_begin)])
    print("total_length ", total_length())
    def line_search(h: om.HalfedgeHandle, k):
        lr = [-0.49, 0.49]
        if curve_mesh.calc_curve_sector_curvature(h, lr[0]) > curve_mesh.calc_curve_sector_curvature(h, lr[1]):
            lr[1], lr[0] = lr[0], lr[1]
        while np.abs(lr[0]-lr[1]) > 1e-10:
            angle_l = curve_mesh.calc_curve_sector_curvature(h, lr[0])
            angle_r = curve_mesh.calc_curve_sector_curvature(h, lr[1])
            angle_m = curve_mesh.calc_curve_sector_curvature(h, np.mean(lr))
            if angle_m > k: lr[1] = np.mean(lr)
            else: lr[0] = np.mean(lr)
        # print("Line search ", curve_mesh.calc_curve_sector_curvature(h, lr[1]), k)
        return np.mean(lr)


    for i_iter in range(50): 
        mean_k = total_angle/total_length()
        is_convergent = True
        for c_h in curve_mesh.curve_halfedges(curve_begin):
            curr_k = curve_mesh.calc_curve_sector_curvature(c_h)
            if np.abs(curr_k-mean_k) > mean_k/100:  
                # moving vt on it's host edge
                x_ = line_search(c_h, curr_k+(mean_k-curr_k)*0.8 )
                curve_mesh.set_host_w(curve_mesh.to_vertex_handle(c_h), x_)
                is_convergent = False


        dis = []
        for c_h in curve_mesh.curve_halfedges(curve_begin):
            dis.append(curve_mesh.calc_curve_sector_curvature(c_h)-mean_k)
        print("iter:{}/{} : {}".format(i_iter, 50, np.mean(np.square(dis))) )#, end="\r")
        if is_convergent: break
                


if __name__ == "__main__":
    def gen_circle_test(n_sample=25):
        linspace = np.linspace(-1, 1, n_sample)
        x, y = np.meshgrid(linspace, linspace)
        V = np.zeros((n_sample, n_sample, 3))
        V[:, :, 0], V[:, :, 1] = x, y

        V = V.reshape((-1,3))
        VL = np.ones(len(V), dtype=np.int8)
        VL[np.where(np.linalg.norm(V, axis=1) <= 0.8)] = 2
        F = np.zeros((n_sample-1, n_sample-1, 2, 3), dtype=np.uint16)
        for i in range(n_sample - 1):
            for j in range(n_sample - 1):
                F[i,j,0] = np.array([i *  n_sample + j, i * n_sample + j + 1, (i + 1) * n_sample + j + 1], dtype=np.uint16)
                F[i,j,1] = np.array([(i + 1) * n_sample + j + 1, (i + 1) * n_sample + j, i * n_sample + j], dtype=np.uint16)   
        return V, VL, F.reshape((-1,3))  
    
    V, VL, F = gen_circle_test(30)   

    host_mesh = HostMesh(V, F)
    curve_mesh = EmbeddedCurveMesh(host_mesh)
    for h in host_mesh.halfedges():
        vf, vt = host_mesh.from_vertex_handle(h), host_mesh.to_vertex_handle(h)
        vtt = host_mesh.to_vertex_handle(host_mesh.next_halfedge_handle(h))
        if VL[vf.idx()] != VL[vt.idx()] != VL[vtt.idx()] :
            curve_mesh.add_embedd_edge(h, 0.5, host_mesh.next_halfedge_handle(h), 0.5).idx()

    color_map = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0]])
    curve_mesh.request_vertex_colors()
    curve_mesh.request_face_colors()
    for i in range(len(V)):
        curve_mesh.set_color(curve_mesh.vertex_handle(i), np.append(color_map[VL[i],:], 1))
        for f in curve_mesh.vf(curve_mesh.vertex_handle(i)):
            curve_mesh.set_color(f, np.append(color_map[VL[i],:], 1))
    for i in range(len(V), curve_mesh.n_vertices()):
        curve_mesh.set_color(curve_mesh.vertex_handle(i), (1, 1, 1, 1))
    om.write_mesh("1.off", curve_mesh, vertex_color=True, face_color=True)

    for curve in curve_mesh.extract_em_curves():
        curve_length_minimize(curve_mesh, curve, 0.35)
        curvature_variation_minimize(curve_mesh, curve)
        break


    om.write_mesh("smoothed.off", curve_mesh, vertex_color=True, face_color=True)



