'''
Author: Pupa
LastEditTime: 2020-12-03 18:07:34
'''
import numpy as np
import openmesh as om 
import scipy.io as sio
import scipy.sparse as sparse
import cvxpy as cp 
import numpy as np 

def _convert_to_trimesh(V, F, VL) -> om.TriMesh:
    m = om.TriMesh()
    for i, e in enumerate(V):
        m.set_vertex_property("cur_v:label", m.add_vertex(e), VL[i])
    for e in F:
        m.add_face([m.vertex_handle(e[0]), m.vertex_handle(e[1]), m.vertex_handle(e[2])])
    return m


'''Deprecated Implementation'''
class InterPolyMesh():
    def __init__(self, V, F, VL):
        self.m = _convert_to_trimesh(V, F, VL)

    def perface_coloring(self, color_map):
        m = self.m 
        m.request_face_colors()
        m.request_vertex_colors()
        FL = np.zeros(m.n_faces())
        
        for f in m.faces():
            h = m.halfedge_handle(f)
            vh0, vh1, vh2 = m.from_vertex_handle(h), m.to_vertex_handle(h), m.opposite_vh(h)
            max_v_label = np.max([m.vertex_property("cur_v:label", v) for v in [vh0, vh1, vh2]])
            m.set_color(f, np.append(color_map[max_v_label, :], 1))
            FL[f.idx()] = max_v_label
        for v in m.vertices():
            v_label = m.vertex_property("cur_v:label", v)
            if v_label > -1:  m.set_color(v, np.append(color_map[v_label, :], 1))
            else : m.set_color(v, np.ones(4))
        return m.face_colors(), FL 

    def subdivide_region_boundary(self, n_subd=1):
        m = self.m 
        avg_len = np.mean([m.calc_edge_length(e) for e in m.edges()])/(2**n_subd)
        for subd_i in range(n_subd): 
            for i, (vh0, vh1) in enumerate(m.halfedge_vertex_indices()):
                if m.is_boundary(m.halfedge_handle(i)): continue
                vh0, vh1 = m.vertex_handle(vh0), m.vertex_handle(vh1)
                (vl0, vl1, vl2) = (m.vertex_property("cur_v:label", v) for v in (vh0, vh1, m.opposite_vh(m.halfedge_handle(i)) ) )
                if vl1 == vl0 and vl2 != vl0 and m.calc_edge_length(m.edge_handle(m.halfedge_handle(i))) > avg_len: 
                    p0, p1 = m.point(vh0), m.point(vh1)
                    v_new = m.split(m.edge_handle(m.halfedge_handle(i)), (p0+p1)/2)
                    m.set_vertex_property("cur_v:label", v_new, vl0)

    def extract_region_graph_boundary(self):
        m = self.m 
        n_vertices = m.n_vertices()
        interpoly_matrix = sparse.eye(n_vertices, format="dok")
        # split faces
        for e in m.edges():
            labels = self._n_labels_diag(e)
            if len(labels[np.where( labels >= 0)]) == 4:
                h = m.halfedge_handle(e, 0)
                
                interpoly_matrix.resize((m.n_vertices()+1, n_vertices))
                interpoly_matrix[m.n_vertices(), m.from_vertex_handle(h).idx()] = 0.5
                interpoly_matrix[m.n_vertices(), m.to_vertex_handle(h).idx()] = 0.5  

                nvh = m.add_vertex((m.point(m.from_vertex_handle(h))+m.point(m.to_vertex_handle(h)))/2)
                m.set_vertex_property("cur_v:label", nvh, -1)
                m.split(e, nvh)
               

        # 2. split edge
        for i, ei in enumerate(m.edge_vertex_indices()):
            vh0, vh1 = m.vertex_handle(ei[0]), m.vertex_handle(ei[1])
            vl0, vl1 = m.vertex_property("cur_v:label", vh0), m.vertex_property("cur_v:label", vh1)
            if np.min([vl0, vl1]) >= 0 and vl0 != vl1:
                interpoly_matrix.resize((m.n_vertices()+1, n_vertices))
                interpoly_matrix[m.n_vertices(), vh0.idx()] = 0.5
                interpoly_matrix[m.n_vertices(), vh1.idx()] = 0.5  
                
                nvh = m.add_vertex((m.point(vh0)+m.point(vh1))/2)
                m.split(m.edge_handle(i), nvh)
                m.set_vertex_property("cur_v:label", nvh, -1)
                m.set_vertex_property("cur_v:barycoord", nvh, (vh0, vh1))
                
        # 3. flip topology optimization  for small border angle
        for h in m.halfedges():
            vf, vt = m.from_vertex_handle(h), m.to_vertex_handle(h)
            vh1, vh2 = m.opposite_vh(h), m.opposite_he_opposite_vh(h)
            if m.vertex_property("cur_v:label", vf) == -1 and m.vertex_property("cur_v:label", vt) != -1 \
                and m.vertex_property("cur_v:label", vh1) == -1 and m.vertex_property("cur_v:label", vh2) == -1 :
                angle_t = m.calc_sector_angle(h) + m.calc_sector_angle(m.prev_halfedge_handle(m.opposite_halfedge_handle(h)))
                angle_f = m.calc_sector_angle(m.prev_halfedge_handle(h)) + m.calc_sector_angle(m.opposite_halfedge_handle(h))
                angle_1 = m.calc_sector_angle(m.next_halfedge_handle(h))
                angle_2 = m.calc_sector_angle(m.next_halfedge_handle(m.opposite_halfedge_handle(h)))
                if angle_f + angle_t < np.pi * 0.67: 
                    m.flip(m.edge_handle(h ))
        # post flip
        for h in m.halfedges():
            vf, vt = m.from_vertex_handle(h), m.to_vertex_handle(h)
            vh1, vh2 = m.opposite_vh(h), m.opposite_he_opposite_vh(h)
            if m.vertex_property("cur_v:label", vf) != -1 or m.vertex_property("cur_v:label", vt) != -1: continue
            if np.sign(m.vertex_property("cur_v:label", vh1)) *  np.sign(m.vertex_property("cur_v:label", vh2)) <= 0 :
                angle_t = m.calc_sector_angle(h) + m.calc_sector_angle(m.prev_halfedge_handle(m.opposite_halfedge_handle(h)))
                angle_f = m.calc_sector_angle(m.prev_halfedge_handle(h)) + m.calc_sector_angle(m.opposite_halfedge_handle(h))
                angle_1 = m.calc_sector_angle(m.next_halfedge_handle(h))
                angle_2 = m.calc_sector_angle(m.next_halfedge_handle(m.opposite_halfedge_handle(h)))
                if angle_1 + angle_2 > np.pi * 0.67:
                    m.flip(m.edge_handle(h ))

        self.interpoly_matrix = interpoly_matrix 
        return self.m.points(), self.m.face_vertex_indices()

    def _n_labels(self, vh: om.VertexHandle):
        return np.unique([self.m.vertex_property("cur_v:label", v) for v in self.m.vv(vh)])
    
    def _n_labels(self, eh: om.EdgeHandle):
        h1, h2 = self.m.halfedge_handle(eh, 0 ), self.m.halfedge_handle(eh, 1)
        labels = [self.m.vertex_property("cur_v:label", self.m.from_vertex_handle(h)) for h in (h1, h2)]
        return np.unique(labels)

    def _n_labels_diag(self, eh: om.EdgeHandle):
        h1, h2 = self.m.halfedge_handle(eh, 0 ), self.m.halfedge_handle(eh, 1)
        labels = [self.m.vertex_property("cur_v:label", self.m.from_vertex_handle(h)) for h in (h1, h2)]
        if self.m.opposite_vh(h1).is_valid():
            labels.append(self.m.vertex_property("cur_v:label", self.m.opposite_vh(h1)))
        if self.m.opposite_vh(h2).is_valid():
            labels.append(self.m.vertex_property("cur_v:label", self.m.opposite_vh(h2)))
        return np.unique(labels)

    def _boundary_divide_and_packing(self, n_var=500):
        m = self.m
        # pre-labeling
        for v in m.vertices(): 
            if m.vertex_property("cur_v:label", v) != -1: continue # super border pixels
            cur_neighbors = []
            for vv in m.vv(v):
                if m.vertex_property("cur_v:label", vv) > -1: continue
                cur_neighbors.append(vv)
            if len(cur_neighbors) == 2:    
                m.set_vertex_property("cur_v:neighbor", v, cur_neighbors)
                m.set_vertex_property("cur_v:smoothed", v, False)
                assert(len(cur_neighbors) == 2)
            elif len(cur_neighbors) > 2:
                m.set_vertex_property("cur_v:label", v, -2)
        
        print("dividing for closed boundary")
        inter_polygons = []
        for v in m.vertices():
            if m.vertex_property("cur_v:label", v) != -2: continue # super border vertex
            m.set_vertex_property("cur_v:smoothed", v, True)
            for vv in m.vv(v):
                if m.vertex_property("cur_v:label", vv) != -1: continue
                if m.vertex_property("cur_v:smoothed", vv): continue
                segs = []
                while m.vertex_property("cur_v:label", vv) == -1 and m.vertex_property("cur_v:smoothed", vv) == False:
                    m.set_vertex_property("cur_v:smoothed", vv, True)
                    segs.append(vv.idx())
                    for vvv in m.vv(vv):
                        if m.vertex_property("cur_v:label", vvv) >= 0: continue
                        if m.vertex_property("cur_v:smoothed", vvv) : continue
                        vv = vvv
                        
                if len(segs) == 1: continue
                if len(inter_polygons) and  len(segs) + len(inter_polygons[-1]) < n_var:
                    inter_polygons[-1] += segs
                else: inter_polygons.append(segs)
      
        print("dividing for open boundary")
        for v in m.vertices():
            if m.vertex_property("cur_v:label", v) != -1: continue 
            if m.vertex_property("cur_v:smoothed", v): continue
            segs = []
            while m.vertex_property("cur_v:label", v) == -1:
                m.set_vertex_property("cur_v:smoothed", v, True)
                segs.append(v.idx())
                for vv in m.vv(v):
                    if m.vertex_property("cur_v:label", vv) >= 0: continue
                    if m.vertex_property("cur_v:smoothed", vv) : continue
                    v = vv
                    break
                if m.vertex_property("cur_v:smoothed", vv) : break 
            if len(segs) == 1: continue
            if len(inter_polygons) and  len(segs) + len(inter_polygons[-1]) < n_var:
                inter_polygons[-1] += segs
            else: inter_polygons.append(segs)
            # print(len(segs), segs)
            
        print("post checking")
        for v in m.vertices():
            if m.vertex_property("cur_v:label", v) == -1:
                assert(m.vertex_property("cur_v:smoothed", v))
            
        return inter_polygons

    def minimize_quadric_energy(self, segment, n_step=5, alpha=1.0):
        m = self.m 
        X = [cp.Variable() for i in range(len(segment))]
        X_constraints = [x >= 0.01 for x in X]+[x <= 1-0.01 for x in X]
        for x in X: x.value = 0.5

        # vertex mapping 
        X_v = np.ones( m.n_vertices() , dtype=np.int) * -1
        V_x = []
        for v_idx in segment:
            if m.vertex_property("cur_v:label", m.vertex_handle(v_idx)) != -1: continue
            X_v[v_idx] = len(V_x)
            V_x.append(v_idx )
            
        # print("Calc QP coefficient: ", len(V_x), len(X))

        v_neighbors = m.vertex_property("cur_v:neighbor")
        v_barycenter = m.vertex_property("cur_v:barycoord")
        E1, E2 = 0, 0
        for i1, vh1 in enumerate( segment ):
            vh0, vh2 = v_neighbors[vh1]
            x1 = X[i1]
            p1 = x1*m.point(v_barycenter[vh1][1])+(1-x1)*m.point(v_barycenter[vh1][0])
            p0, p2 = m.point(vh0), m.point(vh2)
            if X_v[vh0.idx()] != -1:
                x0 = X[X_v[vh0.idx()]]
                p0 = x0*m.point(v_barycenter[vh0.idx()][1])+(1-x0)*m.point(v_barycenter[vh0.idx()][0])
            if X_v[vh2.idx()] != -1:
                x2 = X[X_v[vh2.idx()]]
                p2 = x2*m.point(v_barycenter[vh2.idx()][1])+(1-x2)*m.point(v_barycenter[vh2.idx()][0])
            E1 += cp.norm(p0-p1) + cp.norm(p2-p1)
            # l0, l2 = 1/np.clip(cp.norm(p0-p1).value, 0.01, 0.99), 1/np.clip(cp.norm(p2-p1).value, 0.01, 0.99)
            # E2 += cp.sum_squares((p0 * l0 +p2 * l2 -p1*(l0+l2)) )
            # E2 += cp.sum_squares(p0 + p2  -p1*2 )
            
        for smooth_i in range(n_step):
            prev_x = np.array([x.value for x in X])
            E2 = 0
            for i1, vh1 in enumerate( segment ):
                vh0, vh2 = v_neighbors[vh1]
                x1 = X[i1]
                p1 = x1*m.point(v_barycenter[vh1][1])+(1-x1)*m.point(v_barycenter[vh1][0])
                p0, p2 = m.point(vh0), m.point(vh2)
                if X_v[vh0.idx()] != -1:
                    x0 = X[X_v[vh0. idx()]]
                    p0 = x0*m.point(v_barycenter[vh0.idx()][1])+(1-x0)*m.point(v_barycenter[vh0.idx()][0])
                if X_v[vh2.idx()] != -1:
                    x2 = X[X_v[vh2.idx()]]
                    p2 = x2*m.point(v_barycenter[vh2.idx()][1])+(1-x2)*m.point(v_barycenter[vh2.idx()][0])
                # E1 += cp.norm(p0-p1) + cp.norm(p2-p1)
                l0, l2 = 1/np.clip(cp.norm(p0-p1).value, 0.01, 0.99), 1/np.clip(cp.norm(p2-p1).value, 0.01, 0.99)
                E2 += cp.sum_squares((p0 * l0 +p2 * l2 -p1*(l0+l2))/(l0+l2) ) # 
                # E2 += cp.sum_squares(p0 + p2  -p1*2 )
            prob = cp.Problem(cp.Minimize(E1*0+E2*alpha), X_constraints)
            assert(prob.is_dqcp())
            try: prob.solve("ECOS")
            except Exception as e: print(e)

            # print("The optimal value:", prob.value,  E1.value, E2.value)
            # print("problem setup_time: ", prob.solver_stats.setup_time, "s")
            # print("problem solve_time: ", prob.solver_stats.solve_time, "s")
            # print("problem num_iters: ", prob.solver_stats.num_iters)
            next_x = np.array([x.value for x in X])
            print("SmoothStep {}: {:.3f} = {:.3f} + {:.3f} ===> {:.3f}".format(smooth_i, prob.value, E1.value, E2.value,np.max(np.abs(next_x - prev_x))))
            if np.max(np.abs(next_x - prev_x)) < 0.015: break   

        for i, x in enumerate(X):
            assert(-1e-6 <= x.value <= 1)
            p1 = x.value*m.point(v_barycenter[V_x[i]][1])+(1-x.value)*m.point(v_barycenter[V_x[i]][0])
            m.set_point(m.vertex_handle(V_x[i]), p1)
            self.interpoly_matrix[V_x[i], v_barycenter[V_x[i]][1].idx()] = x.value
            self.interpoly_matrix[V_x[i], v_barycenter[V_x[i]][0].idx()] = 1-x.value

    def extract_inter_poly(self):
        m = self.m 
        polygons = []
        for h in m.halfedges(): m.set_halfedge_property("visited", h, False)
        for h in m.halfedges():
            if m.halfedge_property("visited", h) or m.is_boundary(h): continue
            vf, vt = m.from_vertex_handle(h), m.to_vertex_handle(h)
            if m.vertex_property("cur_v:label", vf) >= 0 or m.vertex_property("cur_v:label", vt) >= 0 : continue
            poly_label = m.vertex_property("cur_v:label", m.opposite_vh(h))
            polygon = []
            while m.halfedge_property("visited", h) == False:
                polygon.append(m.to_vertex_handle(h).idx())
                m.set_halfedge_property("visited", h, True)
                for hh in m.voh(m.to_vertex_handle(h)):
                    if m.is_boundary(hh) or m.vertex_property("cur_v:label", m.to_vertex_handle(hh)) >= 0: continue
                    if m.halfedge_property("visited", hh) or  m.vertex_property("cur_v:label", m.opposite_vh(hh)) != poly_label: continue
                    h = hh 
                    break
            polygons.append((poly_label, polygon))
        return polygons

    def quadric_smoothing(self, n_step=10):
        print("quadric_smoothing")
        # build curve neighbors info
        inter_polygons = self._boundary_divide_and_packing()
        for i, curve_component in enumerate(inter_polygons):
            print("Solving sub smooth: ", i, "/", len(inter_polygons))
            self.minimize_quadric_energy(curve_component, n_step)
