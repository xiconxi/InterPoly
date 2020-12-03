# InterPoly

### Overview

<img src="https://github.com/xiconxi/InterPoly/blob/main/imgs/RawPolyCurves.png" alt="alt text" width="400" height="whatever" /> <img src="https://github.com/xiconxi/InterPoly/blob/main/imgs/MinimumVariationPolyCurves.png" alt="alt text" width="400" height="whatever"/>

![](https://github.com/xiconxi/InterPoly/blob/main/imgs/brain_interpoly.gif)

### Install 
```
git clone https://github.com/xiconxi/InterPoly/
ipython
```

### Usage
```python
import InterPoly
import scipy.io as sio 
import numpy as np 
import openmesh as om 
import scipy.sparse as sparse

# V(mesh vertices), F(mesh triangles), VL(per-vertex ROI-label)
inter_polyer = InterPoly.InterPolyMesh(V, F, VL)
# _F == inter_polyer.face_vertex_indices()
# it wouldn't change during the remaining process
_, _F = inter_polyer.extract_region_graph_boundary()
_FC, _VC, _FL, _VL  = inter_polyer.coloring(color_map)
om.write_mesh("../runtime/interpoly_brain.off", inter_polyer.m, face_color=True, vertex_color=True)

# inter_polyer.quadric_smoothing(n_step=2)

interpoly_x = inter_polyer.interpoly_matrix
# x is shared weights for triangle mesh with the same topology(connection)
# x stores the inter polygon results and it could be used to reproduce the results on other meshes.  

# np.savez("./output/brain.F", _F)
# np.savez("./output/brain.FL", _FL)
# sparse.save_npz("./output/brain.x", smoother.interpoly_x.tocsr())
om.write_mesh("../runtime/interpoly_smoothed_brain.off", inter_polyer.m, face_color=True, vertex_color=True)

interpolyer_mesh = {"V": inter_polyer.m.points(), "VL": _VL,"VC": _VC, "F": _F, "FL": _FL, "FC": _FC}
polygons, polygons_label = inter_polyer.extract_inter_polygons()

interpolyer_mesh["boundary_polys"] = np.array(polygons, dtype=object)
interpolyer_mesh["boundary_polys_label"] = polygons_label

sio.savemat("../runtime/interpoly_brain.mat", interpolyer_mesh)

```






