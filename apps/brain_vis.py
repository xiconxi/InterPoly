import pyvista as pv 
import vtk
import numpy as np
import scipy.io as sio 
import openmesh as om 

def vtk_actor(V, F, FC, opacity=1):
    vtkcells = vtk.vtkCellArray()
    for face in F:
        vtkcells.InsertNextCell(3)
        vtkcells.InsertCellPoint(face[0])
        vtkcells.InsertCellPoint(face[1])
        vtkcells.InsertCellPoint(face[2])

    
    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(pv.utilities.convert_array(V, deep=True))

    vtkPolyData = vtk.vtkPolyData()
    vtkPolyData.SetPoints(vtkpoints)
    vtkPolyData.SetPolys(vtkcells)

    # vtkPolyData.GetPointData().SetTCoords(pv.utilities.convert_array(VT, deep=True))
    c_arr = pv.utilities.convert_array((FC*255).astype(np.uint8), deep=True)
    # c_arr.SetName("Colors")
    vtkPolyData.GetCellData().SetScalars(c_arr)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(vtkPolyData)
    normals.SetFeatureAngle(30.0)
    normals.ComputePointNormalsOn()
    normals.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(normals.GetOutput())
    vtk_actor = vtk.vtkActor()
    vtk_actor.GetProperty().SetOpacity(opacity)
    vtk_actor.GetProperty().SetInterpolationToPBR()

    vtk_actor.SetMapper(mapper)
    return vtk_actor

def save_obj(V, VC, F, file_name):
    with open(file_name, "w") as f_obj:
        np.savetxt(f_obj, np.hstack([V, VC]), fmt="v %f %f %f %f %f %f")
        np.savetxt(f_obj, F+1, fmt="f %d %d %d")

if __name__ == "__main__":
    import sys 

    p = pv.Plotter(shape=(1, 2), border=False, window_size=(600*2, 600*1))


    p.subplot(0, 0)
    mesh0 = om.read_trimesh("../runtime/interpoly_brain.off", face_color=True)
    mesh0.update_vertex_normals()
    p.add_actor(vtk_actor(mesh0.points(), mesh0.face_vertex_indices(), mesh0.face_colors(), opacity=1))



    p.subplot(0, 1)
    mesh1 = om.read_trimesh("../runtime/interpoly_smoothed_brain.off", face_color=True)
    p.add_actor(vtk_actor(mesh1.points(), mesh1.face_vertex_indices(), mesh1.face_colors(), opacity=1))
    
       
    p.show_axes_all()
    p.link_views()  # link all the views
    p.show(auto_close=False)
    viewup = [0, 0, 1]
    p.open_gif("../runtime/smoothed_brain.gif")
    p.orbit_on_path(p.generate_orbital_path(factor=2.0, n_points=32, viewup=viewup, shift=0.2), write_frames=True, viewup=viewup)
    p.close()