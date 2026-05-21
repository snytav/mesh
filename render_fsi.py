from paraview.simple import *

# 1. Load the FEniCS multi-grid iteration file
filename = "fsi_results.xdmf"
print(f"Loading {filename} via ParaView Core...")
reader = Xdmf3ReaderS(FileName=filename)

# 2. Update the pipeline to process the final frame
reader.UpdatePipeline()

# 3. Create a clean render viewing canvas window
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1024, 768]

# 4. Display the mesh data on screen
display = Show(reader, view)

# 5. Set the active visual tracking field to your Pressure data array
# (ParaView automatically finds point/cell fields out of the box)
ColorBy(display, ('POINTS', 'Pressure'))

# Automatically adjust the visual color spectrum boundaries to match your data
pressureLUT = GetColorTransferFunction('Pressure')
pressureLUT.ApplyPreset('Turbo', True) # Uses the same clean color profile
display.RescaleTransferFunctionToDataRange(False, True)

# Show the scalar color bar legend on the right side
display.SetScalarBarVisibility(view, True)

# 6. Center the camera view onto your FSI geometries
view.ResetCamera()
Render()

print("Opening Interactive Render Viewport... Close window to finish.")
Interact()