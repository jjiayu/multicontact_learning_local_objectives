import multicontact_learning_local_objectives.python.terrain_create as terrain_create
import multicontact_learning_local_objectives.python.visualization as viz
import matplotlib.pyplot as plt #Matplotlib

TerrainModel = terrain_create.pre_designed_terrain(terrain_name="darpa")
ax = viz.display_terrain.drawTerrain(TerrainModel["AllPatches"])
plt.show()
