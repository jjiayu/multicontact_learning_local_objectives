import multicontact_learning_local_objectives.python.terrain_create as terrain_create
import multicontact_learning_local_objectives.python.visualization as viz

TerrainModel = terrain_create.pre_designed_terrain(terrain_name = "flat", NumSteps = 2, NumLookAhead = 10)
viz.DisplayResults(TerrainModel = TerrainModel, SingleOptResult = None, AllOptResult = None)

print(len(TerrainModel["ContactSurfsVertice"]))