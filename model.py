import numpy as np

from sensor import Sensor
from sensor_model import SensorModel

def init_world_state(n: int):

    # creates distribution X0
    x =  np.full((n, 1), (1/n), np.float64)

    return x

def get_neighbor_indices(x: int, y: int , world: np.array, index: np.array) -> tuple:
    dim_x = 4
    dim_y = 16
    #TODO global machen

    # ALle Felder sind standardmäßig mit 1 (belegt) vorbelegt
    neighbors = np.ones((4,1),dtype=np.int32)

    # Liste zum Speichern der Werte, falls sie 0 (frei) sind
    index_list = list()

    # NORTH
    if x-1 >= 0:
        neighbors[0,0] = world[x-1,y]
        
        if neighbors[0,0] == 0:
            neighbor_index = index[x-1,y].item()
            index_list.append(neighbor_index)
    # EAST 
    if y+1 < dim_y:
        neighbors[1,0] = world[x,y+1]

        if neighbors[1,0] == 0:
            neighbor_index = index[x,y+1].item()
            index_list.append(neighbor_index)
    # SOUTH
    if x+1 < dim_x:
        neighbors[2,0] = world[x+1,y]
        
        if neighbors[2,0] == 0:
            neighbor_index = index[x+1,y].item()
            index_list.append(neighbor_index)
    # WEST
    if y-1 >= 0:
        neighbors[3,0] = world[x,y-1]
        if neighbors[3,0] == 0:
            neighbor_index = index[x,y-1].item()
            index_list.append(neighbor_index)   

    # values of neighbors, indizes of neighbors which are free (not wall, not out of range)
    return (neighbors, index_list)

def init_models(world: np.array, index: np.array, n: int, error: float) -> tuple:
    transition_model = np.zeros((n,n), np.float64)
    observation_model = np.zeros((16,n), dtype=np.float64)

    f_index = 0 # index of own field

    dim_x = 4
    dim_y = 16

    for i in range(0,dim_x):
        for j in range(0,dim_y):
            if (world[i,j] == 0):
                k = 0

                (neighbors, index_list) = get_neighbor_indices(i, j, world, index)

                # Create transition model
                for neighbor_index in index_list:
                    # print("Neighbor-Index: "+str(neighbor_index)+ " f_index: "+str(f_index))
                    transition_model[neighbor_index,f_index] = 1/4
                    transition_model[f_index, f_index] = 1 - len(index_list)*(1/4)


                # Create observation model
                # print("Observation probablilities for field: "+str(f_index))
                for direction in Sensor:

                    # Calculate probabilites for the direction sensor to have the value of that fits the current direction
                    north = SensorModel.get_north_probability(direction, neighbors[0,0].item(), error)
                    east = SensorModel.get_east_probability(direction, neighbors[1,0].item(), error)
                    south = SensorModel.get_south_probability(direction, neighbors[2,0].item(), error)
                    west = SensorModel.get_west_probability(direction, neighbors[3,0].item(), error)

                    p =  north * east * south * west

                    # print(f"Observation probablilities for field: {str(f_index)} and direction {direction.name}")
                    # print(f"North:{str(neighbors[0,0].item())} East:{str(neighbors[1,0].item())} South:{str(neighbors[2,0].item())} West:{str(neighbors[3,0].item())} P({direction.value},{f_index}):{str(p)}")
                    # print(f"P(N):{str(north)} P(E):{str(east)} P(S):{str(south)} P(W):{str(west)}")

                    observation_model[direction.value, f_index] = p                 
                
                f_index += 1

    return (transition_model, observation_model)
