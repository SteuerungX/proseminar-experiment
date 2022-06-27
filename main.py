import numpy as np
import random
import model
from sensor_model import SensorModel
from world_model import WorldModel
from run import Run
import timeit

dim_x = 4
dim_y = 16
error = 0.1

def main():
    n = dim_x*dim_y

    (world, index, n) = init_world()

    # Just a data class for improved structure
    w_model = WorldModel(world, index, n, dim_x, dim_y)

    # 1/n everywhere
    position = model.init_world_state(w_model.n)
    
    # get transition model, observation model
    (T,O) = model.init_models(w_model.world, w_model.index, w_model.n, error)

    # Start run at random position
    run = Run(w_model, O)
    run.run(16)

    """
    for i in range(0,16):
        (x,y) = run.positions[i]
        index = w_model.index[x,y]
        print(len(run.observations))
        sensor_value = run.observations[i].name
        print(f"X: {x} Y: {y}, Index: {index}, Sensor value: {sensor_value}")
    """

    #for (x,y) in run.positions:
        #print(f"X: {x} Y: {y}, Index: {w_model.index[x,y]}")


    (x,y) = init_robot(w_model.world)

    # print(forward(w_model, T, O, run.observations))
    # print(backward(w_model, T, O, run.observations[8::]))
    print(forward_backward_k(w_model, T, O, run.observations, 8))

    for i in range(0,16):
        (x,y) = run.positions[i]
        index = w_model.index[x,y]
        print(f"X: {x} Y: {y}, Index: {index}")

    # n = get_neighbors(x,y,w_model.world)
    # e = get_evidences(w_model.n, error)

    # print(e)
    # print(world)
    # print(np.shape(T))
    # print(np.shape(position))
    # print(T)
    # print(O)
     
def init_world()-> np.array:
        world = [
            [0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0],
            [1,1,0,0,1,0,1,1,0,1,0,1,0,1,1,1],
            [1,0,0,0,1,0,1,1,0,0,0,0,0,1,1,0],
            [0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0],
        ]

        index = [
            [0,1,2,3,-1,4,5,6,7,8,-1,9,10,11,-1,12],
            [-1,-1,13,14,-1,15,-1,-1,16,-1,17,-1,18,-1,-1,-1],
            [-1,19,20,21,-1,22,-1,-1,23,24,25,26,27,-1,-1,28],
            [29,30,-1,31,32,33,-1,34,35,36,37,-1,38,39,40,41],
        ]

        # Number of available field in matrix
        n = 42

        world = np.array(world, dtype=np.int32)
        index = np.array(index, dtype=np.int32)

        return (world, index, n)


def get_neighbors(x: int, y: int , world):

    n = world[x-1,y] if x-1 >= 0 else 1
    o = world[x,y+1] if y+1 < dim_y else 1
    s = world[x+1,y] if x+1 < dim_x else 1
    w = world[x,y-1] if y-1 >= 0 else 1

    return np.array([[n],[o],[s],[w]])

def get_evidences(neighbors, error):
    # returns 16-value vector with either 1, 0

    e = np.array([np.zeros((16,1))], dtype=np.int8)

    def switch(a,i,j):
        a[i,j] = 0 if a[i,j] == 1 else 1

    # Putting error into sensor values
    sensor_error = np.copy(neighbors)

    for i in range(0,4):
        if random.random() < error:
            switch(sensor_error,i,0)

    """
        evidences (index increasing)
        ____
        ___W
        __O_
        __OW
        _S__
        _S_O
        _SW_
        _SWO
        N__W
        N_O_
        N_OW
        NS__
        NS_W
        NSO_
        NSOW      
    """

    # map sensor_error to evidence vector
    index = 0
    for i in range(3,-1,-1):
        if sensor_error[i,0] == 1:
            index += 2**(3-i)

    e[0,index] = 1
    return e

def is_free(x: int,y: int, world) -> bool:
    # Checks if a field is in range and not blocked by a wall
    global dim_x
    global dim_y

    if x < dim_x and y < dim_y and world[x,y] == 0:
        return True
    return False


def init_robot(world):
    # Puts robot on random, free field
    x = random.randint(0,dim_x-1)
    y = random.randint(0,dim_y-1)

    while(not is_free(x,y, world)):
         x = random.randint(0,dim_x-1)
         y = random.randint(0,dim_y-1)
    
    world[x,y] = 2
    return (x,y)

def normalize(vector: np.array):
    shape = np.shape(vector)
    vec_sum = 0

    if shape[1] != 1:
        print("Fehler beim Shape! (Normalize)")
        return None
    
    for i in range(0, shape[0]):
        vec_sum += vector[i, 0]

    for i in range(0, shape[0]):
        vector[i, 0]  = ( vector[i, 0] / vec_sum )
    return vector


def forward(w_model: WorldModel, transition_model: np.array, observation_model, evidences: list) -> np.array:
    # Forward algorithm, returns X_k (last distribution)

    if len(evidences) == 0:
        return model.init_world_state(w_model.n)

    world_state = np.zeros((w_model.n, 1), dtype=np.float64)
    evidence = evidences.pop()

    # world_state = np.add(world_state, np.matmul(transition_model, forward(w_model, transition_model, evidences)))

    # Matrix: Probability that a world state creates a given evidence (here: last element)
    evidence_matrix = SensorModel.get_evidence_matrix(w_model, observation_model, evidence)

    world_state = normalize(np.matmul(evidence_matrix, np.matmul(transition_model, forward(w_model, transition_model, observation_model, evidences))))
    return world_state

def backward(w_model: WorldModel, transition_model: np.array, observation_model: np.array, evidences: list):
    # Backward Algorithm, returns X_k (first distribution)

    world_state = np.ones((w_model.n, 1), dtype=np.float64)

    if len(evidences) == 0:
        return world_state

    evidence = evidences.pop(0)
    evidence_matrix = SensorModel.get_evidence_matrix(w_model, observation_model, evidence)

    world_state = normalize(np.matmul(transition_model.T, np.matmul(evidence_matrix, backward(w_model, transition_model, observation_model, evidences))))

    return world_state

def forward_backward_k(w_model: WorldModel, transition_model: np.array, observation_model: np.array, evidences: list, k: int):
    # Forward-Backward-Algorithm for a given k, returns X_k

    return normalize(forward(w_model, transition_model, observation_model, evidences[:k]) * backward(w_model, transition_model, observation_model, evidences[k:]))

if __name__ == "__main__":
    main()

    # TODO
    # - Versionen von forward-backward (mit /ohne Dynamische Programmierung)
    # - Fixed Lag-Smoothing 
    # - Performanceanalyse