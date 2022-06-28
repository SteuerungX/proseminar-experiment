from errno import EISDIR
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
    (world, index, n) = init_world()

    # Just a data class for improved structure
    w_model = WorldModel(world, index, n, dim_x, dim_y)

    # 1/n everywhere
    position = model.init_world_state(w_model.n)
    
    # get transition model, observation model
    (T,O) = model.init_models(w_model.world, w_model.index, w_model.n, error)

    # Start run at random position
    run = Run(w_model, O)
    run.run(24)

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
    # print(forward_backward_k(w_model, T, O, run.observations, 8))

    f1 = forward_backward(w_model, T, O, run.observations)[24]

    f2 = forward_backward_dynamic_programming(w_model, T, O, run.observations)[24]
    # f3 = forward_backward_dynamic_programming_2(w_model, T, O, run.observations)[24]
    f3 = forward_backward_4(w_model, T, O, run.observations)[24]

    if len(f1) == len(f2) and len(f2) == len(f3):
        for i in range(0, len(f1)):
            print('[{:02}] nicht dyn.: {:3.15f}, \t dyn.: {:3.15f}, \t dyn(2).: {:3.15f}'.format(i, f1[i,0], f2[i,0], f3[i,0]))
    else:
        print("Fehler: Einträge habe nicht gleiche Länge")

    """

    for i in range(0, len(f1)):
            print('[{:02}] nicht dyn.: {:3.15f}'.format(i, f1[i,0]))
        
    """

    pos_list = []
    for i in range(0,24):
        (x,y) = run.positions[i]
        pos_index = w_model.index[x,y]
        pos_list.append(pos_index)
        # print(f"X: {x} Y: {y}, Index: {pos_index}")
    print(pos_list)
     
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

    # Matrix: Probability that a world state creates a given evidence (here: last element)
    evidence_matrix = SensorModel.get_evidence_matrix(w_model, observation_model, evidence)

    world_state = normalize(np.matmul(evidence_matrix, np.matmul(transition_model, forward(w_model, transition_model, observation_model, evidences))))
    return world_state

def forward_step(w_model: WorldModel, transition_model: np.array, observation_model, previous_step: np.array, evidence: np.array) -> np.array:
    """
        Uses one evidence set to perform a single 'step' forwards. 
    """
    evidence_matrix = SensorModel.get_evidence_matrix(w_model, observation_model, evidence)

    world_state = normalize(np.matmul(evidence_matrix, np.matmul(transition_model, previous_step)))
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

def backward_step(w_model: WorldModel, transition_model: np.array, observation_model: np.array, previous_step: np.array, evidence: np.array):
    """
        Uses one evidence set to perform a single 'step' backwards. 
    """
    evidence_matrix = SensorModel.get_evidence_matrix(w_model, observation_model, evidence)

    world_state = normalize(np.matmul(transition_model.T, np.matmul(evidence_matrix, previous_step)))
    return world_state


def forward_backward_k(w_model: WorldModel, transition_model: np.array, observation_model: np.array, evidences: list, k: int) -> np.array:
    # Forward-Backward-Algorithm for a given k, returns X_k

    return normalize(forward(w_model, transition_model, observation_model, evidences[:k].copy()) * backward(w_model, transition_model, observation_model, evidences[k:].copy()))

def forward_backward(w_model: WorldModel, transition_model: np.array, observation_model: np.array, evidences: list) -> list:
    """
        Returns list with all estimations from time=1 to time=t; t beeing len(evidences)
        Is less efficient than dynamic programming because parts of the forward-/backward algorithms get calculated multiple times
    """
    estimations = []

    for k in range(1,len(evidences)+1):
        estimations.append(forward_backward_k(w_model, transition_model, observation_model, evidences.copy(), k))
    return estimations

def forward_backward_dynamic_programming(w_model: WorldModel, transition_model: np.array, observation_model: np.array, evidences: list) -> list:
    """
        Does execute the forward-backward algorithm using the dynamic programming paradigm to increase efficency and reduce complexity to linear time complexity
    """
    print(f'Länge der Evidenzen: {len(evidences)}')

    estimations = []
    forward_steps = []
    forward_steps.append(model.init_world_state(w_model.n))
    backward_var = np.ones((w_model.n, 1), dtype=np.float64)

    for i in range(1, len(evidences)+1):
        forward_steps.append(forward_step(w_model,transition_model, observation_model, forward_steps[i-1], evidences[i-1]))

    for i in range(len(evidences), 0, -1):
        estimations.append(normalize(forward_steps[i] * backward_var))
        backward_var = backward_step(w_model, transition_model, observation_model, backward_var, evidences[i-1])

    return estimations


def forward_backward_dynamic_programming_2(w_model: WorldModel, transition_model: np.array, observation_model: np.array, evidences: list) -> list:
    """
        Does execute the forward-backward algorithm using the dynamic programming paradigm to increase efficency and reduce complexity to linear time complexity
    """
    
    estimations = []
    forward_steps = [model.init_world_state(w_model.n)] # filled with X_0
    backward_steps = [np.ones((w_model.n, 1), dtype=np.float64)]

    for k in range(0, len(evidences)):
        forward_steps.append(forward_step(w_model,transition_model, observation_model, forward_steps[k], evidences[k]))
        backward_steps.append(backward_step(w_model, transition_model, observation_model, backward_steps[k], evidences[len(evidences) - k-1]))

    for k in range(len(evidences), 0, -1):
        estimations.append(normalize(forward_steps[k] * backward_steps[k]))

    return estimations

def forward_backward_4(w_model: WorldModel, transition_model: np.array, observation_model: np.array, evidences: list) -> list:
    estimations = []
    fw = [model.init_world_state(w_model.n)] # f1, ..., ft
    bw = [np.ones((w_model.n, 1), dtype=np.float64)]
    bw2 = [np.ones((w_model.n, 1), dtype=np.float64)] # bt, ...., b1

    b = np.ones((w_model.n, 1), dtype=np.float64) # bt, ..., b1

    for k in range(1,len(evidences)+1):
        fw.append(forward_step(w_model, transition_model, observation_model, fw[k-1], evidences[k-1]))
        bw2.append(backward_step(w_model, transition_model, observation_model, bw2[k-1], evidences[len(evidences)-k]))
        pass

    """
    for k in range(1,len(evidences)+1):
        # fw.append(forward(w_model, transition_model, observation_model, evidences[:k].copy()))
        bw.append(backward(w_model, transition_model, observation_model, evidences[k:].copy()))
        estimations.append(normalize(fw[k] * bw2[len(bw)-1-k]))
    """

    for k in range(len(evidences),0,-1):        
        estimations.append(normalize(fw[k] * b)) # x_t, ..., x_1
        b = backward_step(w_model, transition_model, observation_model, bw2[k-1], evidences[k-1]) # bt, ..., b1

    """
    a = len(bw)-2
    for i in range(0,w_model.n):
        # print('[{}] bw: {}, \t bw2: {}'.format(i, bw[len(bw)-1-a][i,0], bw2[a][i,0]))
         print('[{}] bw: {}, \t bw2: {}'.format(i, bw[a][i,0], bw2[len(bw)-1-a][i,0]))
    print(len(bw)-1)
    print(len(bw2)-1)
    print(k)
    print(len(bw2)-k-1)
    """
    estimations.reverse()
    return estimations

if __name__ == "__main__":
    main()

    # TODO
    # - Versionen von forward-backward (mit /ohne Dynamische Programmierung)
    # - Bugfix für T[12,12]
    # - Fixed Lag-Smoothing 
    # - Performanceanalyse