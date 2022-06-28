import numpy as np
import random
from sensor_model import SensorModel
from world_model import WorldModel

class Run():
    
    w_model: WorldModel = None
    o_model: np.array = None
    positions: list = []    # Positions with (x,y)
    evidences: list = []    # Probability distribution for evidences
    observations: list = [] # Actual observations (one of the sixteen observations)

    def __init__(self, w_model: WorldModel, observation_model, pos_x: int  = None, pos_y: int = None):
        self.w_model = w_model
        self.o_model = observation_model

        if pos_x is not None and pos_y is not None:
            # Use given positon
            self.positions.append((pos_x, pos_y))
            self.calculate_evidence(pos_x, pos_y)
        else:
            # Use random position
            self.init_position()
        
    def is_free(self, x: int,y: int) -> bool:
        # Checks if a field is in range and not blocked by a wall
        if x < 0 or y < 0:
            return False

        if x < self.w_model.dim_x and y < self.w_model.dim_y and self.w_model.world[x,y] == 0:
            return True
        return False

    def calculate_evidence(self, pos_x: int, pos_y: int) -> None:

        # Get world state where position is known, then calculate evidence based on that
        field_index: int = self.w_model.index[pos_x, pos_y].item()
        world_state: np.array = np.zeros((self.w_model.n,1),dtype=np.float64)
        world_state[field_index, 0] = 1

        # Computes evidences for known position and appends do evidence list
        self.evidences.append(np.matmul(self.o_model, world_state))

    def init_position(self) -> None:
        # Puts robot on random, free field

        x = random.randint(0,self.w_model.dim_x-1)
        y = random.randint(0,self.w_model.dim_y-1)

        while(not self.is_free(x,y)):
            x = random.randint(0,self.w_model.dim_x-1)
            y = random.randint(0,self.w_model.dim_y-1)
        
        # Append position, sensor value and sensor value probability to their respective lists
        self.positions.append((x,y))
        self.observations.append(SensorModel.get_neighbor_sensor_value(self.get_neighbors(x,y)))
        self.calculate_evidence(x,y)

    def move(self, x: int, y: int) -> None:
    # Moves robot in random direction, provided this field is free
        m: int = random.randint(0,3)

        x_new: int = x
        y_new: int = y

        # 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST
        if m == 0 and self.is_free(x-1,y):
            x_new = x-1
        elif m == 1 and self.is_free(x,y+1):
            y_new = y+1
        elif m == 2 and self.is_free(x+1,y):
            x_new = x+1
        elif self.is_free(x,y-1):
            y_new = y-1       

        # Append to position list, append evidences to evidence list
        self.positions.append((x_new, y_new))
        self.observations.append(SensorModel.get_neighbor_sensor_value(self.get_neighbors(x_new,y_new)))
        self.calculate_evidence(x_new, y_new)

    def run(self, steps: int) -> None:
        # Creates positions and evidences for i steps
        
        for i in range(0, steps):
            # Use last position as basis
            x,y = self.positions[len(self.positions) - 1]
            self.move(x,y)

    def get_neighbors(self, x: int, y: int) -> np.array:
        n = self.w_model.world[x-1,y] if x-1 >= 0 else 1
        o = self.w_model.world[x,y+1] if y+1 < self.w_model.dim_y else 1
        s = self.w_model.world[x+1,y] if x+1 < self.w_model.dim_x else 1
        w = self.w_model.world[x,y-1] if y-1 >= 0 else 1

        arr =  np.array([n, o, s, w])
        arr = arr.reshape(4,1)
        return arr
