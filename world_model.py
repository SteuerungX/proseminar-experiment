import numpy as np

class WorldModel():
    world = None
    index = None

    # TODO avoid hard-coding the variables below, create methods to compute them

    n = None # Number of fields which can be accessed (are not walls)
    dim_x = None
    dim_y = None

    def __init__(self, world: np.array = None, index: np.array = None, n: int = None, dim_x: int = None, dim_y: int = None) -> None:

        if world is not None and index is not None and n is not None and dim_x is not None and dim_y is not None:
            self.world = world
            self.index = index
            self.n = n
            self.dim_x = dim_x
            self.dim_y = dim_y
        else: 
            self.init_world()

    def init_world(self)-> np.array:
        self.dim_x = 4
        self.dim_y = 16

        self.world = [
            [0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0],
            [1,1,0,0,1,0,1,1,0,1,0,1,0,1,1,1],
            [1,0,0,0,1,0,1,1,0,0,0,0,0,1,1,0],
            [0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0],
        ]

        self.index = [
            [0,  1, 2, 3,-1, 4, 5, 6, 7, 8,-1, 9,10,11,-1,12],
            [-1,-1,13,14,-1,15,-1,-1,16,-1,17,-1,18,-1,-1,-1],
            [-1,19,20,21,-1,22,-1,-1,23,24,25,26,27,-1,-1,28],
            [29,30,-1,31,32,33,-1,34,35,36,37,-1,38,39,40,41],
        ]

        # Number of available field in matrix
        self.n = 42

        world = np.array(world, dtype=np.int32)
        index = np.array(index, dtype=np.int32)

        return (self.world, self.index, self.n, self.dim_x, self.dim_y)