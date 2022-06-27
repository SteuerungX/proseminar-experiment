import numpy as np
from sensor import Sensor
from world_model import WorldModel

class SensorModel():

    @staticmethod
    def is_east(direction: Sensor) -> bool:

        east_directions = {
            Sensor.EAST,
            Sensor.SOUTH_EAST,
            Sensor.WEST_EAST,
            Sensor.WEST_SOUTH_EAST,
            Sensor.NORTH_EAST,
            Sensor.NORTH_SOUTH_EAST,
            Sensor.NORTH_WEST_EAST,
            Sensor.ALL
        }

        if direction in east_directions:
            return True
        return False

    @staticmethod
    def is_south(direction: Sensor) -> bool:
        south_directions = {
            Sensor.SOUTH,
            Sensor.SOUTH_EAST,
            Sensor.WEST_SOUTH,
            Sensor.WEST_SOUTH_EAST,
            Sensor.NORTH_SOUTH,
            Sensor.NORTH_SOUTH_EAST,
            Sensor.NORTH_WEST_SOUTH,
            Sensor.ALL        
        }

        if direction in south_directions:
            return True
        return False

    @staticmethod
    def is_west(direction: Sensor) -> bool:
        west_directions = {
            Sensor.WEST,
            Sensor.WEST_EAST,
            Sensor.WEST_SOUTH,
            Sensor.WEST_SOUTH_EAST,
            Sensor.NORTH_WEST,
            Sensor.NORTH_WEST_EAST,
            Sensor.NORTH_WEST_SOUTH,
            Sensor.ALL        
        }

        if direction in west_directions:
            return True
        return False

    @staticmethod
    def is_north(direction: Sensor) -> bool:
        north_directions = {
            Sensor.NORTH,
            Sensor.NORTH_EAST,
            Sensor.NORTH_SOUTH,
            Sensor.NORTH_SOUTH_EAST,
            Sensor.NORTH_WEST,
            Sensor.NORTH_WEST_EAST,
            Sensor.NORTH_WEST_SOUTH,
            Sensor.ALL        
        }

        if direction in north_directions:
            return True
        return False

    @staticmethod
    def get_east_probability(direction: Sensor, east_value: int, error: float) -> float:
        if SensorModel.is_east(direction) and east_value == 1:
            return (1-error)
        elif SensorModel.is_east(direction) and east_value == 0:
            return error
        elif not SensorModel.is_east(direction) and east_value == 1:
            return error
        else:
            return (1-error)

    @staticmethod
    def get_south_probability(direction: Sensor, south_value: int, error: float) -> float:
        if SensorModel.is_south(direction) and south_value == 1:
            return (1-error)
        elif SensorModel.is_south(direction) and south_value == 0:
            return error
        elif not SensorModel.is_south(direction) and south_value == 1:
            return error
        else:
            return (1-error)

    @staticmethod
    def get_west_probability(direction: Sensor, west_value: int, error: float) -> float:
        if SensorModel.is_west(direction) and west_value == 1:
            return (1-error)
        elif SensorModel.is_west(direction) and west_value == 0:
            return error
        elif not SensorModel.is_west(direction) and west_value == 1:
            return error
        else:
            return (1-error)

    @staticmethod
    def get_north_probability(direction: Sensor, north_value: int, error: float) -> float:
        if SensorModel.is_north(direction) and north_value == 1:
            return (1-error)
        elif SensorModel.is_north(direction) and north_value == 0:
            return error
        elif not SensorModel.is_north(direction) and north_value == 1:
            return error
        else:
            return (1-error)

    @staticmethod
    def get_evidence_matrix(world_model: WorldModel, observation_model: np.array, evidence: Sensor):
        # Returns probability matrix that a state results in a given evidence

        evidence_matrix = np.zeros((world_model.n, world_model.n), dtype=np.float64)

        for i in range(0, world_model.n):
            evidence_matrix[i,i] = observation_model[evidence.value, i]

        return evidence_matrix

    @staticmethod
    def get_neighbor_sensor_value(neighbors: np.array):
        # Return sensor value, based on neighbors
        # Neighbors are in order NORTH, EAST, SOUTH, WEST

        value = neighbors[0,0] * 8 + neighbors[1,0] + neighbors[2,0] * 2 + neighbors[3,0] * 4

        return Sensor(value)