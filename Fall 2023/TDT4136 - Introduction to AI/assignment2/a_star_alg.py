# Method source: http://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html
# Code source: https://www.redblobgames.com/pathfinding/a-star/implementation.html

import collections
from typing import Protocol, Iterator, Tuple, TypeVar, Optional

T = TypeVar('T')
Location = TypeVar('Location')

class Graph(Protocol):
    def neighbors(self, id: Location) -> list[Location]: pass

class SimpleGraph:
    def __init__(self):
        self.edges: dict[Location, list[Location]] = {}
    
    def neighbors(self, id: Location) -> list[Location]:
        return self.edges[id]


class Queue:
    def __init__(self):
        self.elements = collections.deque()
    
    def empty(self) -> bool:
        return not self.elements
    
    def put(self, x: T):
        self.elements.append(x)
    
    def get(self) -> T:
        return self.elements.popleft()

# utility functions for dealing with square grids


GridLocation = Tuple[int, int]

class SquareGrid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.walls: list[GridLocation] = []
    
    def in_bounds(self, id: GridLocation) -> bool:
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id: GridLocation) -> bool:
        return id not in self.walls
    
    def neighbors(self, id: GridLocation) -> Iterator[GridLocation]:
        (x, y) = id
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1)] # E W N S
        # see "Ugly paths" section for an explanation:
        if (x + y) % 2 == 0: neighbors.reverse() # S N W E
        results = filter(self.in_bounds, neighbors)
        results = filter(self.passable, results)
        print('here ->', results)
        return results

class WeightedGraph(Graph):
    def cost(self, from_id: Location, to_id: Location) -> float: pass

class GridWithWeights(SquareGrid):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self.weights: dict[GridLocation, float] = {}
    
    def cost(self, from_node: GridLocation, to_node: GridLocation) -> float:
        print('cost value ->', self.weights.get(to_node, 1))
        return self.weights.get(to_node, 1)

import heapq

class PriorityQueue:
    def __init__(self):
        self.elements: list[tuple[float, T]] = []
    
    def empty(self) -> bool:
        # print('priority queue:', self.elements)
        return not self.elements
    
    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self) -> T:
        return heapq.heappop(self.elements)[1]

def reconstruct_path(came_from: dict[Location, Location],
                     start: Location, goal: Location) -> list[Location]:

    current: Location = goal
    path: list[Location] = []
    if goal not in came_from: # no path was found
        print('GOAL NOT FOUND')
        return []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path


heuristic_modes = ['manhattan', 'euclidean', 'task5']


def heuristic(a: GridLocation, b: GridLocation, mode: str = 'manhattan') -> float:
    assert mode in heuristic_modes, f"{mode} is not a norm implemented yet"
    (x1, y1) = a
    (x2, y2) = b
    if mode == 'manhattan':
        return abs(x1 - x2) + abs(y1 - y2)
    if mode == 'euclidean':
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1/2)
    if mode == 'task5':
        print('here task5')
        return abs((x1 - .25) - x2) + abs(y1 - y2)

import Map as m
# def a_star_search(graph: WeightedGraph, start: Location, goal: Location, heuristic_mode: str = 'manhattan'):
def a_star_search(start: Location, goal: Location, map: m.Map_Obj = m.Map_Obj(task=1), heuristic_mode: str = 'manhattan'):
    assert heuristic_mode in heuristic_modes, f"{heuristic_mode} is not a norm implemented yet"

    graph, _, _ = convert_map_into_graph(map_obj=map)

    frontier = PriorityQueue()
    frontier.put(start, 0)

    came_from: dict[Location, Optional[Location]] = {}
    cost_so_far: dict[Location, float] = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    k = 0
    while not frontier.empty():
        current: Location = frontier.get()
        k += 1
    
        if current == goal:
            break
        
        for next in graph.neighbors(current):

            new_cost = cost_so_far[current] + graph.cost(current, next)
            print('cost_so_far[current] ->', cost_so_far[current])
            print('graph.cost(current, next) ->', graph.cost(current, next))
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal, heuristic_mode)
                frontier.put(next, priority)
                came_from[next] = current
    

    return came_from, cost_so_far


import numpy as np



def a_star_moving_goal(
        start: Location,
        goal: Location, 
        heuristic_mode: str = 'manhattan', 
        map: m.Map_Obj = m.Map_Obj(task=5),
        ):
    
    # goal = map.get_goal_pos()
    # goal = (goal[1], goal[0])
    current = start
    currents = [start]
    all_path = []
    while current != goal:
        came_from, cost_so_far, current = a_star_search_task5(current, goal, heuristic_mode=heuristic_mode, map=map)
        goal = map.get_goal_pos()
        goal = (goal[1], goal[0])
        [all_path.append(path) for path in reconstruct_path(came_from, start=start, goal=current)]
        print('path -->', all_path)
        start = current
        currents.append(current)
    return all_path, currents

def a_star_search_task5(
        start: Location,
        goal: Location, 
        heuristic_mode: str = 'manhattan', 
        map: m.Map_Obj = m.Map_Obj(task=5),
    ):
    assert heuristic_mode in heuristic_modes, f"{heuristic_mode} is not a norm implemented yet"

    graph, _, _ = convert_map_into_graph(map)


    frontier = PriorityQueue()
    frontier.put(start, 0)

    came_from: dict[Location, Optional[Location]] = {}
    cost_so_far: dict[Location, float] = {}
    came_from[start] = None
    cost_so_far[start] = 0

    currents = []
    # frontier = PriorityQueue()

    if frontier.empty():
        frontier.put(start, priority)

    if came_from == {}: 
        came_from[start] = None

        cost_so_far[start] = new_cost
    
    while not frontier.empty():
        current: Location = frontier.get()

        currents.append(current)
        if current == goal:
            break
        
        for next in graph.neighbors(current):

            new_cost = cost_so_far[current] + graph.cost(current, next)

            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal, heuristic_mode)
                frontier.put(next, priority)
                came_from[next] = current

                new_goal = map.tick()
                new_goal = (new_goal[1], new_goal[0])
                goal: Location = new_goal
                print('new goal')

    last_position = current # if current != goal else map.get_goal_pos()

    print('currents in loop ->', currents)
    return came_from, cost_so_far, last_position


def convert_map_into_graph(map_obj: m.Map_Obj = m.Map_Obj(task=5),
                            graph_type: str = 'weights'
                            ) -> SquareGrid or WeightedGraph | Location | Location :
    
    map_int, _ = map_obj.get_maps()
    # if graph_type == 'weights':
    map: WeightedGraph = GridWithWeights(map_int.shape[1], map_int.shape[0])
    map.weights = {(loc[1], loc[0]): map_int[loc[0], loc[1]] for loc in np.argwhere(map_int >= 1)}

    walls = np.where(map_int == -1)
    walls_converted = [(walls[1][k], walls[0][k]) for k in range(len(walls[0]))]
    map.walls = walls_converted

    start, goal = map_obj.get_start_pos(), map_obj.get_goal_pos()

    start: Location = (start[1], start[0]) 
    goal: Location = (goal[1], goal[0])
    return map, start, goal


            

