import time

def timer(maze, maze_solver):
    t1 = time.perf_counter()
    maze_solver(maze)
    t2 = time.perf_counter()
    print(t2-t1)
