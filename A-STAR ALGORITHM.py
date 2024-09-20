import heapq
def a_star(start, goal, graph, heuristic):
    open_set = [(heuristic[start], start)]  
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor, cost in graph[current].items():
            tentative_g_score = g_score[current] + cost
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (tentative_g_score + heuristic[neighbor], neighbor))
    return None
if __name__ == "__main__":
    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }
    
    heuristic = {
        'A': 7,
        'B': 6,
        'C': 2,
        'D': 0
    }
    
    start = 'A'
    goal = 'D'
    path = a_star(start, goal, graph, heuristic)
    
    if path:
        print("Path found:", path)
    else:
        print("No path found")
