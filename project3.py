import networkx as nx
import matplotlib.pyplot as plt


# Task 1: Basic Search and Connectivity in an Undirected Graph


def task1():
    print("=== Task 1: Search and Connectivity in an Undirected Graph ===")
    # Construct an undirected graph with two connected components
    # Connected component 1: A, B, C, D
    # Connected component 2: E, F
    G = nx.Graph()
    # Add edges for connected component 1
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
    # Add edge for connected component 2
    G.add_edge('E', 'F')
    
    # (a) Can DFS and BFS starting from any vertex find all connected components?
    # Starting from a single vertex (for example, 'A'), DFS or BFS only traverses the component that contains that vertex.
    start_node = 'A'
    dfs_nodes = list(nx.dfs_preorder_nodes(G, source=start_node))
    bfs_nodes = list(nx.bfs_tree(G, source=start_node).nodes())
    print("DFS traversal starting from node {}: {}".format(start_node, dfs_nodes))
    print("BFS traversal starting from node {}: {}".format(start_node, bfs_nodes))
    # To obtain all connected components, traverse all unvisited nodes, e.g., using nx.connected_components(G)
    all_components = list(nx.connected_components(G))
    print("All connected components in the graph: {}".format(all_components))
    # Answer for (a): A DFS or BFS starting from one vertex can only discover its own connected component. 
    # For a disconnected graph, each component must be searched separately.
    
    # (b) Can both DFS and BFS determine if there is a path between two given nodes?
    node_u = 'A'
    node_v = 'D'
    has_path = nx.has_path(G, node_u, node_v)
    print("Is there a path from node {} to {}? {}".format(node_u, node_v, has_path))
    # Show the paths found by DFS and BFS:
    # For BFS (which returns the path with the fewest edges)
    try:
        bfs_path = nx.shortest_path(G, source=node_u, target=node_v, method='bfs')
    except Exception as e:
        bfs_path = None
    # For DFS, implement a simple recursive function to find a path
    def dfs_path(graph, current, target, visited=None, path=None):
        if visited is None:
            visited = set()
        if path is None:
            path = []
        visited.add(current)
        path = path + [current]
        if current == target:
            return path
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                result = dfs_path(graph, neighbor, target, visited, path)
                if result is not None:
                    return result
        return None

    dfs_found_path = dfs_path(G, node_u, node_v)
    print("Path from {} to {} found by DFS: {}".format(node_u, node_v, dfs_found_path))
    print("Path from {} to {} found by BFS: {}".format(node_u, node_v, bfs_path))
    # Answer for (b): Both methods can determine if there is a path between two nodes and return a valid path.
    
    # (c) If there is a path between vertices u and v, will the DFS and BFS starting from u always return exactly the same path?
    same_path = (dfs_found_path == bfs_path)
    print("Are the paths returned by DFS and BFS identical? {}".format(same_path))
    print("Note: The DFS path depends on the order of traversal, while BFS returns the path with the fewest edges, so they are generally not the same.")
    
    # Plot the undirected graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Task 1: Undirected Graph")
    plt.show()




# Task 2: Strongly Connected Components and Meta Graph in a Directed Graph


def task2():
    print("\n=== Task 2: Strongly Connected Components and Meta Graph of a Directed Graph ===")
    # Construct a directed graph
    # Nodes: A, B, C, D, E, F
    # The edges are designed so that:
    #   {A, B, C} form a strongly connected component (A->B, B->C, C->A)
    #   {D, E} form a strongly connected component (D->E, E->D)
    #   F is an isolated component (its own strongly connected component)
    # Also, A->B->D->E->F establishes the connectivity of the graph.
    DG = nx.DiGraph()
    # Add edges for strongly connected component {A, B, C}
    DG.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
    # Connect to the next part
    DG.add_edge('B', 'D')
    # Add edges for strongly connected component {D, E}
    DG.add_edges_from([('D', 'E'), ('E', 'D')])
    # Edge from E to F (F is an isolated strongly connected component)
    DG.add_edge('E', 'F')
    
    # (a) Compute the strongly connected components of the directed graph
    scc = list(nx.strongly_connected_components(DG))
    print("Strongly connected components: {}".format(scc))
    
    # (b) Construct the meta graph using the strongly connected components
    # NetworkX provides the condensation method, which returns the meta graph (each node represents an SCC)
    C = nx.condensation(DG, scc=scc)
    print("Meta graph (Condensation Graph) nodes: ", C.nodes(data=True))
    print("Meta graph edges: ", list(C.edges()))
    # (c) Perform a topological sort on the meta graph (the meta graph is a DAG)
    topo_order = list(nx.topological_sort(C))
    print("Topological order of the meta graph: {}".format(topo_order))
    
    # Plot the original directed graph
    pos = nx.spring_layout(DG)
    nx.draw(DG, pos, with_labels=True, node_color='lightgreen', arrowstyle='->', arrowsize=15)
    plt.title("Task 2: Original Directed Graph")
    plt.show()
    
    # Plot the meta graph
    pos_meta = nx.spring_layout(C)
    labels = {node: data['members'] for node, data in C.nodes(data=True)}
    nx.draw(C, pos_meta, with_labels=True, labels=labels, node_color='lightcoral', arrowstyle='->', arrowsize=15)
    plt.title("Task 2: Meta Graph (Condensation Graph)")
    plt.show()



# Task 3: Shortest Path Tree and Minimum Spanning Tree in a Weighted Graph


def task3():
    print("\n=== Task 3: Shortest Path Tree and Minimum Spanning Tree of a Weighted Graph ===")
    # Construct a weighted undirected graph
    # Nodes: A, B, C, D, E
    # Edges and their weights:
    # A-B: 2, A-C: 4, B-C: 1, B-D: 7, C-E: 3, D-E: 1
    WG = nx.Graph()
    WG.add_weighted_edges_from([
        ('A', 'B', 2),
        ('A', 'C', 4),
        ('B', 'C', 1),
        ('B', 'D', 7),
        ('C', 'E', 3),
        ('D', 'E', 1)
    ])
    
    # (a) Generate the shortest path tree from node A using Dijkstra's algorithm
    source = 'A'
    distances, paths = nx.single_source_dijkstra(WG, source=source)
    print("Shortest paths and distances from node {}:".format(source))
    for target in sorted(WG.nodes()):
        print("Distance to {}: {}, Path: {}".format(target, distances[target], paths[target]))
    
    # (b) Generate the minimum spanning tree (MST)
    mst = nx.minimum_spanning_tree(WG)
    print("\nEdges of the Minimum Spanning Tree (edge weights):")
    for u, v, data in mst.edges(data=True):
        print("{} - {} (weight {})".format(u, v, data['weight']))
    
    # (c) Comparison between the shortest path tree and the minimum spanning tree
    print("\nAnswer: The shortest path tree and the minimum spanning tree are usually not the same.")
    print("Reason: The shortest path tree ensures the minimum cost path from the source to each node,")
    print("while the minimum spanning tree minimizes the total edge weight of the graph. Their objectives differ.")
    
    # (d) If the graph contains an edge with a negative weight, is Dijkstra's algorithm applicable?
    print("\nAnswer: Dijkstra's algorithm is not applicable to graphs with negative weight edges,")
    print("because negative weights can lead to incorrect shortest path calculations.")
    
    # Plot the weighted graph and its minimum spanning tree
    pos = nx.spring_layout(WG)
    edge_labels = nx.get_edge_attributes(WG, 'weight')
    plt.figure()
    nx.draw(WG, pos, with_labels=True, node_color='lightyellow', edge_color='gray')
    nx.draw_networkx_edge_labels(WG, pos, edge_labels=edge_labels)
    plt.title("Task 3: Weighted Graph")
    plt.show()
    
    pos_mst = nx.spring_layout(mst)
    plt.figure()
    nx.draw(mst, pos_mst, with_labels=True, node_color='lightblue', edge_color='black')
    mst_edge_labels = nx.get_edge_attributes(mst, 'weight')
    nx.draw_networkx_edge_labels(mst, pos_mst, edge_labels=mst_edge_labels)
    plt.title("Task 3: Minimum Spanning Tree")
    plt.show()



# Main function: Execute all tasks sequentially

def main():
    task1()
    task2()
    task3()

if __name__ == "__main__":
    main()