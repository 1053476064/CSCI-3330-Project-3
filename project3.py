import networkx as nx
import matplotlib.pyplot as plt

##############################################
# 任务1：无向图的基本搜索和连通性问题
##############################################

def task1():
    print("=== 任务1：无向图的搜索和连通性 ===")
    # 构造一个无向图，包含两个连通分量
    # 连通分量1: A, B, C, D
    # 连通分量2: E, F
    G = nx.Graph()
    # 添加边：连通分量1
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
    # 添加边：连通分量2
    G.add_edge('E', 'F')
    
    # (a) 从任一顶点出发的 DFS 与 BFS 能否找到图中的所有连通分量？
    # 直接从一个顶点（例如'A'）出发，DFS 或 BFS 只能遍历所在连通分量
    start_node = 'A'
    dfs_nodes = list(nx.dfs_preorder_nodes(G, source=start_node))
    bfs_nodes = list(nx.bfs_tree(G, source=start_node).nodes())
    print("从节点 {} 出发的 DFS 遍历结果: {}".format(start_node, dfs_nodes))
    print("从节点 {} 出发的 BFS 遍历结果: {}".format(start_node, bfs_nodes))
    # 要得到所有连通分量需要遍历所有未访问过的节点，可用 nx.connected_components(G)
    all_components = list(nx.connected_components(G))
    print("图中所有的连通分量: {}".format(all_components))
    # 答案：(a) 单个起点的 DFS 或 BFS 只能发现所在连通分量，若图不连通，则需要对所有组件分别搜索。
    
    # (b) DFS 与 BFS 是否能判断两个给定节点间是否存在路径？
    node_u = 'A'
    node_v = 'D'
    has_path = nx.has_path(G, node_u, node_v)
    print("节点 {} 到 {} 是否存在路径? {}".format(node_u, node_v, has_path))
    # 进一步展示 DFS 和 BFS 分别找到的路径：
    # 对于 BFS（找到边数最少的路径）
    try:
        bfs_path = nx.shortest_path(G, source=node_u, target=node_v, method='bfs')
    except Exception as e:
        bfs_path = None
    # 对于 DFS，我们自己实现一个简单的递归查找路径函数
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
    print("使用 DFS 搜索到的 {} 到 {} 的路径: {}".format(node_u, node_v, dfs_found_path))
    print("使用 BFS 搜索到的 {} 到 {} 的路径: {}".format(node_u, node_v, bfs_path))
    # 答案：(b) 都可以判断两节点间是否存在路径，并返回一个可行的路径。
    
    # (c) 若 u 与 v 间存在路径，从 u 开始的 DFS 与 BFS 是否总能找到完全相同的路径？
    same_path = (dfs_found_path == bfs_path)
    print("DFS 与 BFS 返回的路径是否完全相同？ {}".format(same_path))
    print("注：DFS 搜索的路径依赖于遍历顺序，BFS 返回的是边数最少的路径，因此一般不相同。")
    
    # 绘制无向图
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("任务1：无向图示意图")
    plt.show()


##############################################
# 任务2：有向图的强连通分量及元图
##############################################

def task2():
    print("\n=== 任务2：有向图的强连通分量与元图 ===")
    # 构造一个有向图
    # 节点：A, B, C, D, E, F
    # 边的设计使得:
    #   {A, B, C} 形成一个强连通分量 (A->B, B->C, C->A)
    #   {D, E} 形成一个强连通分量 (D->E, E->D)
    #   F 为孤立点（单独的强连通分量）
    # 同时 A->B->D->E->F 构成图的连接结构
    DG = nx.DiGraph()
    # 添加强连通分量 {A, B, C}
    DG.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
    # 连接到下一个部分
    DG.add_edge('B', 'D')
    # 强连通分量 {D, E}
    DG.add_edges_from([('D', 'E'), ('E', 'D')])
    # 从 E 指向 F（F 为单独的强连通分量）
    DG.add_edge('E', 'F')
    
    # (a) 计算有向图的强连通分量
    scc = list(nx.strongly_connected_components(DG))
    print("强连通分量: {}".format(scc))
    
    # (b) 利用强连通分量构造元图
    # NetworkX 提供 condensation 方法，其返回的图即为元图（每个节点代表一个 SCC）
    C = nx.condensation(DG, scc=scc)
    print("元图（Condensation Graph）的节点: ", C.nodes(data=True))
    print("元图的边: ", list(C.edges()))
    # (c) 对元图进行拓扑排序（元图一定是 DAG）
    topo_order = list(nx.topological_sort(C))
    print("元图拓扑排序的结果: {}".format(topo_order))
    
    # 绘制原始有向图
    pos = nx.spring_layout(DG)
    nx.draw(DG, pos, with_labels=True, node_color='lightgreen', arrowstyle='->', arrowsize=15)
    plt.title("任务2：原始有向图")
    plt.show()
    
    # 绘制元图
    pos_meta = nx.spring_layout(C)
    labels = {node: data['members'] for node, data in C.nodes(data=True)}
    nx.draw(C, pos_meta, with_labels=True, labels=labels, node_color='lightcoral', arrowstyle='->', arrowsize=15)
    plt.title("任务2：元图 (Condensation Graph)")
    plt.show()


##############################################
# 任务3：带权图的最短路径树与最小生成树
##############################################

def task3():
    print("\n=== 任务3：带权图的最短路径树与最小生成树 ===")
    # 构造一个带权无向图
    # 节点: A, B, C, D, E
    # 边及其权重:
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
    
    # (a) 利用 Dijkstra 算法从节点 A 生成最短路径树
    source = 'A'
    distances, paths = nx.single_source_dijkstra(WG, source=source)
    print("从节点 {} 出发的最短路径及距离：".format(source))
    for target in sorted(WG.nodes()):
        print("到 {} 的距离: {}，路径: {}".format(target, distances[target], paths[target]))
    
    # (b) 生成最小生成树
    mst = nx.minimum_spanning_tree(WG)
    print("\n最小生成树的边 (边权):")
    for u, v, data in mst.edges(data=True):
        print("{} - {} (weight {})".format(u, v, data['weight']))
    
    # (c) 对比最短路径树和最小生成树
    print("\n答案：最短路径树和最小生成树通常不相同。")
    print("原因：最短路径树保证从起点到各节点的路径代价最小，而最小生成树保证整个图的边权总和最小，两者目标不同。")
    
    # (d) 若图中存在负权边，Dijkstra 算法是否适用？
    print("\n答案：Dijkstra 算法不适用于含负权边的图，因为负权边可能导致错误的最短路径计算。")
    
    # 绘制带权图及其最小生成树
    pos = nx.spring_layout(WG)
    edge_labels = nx.get_edge_attributes(WG, 'weight')
    plt.figure()
    nx.draw(WG, pos, with_labels=True, node_color='lightyellow', edge_color='gray')
    nx.draw_networkx_edge_labels(WG, pos, edge_labels=edge_labels)
    plt.title("任务3：带权图")
    plt.show()
    
    pos_mst = nx.spring_layout(mst)
    plt.figure()
    nx.draw(mst, pos_mst, with_labels=True, node_color='lightblue', edge_color='black')
    mst_edge_labels = nx.get_edge_attributes(mst, 'weight')
    nx.draw_networkx_edge_labels(mst, pos_mst, edge_labels=mst_edge_labels)
    plt.title("任务3：最小生成树")
    plt.show()


##############################################
# 主函数：依次执行各任务
##############################################

def main():
    task1()
    task2()
    task3()

if __name__ == "__main__":
    main()
