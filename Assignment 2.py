import numpy as np

def pagerank(M, E, teleportation_prob, max_iterations=1000, convergence_threshold=1e-9):
    n = len(M)
    rank = np.copy(E)
    last_rank = np.zeros(n)

    for i in range(max_iterations):
        last_rank = rank
        rank = (1 - teleportation_prob) * np.dot(M, rank) + teleportation_prob * E
        if np.linalg.norm(rank - last_rank) < convergence_threshold:
            break
    print(i)
    return rank

def pagerank_closed(M, E, teleportation_prob):
    n = len(M)
    rank = np.dot(np.linalg.inv((np.identity(n)-(1-teleportation_prob)*M)),(teleportation_prob)*E)
    return rank

# Example: Four-webpage graph
M = np.array([[0, 1/2, 0, 0],
              [1/3, 0, 0, 1/2],
              [1/3, 0, 1, 1/2],
              [1/3, 1/2, 0, 0]])
E = np.array([1/4, 1/4, 1/4, 1/4])

M2 = np.array([[0, 1/2, 1, 0],
              [1/3, 0, 0, 1/2],
              [1/3, 0, 0, 1/2],
              [1/3, 1/2, 0, 0]])

# Compute PageRank
teleportation_prob = 0.2
for i in range(10):
    teleportation_prob = (i+1)*0.1
    page_rank = pagerank(M2, E, teleportation_prob)
    print("PageRank for the four-webpage graph:")
    print(page_rank)

    # Compute PageRank closed
    page_rank_closed = pagerank_closed(M2, E, teleportation_prob)
    print("PageRank closed form for the four-webpage graph:")
    print(page_rank_closed)

E_a = [0.2, 0.4, 0.6, 0.8, 1.0]
for i in E_a:
    other = (1-i)/3
    E_new = np.array([other, other, i, other])
    print(E_new)
    page_rank = pagerank(M, E_new, teleportation_prob)
    print(page_rank)
    page_rank_closed = pagerank_closed(M, E_new, teleportation_prob)
    print(page_rank_closed)

M3= np.array([[0, 1/2, 0, 1, 1/2],
              [1/2, 0, 1/3, 0, 0],
              [1/2, 1/2, 0, 0, 1/2],
              [0, 0, 1/3, 0, 0],
              [0, 0, 1/3, 0, 0]])
E3 = np.array([1/5, 1/5, 1/5, 1/5, 1/5])

M4= np.array([[0, 1/2, 0, 1, 1/2, 0],
              [1/2, 0, 1/3, 0, 0, 0],
              [1/2, 1/2, 0, 0, 1/2, 0],
              [0, 0, 1/3, 0, 0, 1],
              [0, 0, 1/3, 0, 0, 0],
              [0, 0, 0, 1/2, 0, 0]])
E4 = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
page_rank = pagerank(M4, E4, teleportation_prob)
print("PageRank for the four-webpage graph:")
print(page_rank)

# Compute PageRank closed
page_rank_closed = pagerank_closed(M4, E4, teleportation_prob)
print("PageRank closed form for the four-webpage graph:")
print(page_rank_closed)