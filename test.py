import matplotlib.pyplot as plt
global Tdict

Tdict ={ (0,0):1 }

def T(max_length, k, n):
    if (k, n) in Tdict:
        s = Tdict[k, n]
    elif k == 0 or n == max_length:
        s = 1
    else:
        s = sum((T(max_length, p, n + 1) for p in range(k + 1)))
    Tdict[k, n] = s
    return s

#A longueur fixé, graphe de la taille de l'espace de recherche en faisant varie rla distance k
def graph(n, k_max):
    T(n, k_max, 0)
    print(Tdict)
    ab = []
    ordo = []
    for k in range(k_max+1):
        ab.append(k)
        ordo.append(T(n,k,0))
    plt.plot(ab,ordo)
    plt.show()

# A k fixé, graphe de la taille de l'espace de recherche en faisant varier la longueur du mot, n
def graph_n(n_max, k):
    ab = []
    ordo = []
    for n in range(k,n_max + 1):
        ab.append(n)
        ordo.append(T(n, k, 0))
    plt.plot(ab, ordo)
    plt.show()


if __name__ == "__main__":
    graph_n(50, 10)
