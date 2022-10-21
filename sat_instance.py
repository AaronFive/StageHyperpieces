import play_parsing
import parameterized_matching
from scene_speech_repartition import normalize_scene

def get_alphabet_size(s_1, s_2):
    return max(max(ord(x) for x in s_1), max(ord(x) for x in s_2)) -64


def get_pi(alphabet_size):
    pi = []
    for i in range(alphabet_size):
        pi.append(chr(65 + i))
    return pi


# We fix the following enumeration of variables xi,j for 1< i < n, then for 1< j < m, then ya,b for a, then for b.
def make_corresp_dictionnaries(string_1, string_2):
    n, m = len(string_1), len(string_2)
    pi_size = get_alphabet_size(string_1, string_2)
    pi = get_pi(pi_size)
    x_dict, y_dict = dict(), dict()
    for i in range(n):
        for j in range(m):
            x_dict[i, j] = n * i + j
    for a in pi:
        for b in pi:
            y_dict[a, b] = n * m + pi_size * (ord(a)-65) + ord(b) - 65
    return x_dict, y_dict


def no_double_i_clause(x_dict, i, j1, j2):
    return " ".join([str(-x_dict[i, j1]), str(-x_dict[i, j2])])


def no_double_j_clause(x_dict, i1, i2, j):
    return " ".join([str(-x_dict[i1, j]), str(-x_dict[i2, j])])


def no_crossing_clause(x_dict, i1, i2, j1, j2):
    return " ".join([str(-x_dict[i1, j1]), str(-x_dict[i2, j2])])


def function_clause(y_dict, a, b1, b2):
    return " ".join([str(-y_dict[a, b1]), str(-y_dict[a, b2])])


def match_clause(x_dict, y_dict, i, j, u, v):
    return " ".join([str(-x_dict[i, j]), str(y_dict[u[i], v[j]])])


def make_sat_instance(comments, string_1, string_2):
    n, m = len(string_1), len(string_2)
    pi_size = get_alphabet_size(string_1, string_2)
    pi = get_pi(pi_size)
    x_dict, y_dict = make_corresp_dictionnaries(string_1, string_2)
    comments_list = [" ".join(["c", x]) for x in comments]
    comment_string = "\n".join(comments_list)

    top = n*m

    # No_Double_i
    clauses_i = []
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                if j1 != j2:
                    clauses_i.append((" ".join([f"{top}", no_double_i_clause(x_dict, i, j1, j2)])))
    clauses_i_string = "\n".join(clauses_i)

    # No_Double_j
    clauses_j = []
    for j in range(m):
        for i1 in range(n):
            for i2 in range(n):
                if i1 != i2:
                    clauses_j.append((" ".join([f"{top}",no_double_j_clause(x_dict, i1, i2, j)])))
    clauses_j_string = "\n".join(clauses_j)

    # No_Crossing
    clauses_crossings = []
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for j1 in range(m):
                for j2 in range(j1 + 1, m):
                    clauses_crossings.append(" ".join([f"{top}", no_crossing_clause(x_dict, i1, i2, j1, j2)]))
    clauses_crossings_string = "\n".join(clauses_crossings)

    # Function
    clauses_function = []
    for a in pi:
        for b in pi:
            for c in pi:
                if b != c:
                    clauses_function.append(" ".join([f"{top}", function_clause(y_dict, a, b, c)]))
    clauses_function_string = "\n".join(clauses_function)

    # Match
    clauses_match = []
    for i in range(n):
        for j in range(m):
            clauses_match.append((" ".join([f"{top}",match_clause(x_dict, y_dict, i, j, string_1, string_2)])))
    clauses_match_string = "\n".join(clauses_match)

    nbvar = len(string_1) * len(string_2) + pi_size ** 2
    nb_clauses = len(clauses_match) + len(clauses_function) + len(clauses_crossings) + len(clauses_j) + len(clauses_i)

    header = f"p wcnf {nbvar} {nb_clauses} {top}"

    # To maximize

    clauses_max = []
    for i in range(n):
        for j in range(m):
            clauses_max.append(" ".join(["1", str(x_dict[i, j])]))
    clauses_max_string = "\n".join(clauses_max)

    final_string = "\n".join([comment_string, header, clauses_i_string, clauses_j_string, clauses_crossings_string, clauses_function_string, clauses_match_string, clauses_max_string])

    return final_string


def encode_scenes(scene1, scene2):
    u, d1 = normalize_scene(scene1)
    v, d2 = normalize_scene(scene2)
    s = make_sat_instance([str(d1), str(d2)],u,v)
    print(s)
    answer = input("Donnez la réponse de maxHS (lignes avec v uniquement")
    positives = answer.split(" ")[1:]
    positives = [int(x) for x in positives]
    x_dict, y_dict = make_corresp_dictionnaries(u,v)
    print("Littéraux vrais :")
    for x in x_dict:
        if x_dict[x] in positives:
            print(f"x_{x}")
    for y in y_dict:
        if y_dict[y] in positives:
            a,b = y
            print(f"y_{y} ({d1[a]} renommé en {d2[b]}")


if __name__ == "__main__":
    u = "ABC"
    v = "A"
    s = make_sat_instance(["Test"], u,v)
    print(s)
