"""
Functions to generate random parameterized words,
and random instances of various parameterized matching problems.
"""
import string, random

alphabet = string.ascii_letters


def get_pi(n):
    """Returns a parameterized alphabet of size n, as a list.
    Only works up to n = 52 """
    return list(alphabet[:n])


def random_p_word(pi, l):
    """Returns a random parameterized word on pi of length l, as a list """
    return [random.choice(pi) for _ in range(l)]


def random_renaming_bijective(pi):
    image = pi.copy()
    random.shuffle(image)
    graphe = {x: image[pi.index(x)] for x in pi}
    return lambda x: graphe[x]


def random_renaming_function(pi):
    n = len(pi)
    image = {x: pi[random.randint(0, n - 1)] for x in pi}
    return lambda x: image[x]


def rename(w, f):
    return [f(x) for x in w]


def random_rename_bijective(pi, w):
    f = random_renaming_bijective(pi)
    return rename(w, f)


def random_rename_function(pi, w):
    f = random_renaming_function(pi)
    return rename(w, f)


def random_edit(pi, w, k, deletions=True, insertions=True, substitutions=True):
    """Generates a word on Pi at edit distance <=k from w.
    Available edit operations can be toggled on and off,and are all allowed by default."""
    # The code is a bit convoluted because we have to make sure we do not edit the same position twice,
    # and the length of the word we create may vary.
    # We annotate each position with a boolean indicating if it has been modified already or not.
    n = len(w)
    new_w = [(x, False) for x in w]  # Contains the new word w. A letter is marked True if it has been modified already.

    # Filtering allowed operations
    operations = []
    if deletions:
        operations.append('d')
    if substitutions:
        operations.append('s')
    if insertions:
        operations.append('i')
    operations_done = 0

    while operations_done < k:
        op = random.choice(operations)
        # Deletion
        if op == 'd':
            # Getting an unused position
            rand_pos = random.randint(0, len(new_w) - 1)
            while new_w[rand_pos][1]:
                rand_pos = random.randint(0, len(new_w) - 1)
            # Deleting
            new_w.pop(rand_pos)

        # Substitution
        if op == 's':
            # Getting an unused position
            rand_pos = random.randint(0, len(new_w) - 1)
            while new_w[rand_pos][1]:
                rand_pos = random.randint(0, len(new_w) - 1)
            # Getting an new letter
            random_new_letter = random.choice(pi)
            while random_new_letter == new_w[rand_pos][0]:
                random_new_letter = random.choice(pi)
            # Substituting
            new_w[rand_pos] = (random_new_letter, True)

        # Insertion
        if op == 'i':
            # Getting an unused position
            rand_pos = random.randint(0, len(new_w))  # We choose a position between letters in w, not a position in w

            # Getting an new letter and inserting
            random_new_letter = random.choice(pi)
            new_w.insert(rand_pos, (random_new_letter, True))

        operations_done += 1
    new_w = [x[0] for x in new_w]
    return new_w


def random_PMd_instance(alphabet_size, u_size, k, deletions=True, insertions=True, substitutions=True):
    """Returns random parameterized words u and v on an alphabet of size n,
    such that |u| = u_size, and PMd(u,v) <= k, along with a solution f.
    Allowed edit operations can be togggled on and off, all are authorized by default"""
    pi = get_pi(alphabet_size)
    u = random_p_word(pi, u_size)
    v = random_edit(pi, u, k, deletions, insertions, substitutions)
    f = random_renaming_bijective(pi)
    v = rename(v, f)
    return u, v, {x: f(x) for x in pi}


def random_FM1d_instance(alphabet_size, u_size, k, deletions=True, insertions=True, substitutions=True):
    """Returns random parameterized words u and v on an alphabet of size n,
    such that |u| = u_size, and FM1d(u,v) <= k, along with a solution f.
    Allowed edit operations can be togggled on and off, all are authorized by default"""
    # First we apply edit operations, then we apply a random function f
    pi = get_pi(alphabet_size)
    u = random_p_word(pi, u_size)
    v = random_edit(pi, u, k, deletions, insertions, substitutions)
    f = random_renaming_function(pi)
    v = rename(v, f)
    return u, v, {x: f(x) for x in pi}


def random_FM2d_instance(alphabet_size, u_size, k, deletions=True, insertions=True, substitutions=True):
    """Returns random parameterized words u and v on an alphabet of size n,
    such that |u| = u_size, and FM2d(u,v) <= k, along with a solution f.
    Allowed edit operations can be togggled on and off, all are authorized by default"""
    # We apply f first, and then the edit operations.
    # An actual solution to the FMd2 problem would apply the edit operations in reverse from v.
    pi = get_pi(alphabet_size)
    u = random_p_word(pi, u_size)
    f = random_renaming_function(pi)
    v = rename(u, f)
    v = random_edit(pi, v, k, deletions, insertions, substitutions)
    return u, v, {x: f(x) for x in pi}

if __name__ == "__main__":
    u, v, f = random_FM1d_instance(3,5, 2)
    print(''.join(u), ''.join(v), f)

