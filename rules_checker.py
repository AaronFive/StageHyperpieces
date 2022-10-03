import play_parsing


def non_empty_scene_intersection(s1, s2):
    return any(x in s1 for x in s2)


# Rule 1 : always a common character between two successive scenes in an act
def check_rule_1_act(act):
    transgressions = 0
    for i in range(len(act) - 1):
        transgressions += not (non_empty_scene_intersection(act[i], act[i + 1]))
    return transgressions


def check_rule_1_play(play):
    """Takes a play given as a list of acts and returns the number of rule 1 transgressions"""
    transgressions = sum(check_rule_1_act(act) for act in play)
    nb_transitions = sum(len(act) - 1 for act in play)
    if nb_transitions != 0:
        return transgressions != 0, 100 * transgressions / nb_transitions
    else:
        return False, 0


# Rule 2 : no common characters at the frontiers of acts
def check_rule_2_play(play):
    """Takes a play given as a list of acts and returns the number of rule 2 transgressions"""
    transgressions, nb_scenes = 0, 0

    for i in range(len(play) - 1):
        if len(play[i]) > 1:
            sc_fin, sc_start = play[i][-1], play[i + 1][0]
            transgressions += non_empty_scene_intersection(sc_fin, sc_start)
    if len(play) == 0:
        transgressions = 1
        result = "Empty play"
    else:
        result = 100 * transgressions / len(play)
    return transgressions != 0, result


# Rule 3 : odd number of acts, 5 for tragedies
def check_rule_3_play(play, genre):
    if genre == "Tragédie":
        return len(play) != 5, len(play)
    if genre == "Comédie":
        return len(play) not in [3, 5], len(play)
    return None


# Rule 4 : "balanced" acts
def check_rule_4_play(play):
    """Takes a play and returns the difference between the longest and shortest act in terms of scenes"""
    scene_lengths = [len(acts) for acts in play]
    return True, max(scene_lengths) - min(scene_lengths)


# Rule 5 : if a character leaves the stage, he doesn't come back until the next act
def check_rule_5_act(act, chars):
    char_last_scene = dict()
    problematic_chars = set()
    for (nb_scene, scene) in enumerate(act):
        for x in scene:
            chars.add(x)
            if x not in char_last_scene:
                char_last_scene[x] = nb_scene
            else:
                if char_last_scene[x] != nb_scene - 1:
                    problematic_chars.add(x)
                char_last_scene[x] = nb_scene
    return problematic_chars, len(problematic_chars)


def check_rule_5_play(play):
    transgressions = 0
    pb_chars = set()
    chars = set()
    nb_acts = len(play)
    for a in play:
        pc, nb_pc = check_rule_5_act(a, chars)
        transgressions += nb_pc
        pb_chars = pb_chars.union(pc)
    nb_chars = len(chars)
    if nb_chars == 0:
        pb_chars_proportion = None
    else:
        pb_chars_proportion = len(pb_chars) / nb_chars
    if nb_acts == 0:
        pb_acts_proportion = None
    else:
        pb_acts_proportion = transgressions / nb_acts
    return transgressions != 0, (pb_chars_proportion, pb_acts_proportion)


if __name__ == "__main__":
    pass
