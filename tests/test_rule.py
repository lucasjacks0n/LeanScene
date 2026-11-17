import numpy as np

from leanscene import Transition, find_shots, find_transitions, label_frames


def probs(labels):
    """One-hot probabilities for a list of labels."""
    return np.eye(3, dtype=np.float32)[labels]


def test_thresholds():
    p = np.array([[0.31, 0.69, 0.0], [0.3, 0.7, 0.0], [0.21, 0.0, 0.79], [0.2, 0.0, 0.8]], np.float32)
    assert label_frames(p).tolist() == [0, 1, 0, 2]
    assert label_frames(p, cut=0.6, fade=0.7).tolist() == [1, 1, 2, 2]


def test_fade_runs_are_joined_over_short_gaps():
    gap8 = [2] + [0] * 8 + [2]
    gap9 = [2] + [0] * 9 + [2]
    assert label_frames(probs(gap8)).tolist() == [2] * 10
    assert label_frames(probs(gap9)).tolist() == gap9


def test_transitions_and_shots():
    #         0  1  2  3  4  5  6  7  8  9 10 11
    labels = [0, 0, 0, 1, 0, 0, 2, 2, 2, 0, 0, 0]
    t = find_transitions(np.array(labels))
    assert t == [Transition(3, 3, "cut"), Transition(6, 8, "gradual")]
    # the cut frame opens the new shot; gradual frames belong to neither shot
    assert find_shots(t, len(labels)) == [(0, 2), (3, 5), (9, 11)]


def test_no_transitions_is_one_shot():
    assert find_transitions(np.zeros(5, np.int64)) == []
    assert find_shots([], 5) == [(0, 4)]
    assert find_shots([], 0) == []


def test_transitions_at_the_edges():
    t = find_transitions(np.array([1, 0, 0, 2, 2]))
    assert t == [Transition(0, 0, "cut"), Transition(3, 4, "gradual")]
    assert find_shots(t, 5) == [(0, 2)]


def test_junk_frames_between_two_shots():
    # two cut labels in a row: the frame between the shots belongs to neither
    t = find_transitions(np.array([0, 0, 1, 1, 0, 0]))
    assert t == [Transition(2, 3, "cut")]
    assert find_shots(t, 6) == [(0, 1), (3, 5)]
