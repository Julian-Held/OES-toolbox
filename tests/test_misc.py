"""Miscellaneoustests for small helper functions"""

import pytest

from OES_toolbox.molecules import map_param_to_label, MOLECULE_DB_LABELS


def test_map_param_to_label():
    # Test that known molecule labels are correctly mapped
    assert map_param_to_label("T_rot") == "Trot"
    assert map_param_to_label("T_vib") == "Tvib"

    for key, label in MOLECULE_DB_LABELS.items():
        param_name = f"param_{key}"
        assert map_param_to_label(param_name) == f"param {label}"
        param_name = f"T_rot_{key}"
        assert map_param_to_label(param_name) == f"Trot {label}"
        param_name = f"T_vib_{key}"
        assert map_param_to_label(param_name) == f"Tvib {label}"

    # Test that unknown molecule labels return the original key
    param_name = "param_UNKNOWN"
    assert map_param_to_label(param_name) == "param UNKNOWN"

    # Test that T_rot and T_vib parameters are returned as-is when they have less than 3 parts
    assert map_param_to_label("T_rot_H2") == "Trot H2"
    assert map_param_to_label("T_vib_H2") == "Tvib H2"
