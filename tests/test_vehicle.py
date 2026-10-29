import numpy as np

from cardamage.backends.base import RawDet
from cardamage.postprocess import _G
from cardamage.vehicle import Regions, gate


def _det(x1, y1, x2, y2, label="paint_damage"):
    return RawDet(label=label, conf=0.9, box=(x1, y1, x2, y2),
                  poly=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def _reg(vehicle=None, person=None, n=0):
    v, p = np.zeros((_G, _G), bool), np.zeros((_G, _G), bool)
    if vehicle:
        x1, y1, x2, y2 = (int(c * _G) for c in vehicle); v[y1:y2, x1:x2] = True
    if person:
        x1, y1, x2, y2 = (int(c * _G) for c in person); p[y1:y2, x1:x2] = True
    return Regions(vehicle=v, person=p, n_vehicle=n)


def test_outside_vehicle_rule_is_opt_in_and_drops_outside():
    on, off = _det(.1, .1, .3, .3), _det(.7, .7, .9, .9)
    kept, removed = gate([on, off], _reg(vehicle=(0, 0, .5, .5), n=1), min_in_vehicle=0.5)
    assert kept == [on] and [(r[0], r[1]) for r in removed] == [(off, "outside_vehicle")]


def test_pedestrian_is_removed_even_next_to_a_car():
    car_dmg, ped = _det(.1, .1, .3, .3), _det(.7, .2, .8, .8)
    kept, removed = gate([car_dmg, ped], _reg(vehicle=(0, 0, .5, .5), person=(.65, .1, .85, .9), n=1))
    assert kept == [car_dmg] and removed[0][1] == "on_person"


def test_damage_on_car_under_a_passenger_is_kept():
    d = _det(.1, .1, .3, .3)           # inside the car AND inside a person mask (e.g. someone leaning on it)
    kept, _ = gate([d], _reg(vehicle=(0, 0, .5, .5), person=(0, 0, .4, .4), n=1))
    assert kept == [d]


def test_no_vehicle_found_keeps_closeups_but_still_drops_people():
    closeup, ped = _det(.1, .1, .5, .5), _det(.6, .6, .9, .9)
    kept, removed = gate([closeup, ped], _reg(person=(.55, .55, .95, .95), n=0))
    assert kept == [closeup] and removed[0][1] == "on_person"


def test_default_gate_only_removes_people_not_off_vehicle_damage():
    wreck_part = _det(.7, .7, .9, .9)  # not inside any detected vehicle (e.g. a crushed car the detector missed)
    kept, removed = gate([wreck_part], _reg(vehicle=(0, 0, .5, .5), n=1))
    assert kept == [wreck_part] and not removed
