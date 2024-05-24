from garnet.config.instruments import beamlines
from garnet.reduction.data import DataModel


def test_white():
    corelli = DataModel(beamlines["CORELLI"])

    assert corelli.laue is True

    mandi = DataModel(beamlines["MANDI"])

    assert mandi.laue is True

    snap = DataModel(beamlines["SNAP"])

    assert snap.laue is True

    topaz = DataModel(beamlines["TOPAZ"])

    assert topaz.laue is True


def test_monochromatic():
    demand = DataModel(beamlines["DEMAND"])

    assert demand.laue is False

    wand2 = DataModel(beamlines["WAND²"])

    assert wand2.laue is False
