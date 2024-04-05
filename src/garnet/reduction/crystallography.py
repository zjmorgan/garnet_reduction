from mantid.geometry import PointGroupFactory, SpaceGroupFactory

point_groups = PointGroupFactory.getAllPointGroupSymbols()
space_groups = SpaceGroupFactory.getAllSpaceGroupSymbols()

point_crystal = {}
point_lattice = {}
point_laue = {}

for point_group in point_groups:
    pg = PointGroupFactory.createPointGroup(point_group)
    point_crystal[pg.getHMSymbol()] = pg.getCrystalSystem().name
    point_lattice[pg.getHMSymbol()] = pg.getLatticeSystem().name
    point_laue[pg.getHMSymbol()] = pg.getLauePointGroupSymbol()

space_point = {}
space_number = {}

for space_group in space_groups:
    sg = SpaceGroupFactory.createSpaceGroup(space_group)
    space_point[sg.getHMSymbol()] = sg.getPointGroup().getHMSymbol()
    space_number[sg.getHMSymbol()] = sg.getNumber()
