# coding: utf8
from collections import deque, Counter
import pdb
import math

from model.Direction import Direction
from model.TileType import TileType
from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
# http://russianaicup.ru/forum/index.php?topic=557.0
# https://github.com/Russian-AI-Cup-2015
# visualisator https://github.com/JustAMan/russian-ai-cup-visual/releases/tag/0.1
# 2016: https://github.com/JustAMan/russian-ai-cup-visual/releases/tag/0.12
# best example: https://github.com/asanakoy/ai_cup_2015_code_race


class cached_property(object):
    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class MyStrategy(object):
    _init_game = False

    def init_game(self, me: Car, world: World, game: Game):
        if not getattr(self, '_init_game', False):
            self.__class__._init_game = True

            Meta.game = game
            Meta.world = world
            world.waypoints_track = deque(map(Point, world.waypoints))
            LoggerData.print_map()

            # Step(Point(me), world.starting_direction)
            self.track = deque([(Point(me), world.starting_direction)])
            cost = t_map(self.track)

            LoggerData.print_map(self.track)
            LoggerData.print_track(self.track)

            assert len(self.track) > 0, 'Wrong initialize track'
            assert len(self.track) == cost, 'cost must be equal to len(track)'

            if not (set(map(tuple, world.waypoints)) <=
                    set(map(lambda s: (s[0].xi, s[0].yi), self.track))):
                print(f'Track: {self.track}')
                raise Exception('Not all waypoints contains in track')

            self.log_to_file1 = LoggerData(r'./results1.csv')
            self.log_to_file2 = LoggerData(r'./results2_keyboard.csv')

    def get_next_steps(self, me: Car, length=5):
        self.synch_track_position(me)
        return tuple(self.track)[1:length+1]

    def get_next_step(self, me: Car):
        self.synch_track_position(me)
        return self.track[1]

    def synch_track_position(self, me: Car):
        """deprecated"""
        synch_deque(self.track, Point(me), strict=True)

    def move(self, me: Car, world: World, game: Game, move: Move):
        car = list(filter(lambda c: c.player_id == 2, world.cars))[0]
        self.init_game(me, world, game)
        self.calc_wheel_turn(me, move)
        # self.calc_brake()
        # self.calc_engine()
        # self.calc_nitro()
        self.calc_bonuses(move)
        self.logged(me, world, game, move)

    def calc_wheel_turn(self, me, move):
        move.engine_power = 1.0  

        next_point, next_dir = self.get_next_step(me)
        orient_point = next_point + 0.5
        # nextWaypointX = (next_point.xi + 0.5) * Meta.game.track_tile_size
        # nextWaypointY = (next_point.yi + 0.5) * Meta.game.track_tile_size

        cornerTileOffset = 0.25 #* Meta.game.track_tile_size

        nextTile = next_point.tile
        if nextTile == Tile.LTC:  # Dir.RIGHT
            orient_point += cornerTileOffset
            # nextWaypointX += cornerTileOffset
            # nextWaypointY += cornerTileOffset
        elif nextTile == Tile.RTC:  # Dir.DOWN
            orient_point += [-cornerTileOffset, cornerTileOffset]
            # nextWaypointX -= cornerTileOffset
            # nextWaypointY += cornerTileOffset
        elif nextTile == Tile.LBC:  # Dir.UP
            orient_point += [cornerTileOffset, -cornerTileOffset]
            # nextWaypointX += cornerTileOffset
            # nextWaypointY -= cornerTileOffset
        elif nextTile == Tile.RBC:  # Dir.LEFT
            orient_point -= cornerTileOffset
            # nextWaypointX -= cornerTileOffset
            # nextWaypointY -= cornerTileOffset

        # angleToWaypoint = me.get_angle_to(nextWaypointX, nextWaypointY)
        angleToWaypoint = me.get_angle_to(*orient_point.x_y)
        speedModule = math.hypot(me.speed_x, me.speed_y)

        move.wheel_turn = angleToWaypoint * 32.0 / math.pi
        if nextTile in (Tile.LTC, Tile.RTC, Tile.LBC, Tile.RBC):
            move.engine_power = 0.85

        if ((speedModule ** 2) * abs(angleToWaypoint) > 2.5 * 2.5 * math.pi):
            move.brake = True

    def calc_nitro(self):
        # if world.tick > game.initial_freeze_duration_ticks:
        #     uniq_next_dirs = set(map(lambda s: s[1], self.get_next_steps(me)))
        #     if len(uniq_next_dirs) == 1:
        #         move.use_nitro = True
        pass
    def calc_brake(self):
        pass
    def calc_bonuses(self, move):
        move.throw_projectile = True
        move.spill_oil = True
        pass
        # b = world.bonuses[0]
    def logged(self, me: Car, world: World, game: Game, move: Move):
        # if world.tick % 300 == 0:
        #     print(self.get_next_steps(me))
        if world.tick >= 180:
            self.log_to_file1.write(me, move, game, world)
            self.log_to_file2.write(2, None, game, world)
        # print(me.angular_speed)
        # print(me.get_angle_to(me.next_waypoint_x * tts,me.next_waypoint_y * tts))
        # print('  ', me.get_distance_to(me.next_waypoint_x * tts,me.next_waypoint_y * tts))


#TODO: map=map10
def tgb(variants, to_point):
    """Traversing graph(game_map) in breadth.
    :param list variants: initialized list of subtracks with first steps
    :param Point to_point: end point
    :return int: index of variant, which first go to to_point
    """
    ext_variants = []
    del_variants = []
    if len(variants) > 30:
        variants.sort(key=lambda v: max(Counter(map(lambda s: s[1], v)).values()), reverse=True)
        del variants[5:]

    for index, subtrack in enumerate(variants):
        next_steps = Meta.next_steps(subtrack[-1])
        next_steps = list(filter(lambda s: s not in subtrack, next_steps))

        if to_point == subtrack[-1][0]:
            return index
        if not next_steps:
            del_variants.append(subtrack)

        if len(next_steps) == 1:
            subtrack.append(next_steps[0])
        else:
            # next_steps.sort(key=lambda s: s[0].dist(to_point))
            ext_variants.extend([subtrack + deque([step]) for step in next_steps])
            del_variants.append(subtrack)
    if ext_variants:
        variants.extend(ext_variants)
    for v in del_variants:
        if v in variants: variants.remove(v)
    return tgb(variants, to_point)


def t_map(track, cost=1, max_cost=None):
    """Построение оптимального трека движения по карте.
    :param deque track: Трек, в котором уже содержится начальная точка.
    :param Step|tuple cur_step: Начальный шаг.
    :param int cost: Optional. Текущая длина трека.
    :param int max_cost: Optional. Предполагаемая максимальная длина трека.
    :return int: длина трека."""
    cur_step = track[-1]
    next_steps = Meta.next_steps(cur_step)

    # если нет следующих шагов
    if not next_steps:
        return cost
    if track[0] in next_steps:
        # if start_step in next_steps: then cur_step is finish_step
        return cost
    if max_cost and cost >= max_cost:
        return cost

    # если количество следующих шагов равно 1
    if len(next_steps) == 1:
        track.append(next_steps[0])
        return t_map(track, cost + 1, max_cost)
    else:
        # иначе - если развилка... ПОКА НЕ РАБОТАЕТ!
        # создается список вариантов прохода до следующей waypoint методом обхода в ширину
        # next_steps.sort(key=lambda s: s[0].dist(to_point))
        # variants = [deque([step]) for step in next_steps]
        variants = [deque([cur_step])]

        # обход в ширину до следующей контрольной точки.
        subtrack = variants[tgb(variants, Meta.world.waypoints_track[1])]
        del subtrack[0]
        if subtrack[-1] == track[0]:
            del subtrack[-1]
        track.extend(subtrack)
        return t_map(track, cost + len(subtrack), max_cost)


def synch_deque(deque, item, strict=None):
    # step or Point
    tmp = deque[0]
    if isinstance(tmp, tuple):
        # use step
        if not isinstance(item, tuple):
            # uzko!!!! [0]
            item = list(filter(lambda s: s[0] == item, deque))[0]
    else:
        # use point
        if not isinstance(item, Point):
            item = item[0]

    if item in deque:
        ind = deque.index(item)
        if -1 <= ind <= 1:
            deque.rotate(-ind)
    elif strict is None:
        pass
    elif strict:
        raise Exception(f'{item} is not contains in deque')
    else:
        # relevante rotate
        pass


class LoggerData(object):
    """Class logged info to console or file"""

    class bcolors:
        HEADER = '\033[95m'  # purple
        OKBLUE = '\033[94m'  # blue
        OKGREEN = '\033[92m' # green
        WARNING = '\033[93m' # yellow
        FAIL = '\033[91m'    # red
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        INVERSE = '\033[7m'

    def __init__(self, filename=None):
        self.file = None
        if filename:
            self.file = open(filename, 'w')
            self.file.write(self._get_header())

    @classmethod
    def print_map(cls, track=None):
        if track:
            track = cls._init_track(track)

        skip = len(str(Meta.world.height))
        print(f'{" ":{skip}} ', end='')
        for y in range(Meta.world.width):
            print(str(y)[-1], end='')
        print()
        for y in range(Meta.world.height):
            print(f'{y:{skip}} ', end='')
            for x in range(Meta.world.width):
                p = Point(xi=x, yi=y)
                tile = Tile._REPRESENT[p.tile]
                if p in Meta.world.waypoints_track:
                    tile = cls.bcolors.OKGREEN + tile + cls.bcolors.ENDC
                print(tile, end='')

            if track:
                print(end='   ')
                for x in range(Meta.world.width):
                    print(Tile._REPRESENT[track[x][y]], end='')
            print()
        print(f'Waypoints: {Meta.world.waypoints}')

    @classmethod
    def _init_track(cls, track):
        track_map = []
        for c in Meta.world.tiles_x_y:
            track_map.append([Tile.E for _ in c])
        _tiles = filter(lambda i: i[0] not in [Tile.LT, Tile.RT, Tile.BT, Tile.TT, Tile.CROSS], Tile._DIR.items())
        tiles = {dd:t for t,dds in _tiles for dd in dds}
        # 1 6
        # 11 1
        # 1 4
        # 5 3
        # 4 4
        # 5 5
        # 4 4
        # 6 5
        # 6 4
        # 3 5
        # 6 5
        # 3 3
        # 6 6
        # 5 3
        # 4 6
        # 4 3
        # 6 6
        # 3 3
        # 6 6
        # 3 5
        # 4 4
        # 6 5
        # 3 4
        layout = {
            Tile.V: {Tile.H: Tile.CROSS, Tile.V: Tile.V, Tile.LBC:Tile.RT},
            Tile.H: {Tile.V: Tile.CROSS, Tile.RBC: Tile.TT, Tile.H: Tile.H, Tile.LBC: Tile.TT},
            Tile.LTC: {Tile.H: Tile.BT},
            Tile.RTC: {Tile.H: Tile.BT, Tile.V: Tile.LT},
            Tile.LBC: {Tile.H: Tile.TT},
            Tile.RBC: {Tile.H: Tile.TT, Tile.V: Tile.LT},
            # Tile.LT: {Tile.: Tile.},
            # Tile.RT: {Tile.: Tile.},
            # Tile.TT: {Tile.: Tile.},
            # Tile.BT: {Tile.: Tile.},
            # Tile.CROSS: {Tile.: Tile.},
            # Tile.E: {Tile.: Tile.},
        }

        for i in range(len(track)):
            p, di = track[0]
            np, ndi = track[1]
            tile = track_map[p.xi][p.yi]
            t = tiles.get((di, ndi), Tile.E)
            try:
                track_map[p.xi][p.yi] = t if tile == Tile.E else layout[tile][t]
            except:
                print(tile, t)
            track.rotate(1)
        return track_map

    @classmethod
    def print_track(cls, track):
        print(f'Track: from {track[0]} to {track[-1]}', end='\n   ')
        for step in track:
            print(Dir._REPRESENT[step[1]], end=',')
        print()
        print(f'Length of track: {len(track)}')

    def _get_header(self):
        return (
            'tick, x, y, xi, yi, xr, yr, engine_power, move.engine_power, '
            'wheel_turn, move.wheel_turn,|, angle, angular_speed, speed_x, speed_y')

    def write(self, car, move, game, world):
        if isinstance(car, int):
            car = next(filter(lambda c: c.player_id == car, world.cars))
        p = Point(car)

        row_data = (
            f'{world.tick}, {p.x}, {p.y}, {p.xi}, {p.yi}, {p.xr}, {p.yr}, {car.engine_power:.3}, {move.engine_power if move else 0.:.3}, '
            f'{car.wheel_turn:.3}, {move.wheel_turn if move else 0.:.3},|, {car.angle:.3}, {car.angular_speed:.3},  {car.speed_x:.3}, {car.speed_y:.3}')
        if self.file:
            self.file.write(row_data)
        # elif pred:
        #     plotly.update(row_data)
        else:
            print(row_data)


class Meta(object):
    game = None
    world = None

    @classmethod
    def next_steps(cls, step, only_direct=None) -> list:
        """Возвращается список следующих step'ов. Если указан only_direct,
        то оставит только это значение в списке.

        :param Step|tuple step: Текущий шаг.
        :param only_direct: optional. Фильтрует по указанному направлнию.
        :return list: Список следующих step'ов.
        """
        synch_deque(Meta.world.waypoints_track, step)

        steps = []
        dirs = Meta.get_dirs(step)
        if not dirs:
            return []
        for d in dirs:
            # if only_direct is not None and only_direct != d:
            #     continue
            steps.append((step[0].next_point(d), d))
        return steps

    @classmethod
    def relevant_next_point(cls, cur_step) -> tuple:
        """
        :param Step|tuple|Point cur_step:
        :return:
        """
        cur_point = cur_step[0] if isinstance(cur_step, tuple) else cur_step
        # move head to cur_step
        # if cur_point() in Meta.world.waypoints_track:
        #     Meta.world.waypoints_track.rotate(
        #         -Meta.world.waypoints_track.index(cur_point()))
        synch_deque(Meta.world.waypoints_track, cur_step[0])

        direct = cur_point.get_dirs(Meta.world.waypoints_track[1])

        next_steps = Meta.next_steps(cur_step, direct if isinstance(direct, int) else None)

        if len(next_steps) == 1:
            return next_steps[0]
        else:
            print(next_steps)
            # return next_steps[0]
            raise Exception('halt')
            # define next way_point
            # p = cur_step[0]()
            # if p in cls.world.waypoints_track:
            #     cls.world.waypoints_track.rotate(
            #         -cls.world.waypoints_track.index(p) - 1)
            # # relevant - sorting by distance to next way_point
            # next_steps.sort(key=lambda s:
            #     s[0].dist(Point(cls.world.waypoints_track[0])))
            #
            # return next_steps[0]

    @classmethod
    def get_dirs(cls, tile, dir_in=None) -> list:
        """

        :param TileType|Step|tuple tile:
        :param int dir_in:
        :return:
        """
        if isinstance(tile, tuple):  # Step
            # tile is step = (point, in_dir), where point -> tile

            dirs = filter(lambda p: p[0] == tile[1],
                          Tile._DIR[Meta.world.tiles_x_y[tile[0].xi][tile[0].yi]])
        else:  # tile
            dirs = filter(lambda do: do[0] == dir_in, Tile._DIR[tile])
        # assert len(dirs) == 1
        return list(map(lambda p: p[1], dirs))


class Point(object):
    """class of point"""

    def __init__(self, x: int or Car or list=None, y=None, xi=None, yi=None):
        """Point(me), Point(x, y), Point(*(x, y)), Point(xi=3, yi=3) ==  Point([3, 3])"""
        if hasattr(x, 'x') and hasattr(x, 'y'):
            x, y = x.x, x.y
        elif isinstance(x, list):
            xi, yi = x

        if xi is not None and yi is not None:
            x, y = int(xi * Meta.game.track_tile_size), int(yi * Meta.game.track_tile_size)

        if x is not None and y is not None:
            self.x, self.y = int(x), int(y)
        else:
            self.x = 0
            self.y = 0
    # original
    @property
    def x_y(self, *args, **kwargs):
        return [self.x, self.y]
    # integer, tile number
    @property
    def xi(self):
        return int(self.x / Meta.game.track_tile_size)
    @property
    def yi(self):
        return int(self.y / Meta.game.track_tile_size)
    @property
    def xi_yi(self):
        return [self.xi, self.yi]
    #relative
    @property
    def xr(self):
        return int(self.x % Meta.game.track_tile_size)
    @property
    def yr(self):
        return int(self.y % Meta.game.track_tile_size)
    @property
    def xr_yr(self):
        return [self.xr, self.yr]

    @property
    def tile(self) -> int:
        return Meta.world.tiles_x_y[self.xi][self.yi]

    def __add__(self, other):
        """self + Point, self + int"""
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        elif isinstance(other, int):
            return Point(self.x + other, self.y + other)
        elif isinstance(other, float):
            assert -1 <= other <= 1, 'unnormalize add tile'
            return Point(self.x + other * Meta.game.track_tile_size, 
                         self.y + other * Meta.game.track_tile_size)
        elif isinstance(other, list):
            assert -1 <= other[0] <= 1, 'unnormalize add tile'
            assert -1 <= other[1] <= 1, 'unnormalize add tile'
            assert isinstance(other[0], float), 'unnormalize add tile'
            assert isinstance(other[1], float), 'unnormalize add tile'
            return Point(self.x + other[0] * Meta.game.track_tile_size, 
                         self.y + other[1] * Meta.game.track_tile_size)
        else:
            raise Exception()
    def __iadd__(self, other):
        if isinstance(other, Point):
            self.x += other.x 
            self.y += other.y
        elif isinstance(other, int):
            self.x += other
            self.y += other
        elif isinstance(other, float):
            assert -1 <= other <= 1, 'unnormalize add tile'
            self.x += int(other * Meta.game.track_tile_size) 
            self.y += int(other * Meta.game.track_tile_size)
        elif isinstance(other, list):
            assert -1 <= other[0] <= 1, 'unnormalize add tile'
            assert -1 <= other[1] <= 1, 'unnormalize add tile'
            assert isinstance(other[0], float), 'unnormalize add tile'
            assert isinstance(other[1], float), 'unnormalize add tile'
            self.x += int(other[0] * Meta.game.track_tile_size)
            self.y += int(other[1] * Meta.game.track_tile_size)
        else:
            raise Exception() 
        return self
    def __sub__(self, other):
        """self - Point, self - int"""
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        elif isinstance(other, int):
            return Point(self.x - other, self.y - other)
        elif isinstance(other, float):
            assert -1 <= other <= 1, 'unnormalize add tile'
            return Point(self.x - other * Meta.game.track_tile_size, 
                         self.y - other * Meta.game.track_tile_size)
        elif isinstance(other, list):
            assert -1 <= other[0] <= 1, 'unnormalize add tile'
            assert -1 <= other[1] <= 1, 'unnormalize add tile'
            assert isinstance(other[0], float), 'unnormalize add tile'
            assert isinstance(other[1], float), 'unnormalize add tile'
            return Point(self.x - other[0] * Meta.game.track_tile_size, 
                         self.y - other[1] * Meta.game.track_tile_size)
        else:
            raise Exception()
    def __isub__(self, other):
        if isinstance(other, Point):
            self.x -= other.x 
            self.y -= other.y
        elif isinstance(other, int):
            self.x -= other
            self.y -= other
        elif isinstance(other, float):
            assert -1 <= other <= 1, 'unnormalize add tile'
            self.x -= int(other * Meta.game.track_tile_size) 
            self.y -= int(other * Meta.game.track_tile_size)
        elif isinstance(other, list):
            assert -1 <= other[0] <= 1, 'unnormalize add tile'
            assert -1 <= other[1] <= 1, 'unnormalize add tile'
            assert isinstance(other[0], float), 'unnormalize add tile'
            assert isinstance(other[1], float), 'unnormalize add tile'
            self.x -= int(other[0] * Meta.game.track_tile_size)
            self.y -= int(other[1] * Meta.game.track_tile_size)
        else:
            raise Exception() 
        return self
    def __eq__(self, other):
        """self == Point, self == [int, int]"""
        if isinstance(other, Point):
            return (self.xi == other.xi) and (self.yi == other.yi)
        elif isinstance(other, list):
            return (self.xi == other[0]) and (self.yi == other[1])
        else:
            raise Exception()
    def __str__(self):
        return f'({self.xi}, {self.yi})'
    def __repr__(self):
        return f'Point: {self.__str__()}'

    def dist(self, other: 'Point') -> float:
        """d = sqrt(x*x + y*y)"""
        p = self - other
        return math.sqrt(p.x ** 2 + p.y ** 2)

    # def dist2(self, other: 'Point') -> float:
    #     """d = sqrt(x*x + y*y)"""
    #     p = self - other
    #     return math.sqrt(p.x ** 2 + p.y ** 2)
    
    def next_point(self, dir_out: int) -> 'Point':
        if dir_out == Dir.LEFT:
            p = self + Point(xi=-1, yi=0)
        elif dir_out == Dir.RIGHT:
            p = self + Point(xi=1, yi=0)
        elif dir_out == Dir.UP:
            p = self + Point(xi=0, yi=-1)
        elif dir_out == Dir.DOWN:
            p = self + Point(xi=0, yi=1)
        else:
            raise Exception(f'{dir_out} is error direction!')
        return p

    def get_dirs_to(self, other) -> list:
        """deprecated. it is not always"""
        dirs = []
        p = self - other
        if p.xi < 0:
            dirs.append(Dir.RIGHT)
        elif p.xi > 0:
            dirs.append(Dir.LEFT)

        if p.yi < 0:
            dirs.append(Dir.DOWN)
        elif p.yi > 0:
            dirs.append(Dir.UP)
        return dirs


class Step(object):
    def __init__(self, p, dir_in, nonstrictly=False):
        self.point = p
        self.dir = dir_in
        # self.nonstrictly=nonstrictly
    def __eq__(self, other):
        if getattr(self, 'nonstrictly', False) or getattr(other, 'nonstrictly', False):
            return self.point == other.point
        else:
            return self.point == other.point and self.dir == other.dir
    def __repr__(self):
        return f'{self.point}, dir: {Dir._REPRESENT[self.dir]}'


class Dir(Direction):
    _REPRESENT = {
        Direction.LEFT: '˂',
        Direction.RIGHT: '˃',
        Direction.UP: '˄',
        Direction.DOWN: '˅',
    }


class Tile(TileType):
    """Meta info of tiles"""
    V = TileType.VERTICAL               # ║
    H = TileType.HORIZONTAL             # ═
    LTC = TileType.LEFT_TOP_CORNER      # ╔
    RTC = TileType.RIGHT_TOP_CORNER     # ╗
    LBC = TileType.LEFT_BOTTOM_CORNER   # ╚
    RBC = TileType.RIGHT_BOTTOM_CORNER  # ╝
    LT = TileType.LEFT_HEADED_T         # ╣
    RT = TileType.RIGHT_HEADED_T        # ╠
    TT = TileType.TOP_HEADED_T          # ╩
    BT = TileType.BOTTOM_HEADED_T       # ╦
    CROSS = TileType.CROSSROADS         # ╬
    E = TileType.EMPTY                  # ░
    U = TileType.UNKNOWN                # ▒

    _DIR = { # (IN, OUT) - how can go tile
        V: [(Dir.DOWN, Dir.DOWN), (Dir.UP, Dir.UP)],
        H: [(Dir.LEFT, Dir.LEFT), (Dir.RIGHT, Dir.RIGHT)],
        LTC: [(Dir.UP, Dir.RIGHT), (Dir.LEFT, Dir.DOWN)],
        RTC: [(Dir.UP, Dir.LEFT), (Dir.RIGHT, Dir.DOWN)],
        LBC: [(Dir.DOWN, Dir.RIGHT), (Dir.LEFT, Dir.UP)],
        RBC: [(Dir.DOWN, Dir.LEFT), (Dir.RIGHT, Dir.UP)],
        LT: [(Dir.RIGHT, Dir.UP), (Dir.RIGHT, Dir.DOWN),
             (Dir.UP, Dir.UP), (Dir.UP, Dir.LEFT),
             (Dir.DOWN, Dir.LEFT), (Dir.DOWN, Dir.DOWN)],
        RT: [(Dir.LEFT, Dir.DOWN), (Dir.LEFT, Dir.UP),
             (Dir.DOWN, Dir.DOWN), (Dir.DOWN, Dir.RIGHT),
             (Dir.UP, Dir.RIGHT), (Dir.UP, Dir.UP),],
        TT: [(Dir.DOWN, Dir.RIGHT), (Dir.DOWN, Dir.LEFT),
             (Dir.LEFT, Dir.LEFT), (Dir.LEFT, Dir.UP),
             (Dir.RIGHT, Dir.UP), (Dir.RIGHT, Dir.RIGHT)],
        BT: [(Dir.UP, Dir.RIGHT), (Dir.UP, Dir.LEFT),
             (Dir.LEFT, Dir.LEFT), (Dir.LEFT, Dir.DOWN),
             (Dir.RIGHT, Dir.DOWN), (Dir.RIGHT, Dir.RIGHT)],
        CROSS: [(Dir.UP, Dir.LEFT), (Dir.UP, Dir.UP), (Dir.UP, Dir.RIGHT),
                (Dir.DOWN, Dir.RIGHT), (Dir.DOWN, Dir.DOWN), (Dir.DOWN, Dir.LEFT),
                (Dir.LEFT, Dir.DOWN), (Dir.LEFT, Dir.LEFT), (Dir.LEFT, Dir.UP),
                (Dir.RIGHT, Dir.UP), (Dir.RIGHT, Dir.RIGHT), (Dir.RIGHT, Dir.DOWN), ],
        E: [],
        U: [],
    }

    _REPRESENT = {
        V: '║',
        H: '═',
        LTC: '╔',
        RTC: '╗',
        LBC: '╚',
        RBC: '╝',
        LT: '╣',
        RT: '╠',
        TT: '╩',
        BT: '╦',
        CROSS: '╬',
        E: '░',
        U: '▒',
    }
