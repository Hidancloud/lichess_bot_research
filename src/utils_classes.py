# ys 0 counts from top, right monitor (1920, 1080) with positive x, left - negative x
from mss import mss
import pyautogui as player
from time import sleep, time
import numpy as np
from typing import Tuple, Any
import yaml
from random import shuffle
from stockfish import Stockfish
from math import ceil, log10

# from matplotlib import pyplot as plt

# TODO: probably need to implement explicitly restart function for db when new game starts
# to avoid problems related to singleton re-init (by default it might doesn't work) (see init_color)

# object that is used for making screenshots
sct = mss()  # this line breaks for multithreading and prevents from using keyboard control (can't listen)
# must be fixed to make it init in some object)

# global constants, always the same
letters_to_inds = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
inds_to_letters = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
promotion_cells = {'q': 0, 'n': 1, 'r': 2, 'b': 3}

DEFAULT_STRATEGY_PARAMS = {'debute'}


def cool_round(number, k=4):
    if number == 0:
        return 0
    return round(number, k - 1 - int(ceil(log10(number))))


class Singleton(type):
    """
    Singleton metaclass to prevent creating more than one instance of database
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ChessRelevantData(metaclass=Singleton):
    """
        Data Base class that stores everything related to the game
        Made in Singleton manner so that other classes can use it in easy way
    """

    def __init__(self,
                 config_path: str = '../configs/',
                 be_gentle: bool = False,
                 skip_color: bool = False):
        """
            config_path: path to config with all settings
            total_time_in_minutes: amount of starting time given to bot to play in minutes
            increment_in_seconds: amount of starting time that adds up to bots timer after every move
            be_gentle: False by default, change to True if you want some revenge
        """

        # loading config with all bboxes data
        with open(config_path + 'screenshot_boxes.yaml', 'r') as config_screenshot_boxes_file:
            config_screenshot_boxes = yaml.safe_load(config_screenshot_boxes_file)

        # loading config with all action coordinates data
        with open(config_path + 'click_action_coordinates.yaml', 'r') as config_actions_file:
            config_actions = yaml.safe_load(config_actions_file)

        with open(config_path + 'timer_parameters.yaml', 'r') as config_timer_file:
            config_timer = yaml.safe_load(config_timer_file)

        # bounding boxes in a dictionary format: {'top': top_y, 'left': left_x, 'width': width, 'height': height}
        # (top_x, top_y) coordinates of top left corner of an object
        self.chessboard_bbox = config_screenshot_boxes['bounding_box_chessboard']
        self.side = self.chessboard_bbox['width'] // 8
        self.timer_bbox = config_screenshot_boxes['bounding_box_timer']

        # for each cell pixel coordinate of a background color
        self.chessboard_cell_pixels = [i * self.side + 2 for i in range(8)]

        # green circle indicators of online status on lichess
        self.bot_green_circle = config_screenshot_boxes['my_green_circle']
        self.opponent_green_circle = config_screenshot_boxes['opponent_green_circle']

        self.takeback_red_cross = config_screenshot_boxes['takeback_red_cross']
        self.draw_offer_red_cross = config_screenshot_boxes['draw_offer_red_cross']
        # in general case it has green color if it is opponent's move (timer zone),
        # when opponent afk it is shifted to status circle
        self.afk_gray_green_circle = config_screenshot_boxes['afk_gray_green_circle']
        # only a part of it so that center is blue with 54 value at the red channel
        # 54 red color reference value in the box
        self.defeat_afk_opponent_blue_box = config_screenshot_boxes['defeat_afk_opponent_blue_box']
        # leave page button to force leave the game in case of any problem
        # absolutely blue color, 0 value in red channel
        self.hard_restart_acceptance_box = config_screenshot_boxes['hard_restart_acceptance_box']

        self.actions = config_actions  # click coordinates for every action
        self.gentle = be_gentle  # if True then be toxic as fuck

        # time mode: time to think in minutes + increment in seconds
        self.time_manager = TimeManagement(config_timer)

        # initializes bots color after each game
        self.color = None
        if not skip_color:
            self.init_color()

    def init_color(self):
        # Initializing bots color by recognizing which color rook on bottom left (from bot side)
        # Also it works via pattern matching to make the delay until we actually join into the game
        # It is important to avoid cases in which algorithm starts too early
        is_bots_rook_white = is_bots_rook_black = False
        black_rook_pattern = np.load('../data/patterns/logged_black_rook.npy')
        white_rook_pattern = np.load('../data/patterns/logged_white_rook.npy')

        start_time = time()
        while not is_bots_rook_white and not is_bots_rook_black:
            chessboard = ImageController.get_screenshot(self.chessboard_bbox)

            bots_rook = chessboard[-self.side:, :self.side][:45, :45]  # 45 to avoid artefacts around a or 8 symbols
            is_bots_rook_white = np.all(bots_rook == white_rook_pattern)
            is_bots_rook_black = not is_bots_rook_white and np.all(bots_rook == black_rook_pattern)

            # plt.imshow(bots_rook - white_rook_pattern, cmap='gray')
            # plt.show()
            # print('color:', is_bots_rook_white, is_bots_rook_black)

            # if we wait too late then raise exception
            if time() - start_time > 2.5 * (
                    self.time_manager.minutes * 60 + self.time_manager.increment * 60 + 60):
                # todo: add smarter check with background simple analysis heuristics
                raise TimeoutError('Broken at the color choosing step: too long to wait')

        if is_bots_rook_white:
            self.color = 'White'
        else:
            self.color = 'Black'

    def restart(self):
        self.time_manager.restart()
        self.init_color()

    @staticmethod
    def mirror_move(move: str) -> str:
        """
            Mirrors move in UCI format for Black

            :param move: string representation of a Black's move in UCI format
        """
        def mirror_letter(letter):
            return inds_to_letters[7 - letters_to_inds[letter]]

        def mirror_digit(digit):
            return str(9 - int(digit))

        return mirror_letter(move[0]) + mirror_digit(move[1]) + mirror_letter(move[2]) + mirror_digit(move[3]) + \
            move[4:]  # in case of promotion


class TimeManagement(metaclass=Singleton):
    # TODO: how to make timer work on every call? Maybe my move vs enemy move also track, maybe track smarter
    # TODO: make logs
    # TODO finish init with strategy
    # TODO add properties to the code? why do we need them? do we need them? should we care?

    def __init__(self, strategy_params: dict):
        """
            strategy_params: dictionary with configs from ../configs/timer_parameters.yaml
        """
        self.strategy_params = strategy_params
        self.minutes = strategy_params['timer_settings']['total_time_in_minutes']
        self.increment = strategy_params['timer_settings']['increment_in_seconds']
        self.ping = strategy_params['timer_settings']['ping']
        self.seconds_left = self.minutes * 60.

        self.time_strategy = None
        self._init_time_strategy()

    def _init_time_strategy(self):
        # Initializes list with time to think for every move

        strategy_constants = self.strategy_params['strategy_constants']
        # scale coefficient linearly scales move time component(s)
        # to make it scalable in different time formats
        scale_coefficient = self.minutes + self.increment

        # Debut initialization
        debut_config = strategy_constants['debut_stage']
        # init number of slow, normal and fast moves
        # +1 because in config mentioned value inclusively
        slow_moves_n = np.random.randint(debut_config['slow_moves']['min_number'],
                                         debut_config['slow_moves']['max_number'] + 1)
        normal_moves_n = np.random.randint(debut_config['normal_moves']['min_number'],
                                           debut_config['normal_moves']['max_number'] + 1)
        fast_moves_n = np.random.randint(debut_config['fast_moves']['min_number'],
                                         debut_config['fast_moves']['max_number'] + 1)

        # random time for each move in groups
        slow_moves = scale_coefficient + \
                     np.random.rand(slow_moves_n) * debut_config['slow_moves']['time_multiplier'] * scale_coefficient
        normal_moves = scale_coefficient + np.random.rand(normal_moves_n) * \
                       debut_config['normal_moves']['time_multiplier'] * scale_coefficient
        fast_moves = self.ping + \
                     np.random.rand(fast_moves_n) * debut_config['fast_moves']['time_multiplier'] * scale_coefficient

        # permute moves to make move timings feel natural
        permutes_part1 = [*fast_moves[2:4], normal_moves[0]]
        shuffle(permutes_part1)
        permutes_part2 = [*fast_moves[4:fast_moves_n], *normal_moves[1:], *slow_moves]
        shuffle(permutes_part2)
        moves_time_queue = fast_moves[:2].tolist() + permutes_part1 + permutes_part2

        # middle game initialization
        middle_game_config = strategy_constants['middle_game_stage']
        slow_moves_n = np.random.randint(middle_game_config['slow_moves']['min_number'],
                                         middle_game_config['slow_moves']['max_number'] + 1)
        normal_moves_n = np.random.randint(middle_game_config['normal_moves']['min_number'],
                                           middle_game_config['normal_moves']['max_number'] + 1)
        fast_moves_n = np.random.randint(middle_game_config['fast_moves']['min_number'],
                                         middle_game_config['fast_moves']['max_number'] + 1)

        slow_moves = middle_game_config['slow_moves']['time_multiplier_1'] * scale_coefficient + \
                     np.random.rand(slow_moves_n) * middle_game_config['slow_moves']['time_multiplier_2'] * \
                     scale_coefficient
        normal_moves = middle_game_config['normal_moves']['time_multiplier'] * scale_coefficient + \
                       np.random.rand(normal_moves_n) * scale_coefficient
        fast_moves = self.ping + np.random.rand(fast_moves_n) * middle_game_config['fast_moves']['time_multiplier'] * \
                     scale_coefficient

        permutes_part = [*fast_moves, *normal_moves, *slow_moves]
        shuffle(permutes_part)
        moves_time_queue = moves_time_queue + permutes_part

        # endgame: last moves (by default 200)
        rand = np.random.rand(strategy_constants['endgame_stage']['length'])
        fast_move_threshold = strategy_constants['endgame_stage']['slow_moves']['chance']
        slow_multiplier = strategy_constants['endgame_stage']['slow_moves']['time_multiplier']
        fast_multiplier = strategy_constants['endgame_stage']['fast_moves']['time_multiplier']
        for random in rand:
            if random > fast_move_threshold:
                new_time = slow_multiplier * scale_coefficient + np.random.rand(1)[0] * scale_coefficient
            else:
                new_time = self.ping + np.random.rand(1)[0] * fast_multiplier

            moves_time_queue.append(new_time)

        # do not allow bot to lose because of time troubles
        algorithm_delay = self.strategy_params['timer_settings']['delay_per_move']
        curr_sum = 0
        for i, q in enumerate(moves_time_queue):
            step = q - self.increment + self.ping + algorithm_delay
            curr_sum += step
            if curr_sum > self.minutes * 60 - strategy_constants['endgame_stage']['time_to_sonic_speed']:
                moves_time_queue[i] = self.ping + np.random.rand(1)[0] * fast_multiplier
                curr_sum = curr_sum - step + (self.ping + np.random.rand(1)[0] * fast_multiplier - self.increment)

        self.time_strategy = moves_time_queue

    def reduce_timer(self, seconds_past: float):
        # reduce tracking variable time_left by seconds_past
        self.seconds_left -= seconds_past

    def get_time_to_think(self, move_idx: int, engine: Stockfish, enemy_think_time_features: Any = None) -> int:
        """
        Returns time to think in milliseconds for stockfish engine
            :param move_idx: index of the current move
            :param engine: engine instance
            :enemy_think_time_features: useful for future improvement argument (speed up when opponent speeds up)
        """
        # TODO: probably add argument spent time or need to consider it in the main class logic
        # TODO: consider mirroring opponent time in some way
        if self.seconds_left < self.strategy_params['strategy_constants']['endgame_stage']['time_to_sonic_speed']:
            # random part is to avoid constant time moves, 1.5 is custom parameter
            think_time = self.ping + self.ping * 1.5 * np.random.rand(1)[0]

        else:
            init_time_to_think = self.time_strategy[move_idx]
            # check if our best move is most probably a capture fast (only in 10% of time)
            # then immitate a premove (think x10 faster)
            # else think as normal, but without already spent for premove time (90% time left)
            premove = engine.get_best_move_time(int(init_time_to_think * 1000 * 0.1))
            if engine.will_move_be_a_capture(premove) != Stockfish.Capture.NO_CAPTURE:
                think_time = min(init_time_to_think * 0.2, 0.9)  # make a fast move
            else:
                think_time = init_time_to_think * 0.9

        return int(think_time * 1000)

    def restart(self):
        self.seconds_left = self.minutes * 60.
        self._init_time_strategy()


class Controller:
    def __init__(self):
        self.db = ChessRelevantData()  # ChessRelevantData is singleton
        self.time_manager = TimeManagement(dict())

    @staticmethod
    def _left_click(x: int, y: int):
        # makes a left click on the screen by mouse
        # x and y are coordinates related to screen resolution
        # y starts with 0 from top
        player.leftClick(x=x, y=y)

    def convert_notation_to_move(self, notation: str):
        """
        Gets string notation of a move and converts it to cell coordinates to click to make a move + promotion
        Cell coordinates = numbers from 0 to 7 for each axis, 0,0 is in the top left corner

        Move types:
            *a2a4* (any other move)
            *a7a8q* (promotion)

        :param notation: UCI move notation
        :return:
            (start (x, y) in cell inds,
            end (x, y) in cell inds,
            promotion cell (x, y) or None)
        """

        # white and black notation reversed, so we have to consider it by mirroring coordinates
        y_from = 8 - int(notation[1]) if self.db.color == 'White' else int(notation[1]) - 1
        y_to = 8 - int(notation[3]) if self.db.color == 'White' else int(notation[3]) - 1

        x_from = letters_to_inds[notation[0]] if self.db.color == 'White' else 7 - letters_to_inds[notation[0]]
        x_to = letters_to_inds[notation[2]] if self.db.color == 'White' else 7 - letters_to_inds[notation[2]]

        if len(notation) == 5:
            # promotion case, we need an additional click to choose a piece
            # so here we choose a cell where to click
            promotion_figure = notation[-1]
            promotion_coords = (x_to,
                                promotion_cells[promotion_figure])

        else:
            promotion_coords = None

        return (x_from, y_from), (x_to, y_to), promotion_coords

    def get_cell_coords(self, cell_coords: Tuple) -> list[int]:
        """
        Returns coords of top left pixel of a cell by chessboard cell coords

        :param cell_coords: (x, y) of chessboard cell, x and y from 0 to 7 inclusively
        :return: (x, y) coords in pixels
        """
        return [int(self.db.chessboard_bbox['left'] + self.db.side * cell_coords[0]),
                int(self.db.chessboard_bbox['top'] + self.db.side * cell_coords[1])]

    def make_move(self, best_move_notation: str,
                  center_region: float = 0.4):
        """
        Makes move best_move_notation (move should be in UCI format)
        Move types:
            ordinary move: a2a4
            promotion: a7a8q

        :param best_move_notation: move notation in UCI format
        :param center_region: proportion of the center where to randomly click
        :return:
        """
        # best_move_from: indices of x and y for a cell where we start the move
        # best_move_to: indices of x and y for a cell where we end the move
        # promotion_figure
        best_move_from, best_move_to, promotion_cell_inds = self.convert_notation_to_move(best_move_notation)

        # top left corner of the cell from where to go
        point_square_from = self.get_cell_coords(best_move_from)

        # top left corner of the target cell where to go
        point_square_to = self.get_cell_coords(best_move_to)

        border = (1 - center_region) * 0.5
        randoms = np.random.randint(low=self.db.side * border,
                                    high=self.db.side * (1 - border), size=(3, 2))
        _square_from = point_square_from + randoms[0]

        _square_to = point_square_to + randoms[1]

        player.leftClick(x=_square_from[0], y=_square_from[1])
        # maybe insert sleep 0.05
        player.leftClick(x=_square_to[0], y=_square_to[1])

        if promotion_cell_inds is not None:
            point_square = self.get_cell_coords(promotion_cell_inds)
            _square_choose_fig = point_square + randoms[2]
            sleep(0.3)
            player.leftClick(x=_square_choose_fig[0], y=_square_choose_fig[1])
            print('promoted!')

        # print(_square_from, _square_to)

    def hard_restart(self):
        get_xy = lambda dictionary: [dictionary['x'], dictionary['y']]
        self._left_click(*get_xy(self.db.actions['hard_restart']['play']))  # click "play"
        sleep(0.6)
        if ImageController._check_color_equals(
                self.db.hard_restart_acceptance_box, self.db.hard_restart_acceptance_box['color']):
            # click "leave" if something wrong during the game
            self._left_click(*self.db.actions['click_action_coordinates.yaml']['hard_restart']['approve_leave_page'])
            sleep(0.6)
        # click "create game"
        self._left_click(*get_xy(self.db.actions['hard_restart']['create_game']))
        sleep(0.9)
        # start the opponent search
        self._left_click(*get_xy(self.db.actions['hard_restart']['start_search']))

    def click_new_opponent(self):
        # clicks new opponent to start a new game
        self._left_click(*self.db.actions['click_new_opponent'])

    def decline_takeback(self):
        # declines a takeback proposal by opponent
        self._left_click(*self.db.actions['decline_takeback'])

    def decline_draw_offer(self):
        # declines draw offer
        self._left_click(*self.db.actions['decline_draw_offer'])


class ImageController:
    def __init__(self):
        self.db = ChessRelevantData()

    @staticmethod
    def get_screenshot(bounding_box: dict):
        """
        Makes a screenshot from a bounding_box area,
        bounding_box in a specific format: {'top': top_y, 'left': left_x, 'width': width, 'height': height}
        (top_x, top_y) coordinates of top left corner of an object

        :param bounding_box: dictionary with coordinates of bbox in a specific format
        :return: image as numpy array in 8-bit integer format
        """
        sct_img = sct.grab(bounding_box).raw
        sct_img = np.array(sct_img).reshape((bounding_box['height'], bounding_box['width'], 4))
        return sct_img[..., -2].astype(np.uint8)

    @staticmethod
    def _check_color_equals(box_position: dict, check_color: int | list) -> bool:
        """
        Checks if the middle pixel in the box position is of a color check_color
        :param box_position: dictionary with coordinates of bbox in a specific format (see get_screenshot doc string)
        :param check_color: color(s) to check
        :return: True if color is the same (or listed in check_color list), else False
        """
        obj = ImageController.get_screenshot(box_position)
        center_value = obj[obj.shape[0] // 2, obj.shape[1] // 2]

        # in case of list check them all
        if type(check_color) == list:
            ans = False
            for color in check_color:
                ans = ans or center_value == color
            return ans

        return center_value == check_color

    def _find_moved_squares(self, chessboard: np.ndarray):
        """
        Searches for two squares affected by the opponent move
        One of them contains the figure, another one is empty now.
        Returns coords of empty (start cell) and with figure (end cell)

        :return: row and column of the cell from where piece was moved and where it was moved
        """
        # moved_squares = [[row, column, pixel color]_square1, [row, column, pixel color]_square2]
        moved_squares = [-1, -1]
        for iidx, i in enumerate(self.db.chessboard_cell_pixels):
            for jidx, j in enumerate(self.db.chessboard_cell_pixels):
                if chessboard[i, j] in self.db.chessboard_bbox['color']:
                    if moved_squares[0] == -1:
                        moved_squares[0] = (iidx, jidx, chessboard[i, j])
                    else:
                        moved_squares[1] = (iidx, jidx, chessboard[i, j])
                        break

        if moved_squares[1] == -1:
            return None, None

        # both found moved squares screenshot
        squares = [chessboard[self.db.side * moved_squares[i][0]: self.db.side * (moved_squares[i][0] + 1),
                   self.db.side * moved_squares[i][1]: self.db.side * (moved_squares[i][1] + 1)]
                   for i in range(2)]

        def _is_cell_empty(square, color):
            # check if color in several random points inside cell equals to color from argument,
            # this is the same as check if figure present at the sell (if yes, then returns False)
            # WARNING: if your mouse after movement will be at the square that is involved in a new opponent move,
            # it will almost certainly cause an error
            step = int(0.1 * self.db.side)
            coords_1d = np.arange(step, step * 10, step=step)
            x, y = np.meshgrid(coords_1d, coords_1d)
            return np.all(square[x, y] == color)

        if not _is_cell_empty(squares[0], moved_squares[0][-1]):
            start = moved_squares[1][:-1]
            end = moved_squares[0][:-1]
            return start, end
        else:
            start = moved_squares[0][:-1]
            end = moved_squares[1][:-1]
            return start, end

    def get_move_notation(self, engine: Stockfish) -> str | None:
        """
            Takes a chessboard screenshot and returns valid UCI notation for a move that has been made by opponent

            :param engine: current instance of our bot
            :return: UCI move notation or None if something wrong
        """

        chessboard = self.get_screenshot(self.db.chessboard_bbox)
        bot_color = self.db.color
        start, end = self._find_moved_squares(chessboard)

        # consider 0 moved cells have been found
        if start is None:
            return

        # consider mirror coordinates for Black
        if bot_color == 'White':
            opponent_move = inds_to_letters[start[1]] + str(8 - start[0]) + inds_to_letters[end[1]] + str(8 - end[0])
        else:
            opponent_move = inds_to_letters[7 - start[1]] + str(1 + start[0]) + \
                            inds_to_letters[7 - end[1]] + str(1 + end[0])

        # consider promotion case
        if opponent_move[1] == '7' and opponent_move[3] == '8' and \
                (engine.get_what_is_on_square(opponent_move[:2]) == Stockfish.Piece.WHITE_PAWN and bot_color == 'Black' or
                 engine.get_what_is_on_square(opponent_move[:2]) == Stockfish.Piece.BLACK_PAWN and bot_color == 'White'):
            opponent_move += 'q'

        # consider castling case
        if not engine.is_move_correct(opponent_move):
            if bot_color == 'White':
                row = '8'
            else:
                row = '1'

            # pair start end have random order when castled, so start when castle always at 'e'
            # and then move only by 2 cells in correct direction
            where_castle = (opponent_move[0] if opponent_move[2] == 'e' else opponent_move[2])
            if where_castle in ['h', 'g']:
                where_castle = 'g'
            else:
                where_castle = 'c'

            opponent_move = 'e' + row + where_castle + row

        return opponent_move

    def is_bot_to_move(self):
        return self._check_color_equals(self.db.timer_bbox, self.db.timer_bbox['color'])

    # todo check_game_is_over, takeback && draw offer
    def check_game_is_over(self):
        return False

    def check_takeback(self):
        pass

    def check_draw_offer(self):
        pass


if __name__ == '__main__':
    db = ChessRelevantData()
    controller = Controller()
    image_controller = ImageController()

    with open('../configs/stockfish_config.yaml', 'r') as bot_configs_file:
        bot_configs = yaml.safe_load(bot_configs_file)

    test_engine = Stockfish(path=bot_configs['path_to_stockfish'], parameters=bot_configs['fast_stockfish']['api'])
    # test_engine.set_position()
    # test_engine.make_moves_from_current_position(['e2e4', 'e7e5', 'c2c3', 'b8c6', 'f1c4', 'g8f6', 'd2d3', 'f8c5', 'g1e2'])

    print(image_controller.get_move_notation(test_engine))

    """
    while image_controller.get_move_notation(test_engine) is None:  # we test white now
        continue

    move_i = 0

    while True:
        print(f'{move_i}-th move')
        my_move = image_controller.get_move_notation(test_engine)
        print(f'    my move: {my_move}')
        test_engine.make_moves_from_current_position([my_move])
        while not image_controller.is_bot_to_move():
            continue

        opponent_move = image_controller.get_move_notation(test_engine)
        test_engine.make_moves_from_current_position([opponent_move])
        move_i += 1
        print(f'    opponent move: {opponent_move}')
        print(test_engine.get_board_visual(perspective_white=True), '\n\n\n\n')

        while image_controller.is_bot_to_move():
            continue
            
    """