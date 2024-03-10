from utils_classes import *
# import keyboard


# todo: check if opponent afk => wait and continue or kick

# todo: implement check_game_is_over_CHANGE

# todo: DB COLOR BLOCKS GAME START

# todo: write games in PGNs

# todo: write think time to CSVs

# todo: add lose strategy and probability for it

# todo: add tracking for win / lose probability to console and file

# todo: add forced openings

# todo: control time logic: checks, material capture faster and etc.

class ChessBot:
    def __init__(self, stockfish_option='fast_stockfish', debut_option='bullet_normal'):
        self.db = ChessRelevantData(skip_color=True)
        self.controller = Controller()
        self.image_controller = ImageController()

        ###############################
        # bot settings initialization #
        ###############################

        with open('../configs/stockfish_config.yaml', 'r') as bot_configs_file:
            self.bot_configs = yaml.safe_load(bot_configs_file)

        self.openings = self.bot_configs['openings'][debut_option]
        self.engine = Stockfish(path=self.bot_configs['path_to_stockfish'],
                                parameters=self.bot_configs['bot_power_settings'][stockfish_option]['api'])

        if 'elo_rating' in self.bot_configs['bot_power_settings'][stockfish_option].keys():
            self.engine.set_elo_rating(self.bot_configs['bot_power_settings'][stockfish_option]['elo_rating'])

        if 'skill_level' in self.bot_configs['bot_power_settings'][stockfish_option].keys():
            self.engine.set_skill_level(skill_level=self.bot_configs['bot_power_settings']
            [stockfish_option]['skill_level'])

        self.controller.hard_restart()
        self.db.init_color()
        # keyboard.on_press_key("r", lambda _: self.start_new_game())
        while True:
            self.infinite_play()

    def wait_for_opponent_move(self):
        # wait until timer becomes green
        # doesn't work for the first move for black
        while not self.image_controller.is_bot_to_move():
            continue

        opponent_move = self.image_controller.get_move_notation(self.engine)
        self.engine.make_moves_from_current_position([opponent_move])
        return opponent_move

    def try_make_premove(self, time_to_think):
        # todo: (at least for recaptures (look at the best opponent move, if it is capture -> premove recapture))
        for_premove = 0.1 * time_to_think

        return None

    def infinite_play(self):
        opening = self.openings[np.random.randint(low=0, high=len(self.openings))]
        if self.db.color == 'Black':
            for i in range(len(opening)):
                opening[i] = ChessRelevantData.mirror_move(opening[i])

        opening_correct = True

        print(f'our opening: {opening}')
        move_idx = 0

        if self.db.color == 'Black':
            opponent_move = None
            while opponent_move is None:
                opponent_move = self.image_controller.get_move_notation(self.engine)
            self.engine.make_moves_from_current_position([opponent_move])
            opponent_move_duration = 0

        while not self.image_controller.check_game_is_over():
            bot_move_start = time()
            move_idx += 1

            time_to_think = self.db.time_manager.get_time_to_think(move_idx, self.engine)
            print(f'planned time to think: {cool_round(time_to_think / 1000)}')

            if move_idx <= len(opening) and self.engine.is_move_correct(opening[move_idx - 1]) and opening_correct:
                best_move = opening[move_idx - 1]
                sleep(time_to_think * 0.001 * 0.9)
            elif move_idx <= len(opening) and not self.engine.is_move_correct(
                    opening[move_idx - 1]) and opening_correct:
                opening_correct = False
                # todo: stockfish.get_best_move(wtime=1000, btime=1000) white time black time maybe even better!
                best_move = self.engine.get_best_move_time(time_to_think)
            else:
                best_move = self.engine.get_best_move_time(time_to_think)
            self.engine.make_moves_from_current_position([best_move])
            self.controller.make_move(best_move)
            bot_move_end = time()
            bot_move_duration = bot_move_end - bot_move_start
            self.db.time_manager.reduce_timer(bot_move_duration)

            if self.db.color == 'Black':
                print(f'{move_idx}. {opponent_move} {best_move}; \n'
                      f'our time to think: {cool_round(bot_move_duration)} seconds || '
                      f'opponents: {cool_round(opponent_move_duration)} seconds\n')

            opponent_move_start = time()
            opponent_move = self.wait_for_opponent_move()
            opponent_move_end = time()
            opponent_move_duration = opponent_move_end - opponent_move_start

            if self.db.color == 'White':
                print(f'{move_idx}. {best_move} {opponent_move}; \n'
                      f'opponents: {cool_round(opponent_move_duration)} seconds || '
                      f'our time to think: {cool_round(bot_move_duration)} seconds\n')

        self.start_new_game()

    def start_new_game(self):
        print('start the game\n\n\n')
        self.engine.set_position()
        self.controller.hard_restart()
        self.db.restart()


if __name__ == '__main__':
    nonstop_chess = ChessBot('custom_stockfish', debut_option='bullet_normal')

    # db = ChessRelevantData(skip_color=True)
    # print(db.time_manager.time_strategy)
