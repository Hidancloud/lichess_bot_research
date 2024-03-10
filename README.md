 The repository contains an automatic bot playing on lichess.org from your computer. 
 It immitates human thinking time, can be fast and plays on elo which you choose.
 It does work out of the box on Mozilla with Incognito mode on Windows with 1920x1080 resolution (you can open lichess tab on the second monitor).

In other scenarios all the bbox coordinates at *configs/screenshot_boxes.yaml* and *configs/click_action_coordinates.yaml* should be tuned. 
Automatic recognition of relevant fields is not implemented (yet!)

HOWTO run: 
  - Install requirements from *requirements.txt*
  - Download the latest *Stockfish engine*
  - Fill the *"path_to_stockfish"* field (path to .exe) in *configs/stockfish_config.yaml* (by default SF is intended to be located in the project folder)
  - Open *lichess.org*, log in, restore all settings to default (or create new account)
  - Set up the desired mode at the timer slider and close it
  - Fill with corresponding values: *total_time_in_minutes* and *increment_in_seconds* at *configs/timer_parameters.yaml* (bullet 2 + 1 by default)
  - Open Full Screen (*F11*)
  - Run *src/main.py* (you may have to shift lichess to another monitor) to play exactly one rated game.

  Feel free to change any bot related stuff in stockfish_config.yaml (by default custom_stockfish is used in src/main.py)

Demo is available at twitch.tv/hidancloud (you can watch VODs if they are available)
