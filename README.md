Solution does work out of the box on Mozilla with Incognito mode with 1920x1080 resolution (you can open lichess tab on the second monitor).

In other scenarios all the bbox coordinates at *configs/screenshot_boxes.yaml* and *configs/click_action_coordinates.yaml* should be tuned. Automatic recognition of relevant fields is not ready (yet!)

HOWTO run: 
  - Install requirements from *requirements.txt*
  - Download the latest *Stockfish engine*
  - Fill the *"path_to_stockfish"* field (path to .exe) in *configs/stockfish_config.yaml* (by default SF is intended to be located in the project folder)
  - Open *lichess.org*, restore all settings to default (or create new account)
  - Set up the desired mode at the timer slider and close it
  - Fill with corresponding values: *total_time_in_minutes* and *increment_in_seconds* at *configs/timer_parameters.yaml*
  - Open Full Screen (*F11*)
  - Run *main.py* (you may have to shift lichess to another monitor)
  