from configuration_pi05_control import PI05ControlConfig
from state_jitter import StateJitterStep
from lerobot.scripts.lerobot_train import main

assert PI05ControlConfig.get_choice_name(PI05ControlConfig) == "pi05_control"
assert StateJitterStep is not None

if __name__ == "__main__":
    main()
