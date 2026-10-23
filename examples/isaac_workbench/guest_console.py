import code
import sys


class Mirror:
    def __init__(self, terminal, serial):
        self.terminal = terminal
        self.serial = serial

    def write(self, text):
        self.terminal.write(text)
        self.terminal.flush()
        self.serial.write(text)
        self.serial.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.serial.flush()


class Console(code.InteractiveConsole):
    def raw_input(self, prompt=""):
        line = input(prompt)
        serial.write(line + "\n")
        serial.flush()
        return line


with open("/dev/ttyS0", "w", buffering=1) as serial:
    sys.stdout = Mirror(sys.stdout, serial)
    sys.stderr = Mirror(sys.stderr, serial)
    print("\033[2J\033[HCarrot / Linux robot workbench")
    print("Physical keyboard -> Linux input driver -> Python")
    print("Type a Python program. Example: answer=6*7, then answer.")
    print("WORKBENCH_READY")
    Console().interact(banner="", exitmsg="Console finished")
