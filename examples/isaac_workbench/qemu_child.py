import ctypes
import os
import signal
import sys


def main() -> None:
    assert sys.platform == "linux" and len(sys.argv) > 2
    parent = int(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    # A fresh interpreter avoids unsafe preexec_fn calls inside the multithreaded VM pool.
    assert libc.prctl(1, signal.SIGTERM, 0, 0, 0) == 0, os.strerror(ctypes.get_errno())
    assert os.getppid() == parent, "VM owner died before the parent-death signal was installed"
    os.execvp(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
