# #!/usr/bin/env python
# import signal
# import sys


# def signal_handler(sig, frame):
#     print('You pressed Ctrl+C!')
#     sys.exit(0)


# signal.signal(signal.SIGINT, signal_handler)
# print('Press Ctrl+C')
# signal.pause()


# import time
# import sys

# x = 1
# while True:
#     try:
#         print(x)
#         time.sleep(.3)
#         x += 1
#     except KeyboardInterrupt:
#         print("Bye")
#         sys.exit()


import signal
import time


class GracefulInterruptHandler(object):

    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):

        self.interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            self.interrupted = True
        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False
        # ejecuta la funcion con el actual self.sig
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True


# while 1:
#     with GracefulInterruptHandler() as h:
#         # processing
#         print("doing1 ...")
#         time.sleep(1)
#         if h.interrupted:
#             print("interrupted!")
#             print("continue doing ...")
#             time.sleep(2)
#             break
#         # commit

# with GracefulInterruptHandler() as h1:
#     while True:
#         print("(1)...")
#         time.sleep(1)
#         with GracefulInterruptHandler() as h2:
#             while True:
#                 print("\t(2)...")
#                 time.sleep(1)
#                 if h2.interrupted:
#                     print("\t(2) interrupted!")
#                     time.sleep(2)
#                     break
#         if h1.interrupted:
#             print("(1) interrupted!")
#             time.sleep(2)
#             break


with GracefulInterruptHandler() as h1:
    while True:
        print("\t(1)...")
        time.sleep(10)
        if h1.interrupted:
            print("(1) interrupted!")
            time.sleep(2)
            break
