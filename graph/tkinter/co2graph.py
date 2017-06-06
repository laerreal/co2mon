#!/usr/bin/python

# See http://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
from __future__ import print_function

import sys
from posix import ST_NOEXEC

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

from matplotlib import use, rc
use("TkAgg")
# Turn on TeX in graph labels
# See http://stackoverflow.com/questions/18739703/text-usetex-true-setting-renders-the-axis-labels-as-well
try:
    rc("text", usetex=True)
except:
    # TeX is not available.
    # TODO: Branch was not tested yet.
    eprint("Cannot use TeX for labels. Simple text will be used.")
    TEMP_STR = "Temperature, C"
    CO2_STR = "CO2, ppm"
else:
    TEMP_STR = r"Temperature, $C^{\circ}$"
    CO2_STR = r"${CO}_2$, $({ppm)}$"

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from six.moves.tkinter import Tk, BOTH
from subprocess import Popen, PIPE
from threading  import Thread, Event

from time import time, localtime, strftime
from numpy import concatenate, array, NaN, isnan

from collections import OrderedDict

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x

ON_POSIX = 'posix' in sys.builtin_module_names

CO2MOND = "co2mond"
LOG_NAME = "co2graph.log"
LOG_INDEX_INTERVAL = 60 * 60 # in seconds
REFRESH_PERIOD = 3 # in seconds
INTERPOLATION_PERIOD = float(REFRESH_PERIOD) * 2.0 # in seconds
POLL_PERIOD = REFRESH_PERIOD * (1000 + 1) # in milliseconds
GRAPH_Y_DELTA_FACTOR = .1
EPS = .000000001

if __name__ == "__main__":
    # build log index
    # TODO: use binary tree
    log_index = OrderedDict()
    try:
        l = open(LOG_NAME, "rb")
    except:
        pass
    else:
        t0 = 0.0
        off = l.tell()
        # preserving tell() consistency under Python 2
        # See http://stackoverflow.com/questions/14145082/file-tell-inconsistency
        for rec in iter(l.readline, ''):
            # kind, time stamp, value
            if rec == b"":
                break
            k, t, v = rec.split()
            t, v = int(float(t)), float(v)
            if t - t0 >= LOG_INDEX_INTERVAL:
                log_index[t] = off
                t0 = t
            off = l.tell()
        l.close()

    def open_log(t0 = None):
        if t0 is None:
            return open(LOG_NAME, "ab")

        if not log_index:
            return open(LOG_NAME, "wb+")

        t0 = int(t0)
        for t, off in log_index.items():
            if t > t0:
                break
            prev_off = off

        l = open(LOG_NAME, "rb")
        l.seek(prev_off)
        for rec in iter(l.readline, ''):
            if rec == b"":
                break
            k, t, v = rec.split()
            t = int(float(t))
            if t > t0:
                l.seek(prev_off)
                break
            prev_off = l.tell()

        return l

    # GUI
    root = Tk()
    root.title("CO2 monitor graph")
    root.geometry("800x400")

    # Graph
    # See https://pythonprogramming.net/how-to-embed-matplotlib-graph-tkinter-gui/
    f = Figure(figsize=(5,5), dpi=100)

    # CO2 graph
    co2sp = f.add_subplot(2,1,1)
    co2sp.set_autoscaley_on(True)
    co2sp.set_ylabel(CO2_STR)

    co2p, = co2sp.plot([], [])

    # Temperature graph
    tsp = f.add_subplot(2,1,2, sharex=co2sp)
    tsp.set_autoscaley_on(True)
    tsp.set_ylabel(TEMP_STR)

    tp, = tsp.plot([], [])

    fc = FigureCanvasTkAgg(f, root)
    canvas = fc.get_tk_widget()
    canvas.pack(fill=BOTH, expand=1)

    def update_plot(p, t, v=None):
        global period

        ydat = p.get_ydata()
        if v is None:
            v = ydat[-1]

        xdat = p.get_xdata()

        extv = [v]
        extt = [t]

        t0 = xdat[-1]
        if t - t0 > INTERPOLATION_PERIOD:
            extv.insert(0, NaN)
            extt.insert(0, t - EPS)
            extv.insert(0, NaN)
            extt.insert(0, t0 + EPS)

        while len(xdat) > 1:
            diff = extt[-1] - xdat[0]
            if diff <= period:
                break
            xdat = xdat[1:]
            ydat = ydat[1:]

        ydat = concatenate((ydat, extv))
        xdat = concatenate((xdat, extt))

        p.set_ydata(ydat)
        p.set_xdata(xdat)

    # kind, time stamp (float), value 
    def commit(k, t, v):
        if k == b"t":
            update_plot(tp, t, v)
            update_plot(co2p, t)
        elif k == b"c":
            update_plot(tp, t)
            update_plot(co2p, t, v)

    # See: http://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels
    def get_ax_size(ax, fig):
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        width *= fig.dpi
        height *= fig.dpi
        return width, height

    scale = 1
    period = 0
    def update_period():
        global period
        global scale

        width = get_ax_size(co2sp, f)[0]
        period = width * float(scale)

        t0 = time() - period - float(INTERPOLATION_PERIOD * 2)

        tS = t0 - time()

        co2p.set_ydata([NaN])
        co2p.set_xdata([tS])
        tp.set_ydata([NaN])
        tp.set_xdata([tS])

        l = open_log(t0)

        for rec in l:
            k, t, v = rec.split()
            t, v = float(t), float(v)
    
            commit(k, t, v)

        fc.draw()

    def fix_NaN(func, arr):
        if isnan(arr[0]):
            for i in arr:
                if not isnan(i):
                    arr[0] = i
                    break
        return func(arr)

    def update_limits():
        for p, ax in ((co2p, co2sp), (tp, tsp)):
            ydat = p.get_ydata()
            try:
                ma, mi = fix_NaN(max, ydat), fix_NaN(min, ydat)
            except ValueError:
                ma, mi = 1.0, 0.0
            else:
                if ma == mi:
                    ma = mi + 1.0
            # add extra spaces above and below
            A = abs(ma - mi)
            delta = A * GRAPH_Y_DELTA_FACTOR
            ax.set_ylim([mi - delta, ma + delta])

        # x-axis is shared
        xdat = co2p.get_xdata()
        t = xdat[-1]
        mins = t - period
        co2sp.set_xlim([mins, t])

        # format time
        xticks = co2sp.get_xticks()
        xti = reversed(xticks)

        new_ticks = []
        prev = next(xti)

        full_date_i = len(xticks) // 2 - 1

        for i, t in enumerate(xti):
            tmpt = int(t)
            tmpp = int(prev)
            cur_fmt = ""
            for div, fmt in [
                (60, "%Ss"),
                (60, "%Mm"),
                (24, "%Hh")
            ]:
                if (tmpt % div) != (tmpp % div):
                    cur_fmt = fmt + cur_fmt
                tmpt //= div
                tmpp //= div

            if i == full_date_i:
                new_ticks.append(strftime(
                    "%H:%M:%S\n%a %B'%d %Y",
                    localtime(t)
                ))

            else:
                if i == 0:
                    new_ticks.append(strftime(cur_fmt, localtime(prev)))

                new_ticks.append(strftime(cur_fmt, localtime(t)))

            prev = t

        new_ticks.reverse()
        co2sp.set_xticklabels(new_ticks, visible = False)
        tsp.set_xticklabels(new_ticks)

    def validate():
        f.tight_layout()
        update_period()
        update_limits()

    fc.show()
    validate()
    fc.draw()

    def on_resize(event):
        validate()

    canvas.bind("<Configure>", on_resize, "+")

    def on_key(event):
        global scale
        if event.char == "-":
            if scale > 1:
                scale //=2
                validate()
                fc.draw()
        elif event.char == "+":
            scale  *= 2
            validate()
            fc.draw()

    root.bind("<Key>", on_key)

    # Process
    # See http://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python
    co2 = Popen([CO2MOND], stdout=PIPE, bufsize=1, close_fds=ON_POSIX)
    q = Queue()
    end = Event()

    def enqueue_output(out=co2.stdout):
        for line in iter(out.readline, b''):
            q.put(line)

            if end.is_set():
                break
        out.close()

    t = Thread(target=enqueue_output)
    t.daemon = True # thread dies with the program
    t.start()

    def commit_raw(line):
        kind, val = line.split(b"\t")
        val = float(val)
        t = time()
        if kind == b"Tamb":
            kind = b"t"
        elif kind == b"CntR":
            kind = b"c"

        commit(kind, t, val)

        update_limits()
        fc.draw()

        log = open_log()
        off = log.tell()
        log.write(kind
            + b" " + str(t).encode("utf-8")
            + b" " + str(val).encode("utf-8")
            + b"\n"
        )
        log.close()

        # Update log index
        t = int(t)
        if not log_index:
            log_index[t] = off
        else:
            t0 = next(reversed(log_index))
            if t0 - t >= LOG_INDEX_INTERVAL:
                log_index[t] = off

    def poll_co2():
        root.after(POLL_PERIOD, poll_co2)

        while True:
            # read line without blocking
            try:
                line = q.get_nowait() # or q.get(timeout=.1)
            except Empty:
                break
            else: # got line
                commit_raw(line)

    root.after(0, poll_co2)
    root.mainloop()

    end.set()
    try:
        # Do not wait too long...
        t.join(float(REFRESH_PERIOD) * 4.0)
    except:
        pass
    co2.kill()
