#!/usr/bin/env python3
"""
Tiny auto-refreshing image viewer -- reloads and redisplays a file from
disk on an interval, using Tkinter (not cv2.imshow: this environment's
opencv is a headless conda-forge build with no GUI backend compiled in;
tkinter has no such dependency).

    python image_viewer.py /path/to/frame.jpg [--interval-ms 150]
"""

import argparse
import os
import tkinter as tk

from PIL import Image, ImageTk


class Viewer:
    def __init__(self, root: tk.Tk, path: str, interval_ms: int) -> None:
        self.root = root
        self.path = path
        self.interval_ms = interval_ms
        self.label = tk.Label(root)
        self.label.pack()
        self._photo = None  # keep a reference -- Tkinter drops it otherwise
        self._last_mtime = None
        self.root.after(0, self._refresh)

    def _refresh(self) -> None:
        try:
            mtime = os.path.getmtime(self.path)
            if mtime != self._last_mtime:
                img = Image.open(self.path)
                self._photo = ImageTk.PhotoImage(img)
                self.label.configure(image=self._photo)
                self._last_mtime = mtime
        except (FileNotFoundError, OSError):
            pass  # frame mid-write or not created yet -- just retry
        self.root.after(self.interval_ms, self._refresh)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--interval-ms", type=int, default=150)
    args = p.parse_args()

    root = tk.Tk()
    root.title(f"live: {args.path}")
    Viewer(root, args.path, args.interval_ms)
    root.mainloop()


if __name__ == "__main__":
    main()
