#!/usr/bin/env python3
"""Tkinter ROS2 GUI: live laser analysis image, area plot, and touch threshold."""
import math
import threading
import time
from collections import deque
import tkinter as tk
from tkinter import ttk

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Bool
from PIL import Image as PILImage, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

AREA_TOPIC = '/laser/spot/area_px2'
IMAGE_TOPIC = '/laser/spot/analysis_overlay'
TOUCH_TOPIC = '/tissue_touched'
TOUCH_NODE = '/laser_tissue_touch_node'
PARAMETER = 'area_threshold_px2'


class MonitorNode(Node):
    def __init__(self):
        super().__init__('laser_spot_monitor_gui')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_area = float('nan')
        self.latest_touch = None
        self.last_area_rx = None
        self.times = deque(maxlen=3000)
        self.areas = deque(maxlen=3000)
        self.client = AsyncParameterClient(self, TOUCH_NODE)
        self.create_subscription(Image, IMAGE_TOPIC, self.image_cb, 10)
        self.create_subscription(Float64, AREA_TOPIC, self.area_cb, 10)
        self.create_subscription(Bool, TOUCH_TOPIC, self.touch_cb, 10)

    def image_cb(self, msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            with self.lock:
                self.latest_image = rgb
        except Exception as exc:
            self.get_logger().warn(f'Image conversion failed: {exc}')

    def area_cb(self, msg):
        area = float(msg.data)
        now = time.monotonic()
        with self.lock:
            self.latest_area = area
            self.last_area_rx = now
            self.times.append(now)
            self.areas.append(area)

    def touch_cb(self, msg):
        with self.lock:
            self.latest_touch = bool(msg.data)


class MonitorGUI:
    def __init__(self, root, node):
        self.root, self.node = root, node
        root.title('Laser Spot Monitor')
        root.geometry('1250x760')
        self.running = True
        self.threshold = 1000.0
        self.last_frame_id = None
        self.pending_set = None
        self.pending_get = None
        self.last_param_check = 0.0
        self.last_plot = 0.0
        self.last_image_update = 0.0

        outer = ttk.Frame(root, padding=12)
        outer.pack(fill='both', expand=True)
        controls = ttk.Frame(outer)
        controls.pack(fill='x', pady=(0, 10))
        ttk.Label(controls, text='Contact area threshold (px²):').pack(side='left')
        self.threshold_var = tk.StringVar(value='1000')
        entry = ttk.Entry(controls, textvariable=self.threshold_var, width=12)
        entry.pack(side='left', padx=8)
        entry.bind('<Return>', lambda _: self.apply_threshold())
        self.apply_button = ttk.Button(controls, text='Apply threshold',
                                       command=self.apply_threshold)
        self.apply_button.pack(side='left')
        self.status_var = tk.StringVar(value='Connecting to tissue-touch node...')
        ttk.Label(controls, textvariable=self.status_var).pack(side='left', padx=14)

        stats = ttk.Frame(outer)
        stats.pack(fill='x', pady=(0, 8))
        self.area_var = tk.StringVar(value='Spot area: N/A')
        self.touch_var = tk.StringVar(value='Tissue touched: N/A')
        ttk.Label(stats, textvariable=self.area_var, font=('Arial', 13, 'bold')).pack(side='left', padx=(0, 30))
        ttk.Label(stats, textvariable=self.touch_var, font=('Arial', 13, 'bold')).pack(side='left')

        panels = ttk.Frame(outer)
        panels.pack(fill='both', expand=True)
        left = ttk.LabelFrame(panels, text=IMAGE_TOPIC, padding=6)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self.image_label = ttk.Label(left, text='Waiting for image...', anchor='center')
        self.image_label.pack(fill='both', expand=True)

        right = ttk.LabelFrame(panels, text='Estimated laser spot area vs. time', padding=6)
        right.pack(side='left', fill='both', expand=True, padx=(6, 0))
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (seconds ago)')
        self.ax.set_ylabel('Area (px²)')
        self.ax.grid(alpha=0.25)
        self.line, = self.ax.plot([], [], lw=1.7, label='Measured area')
        self.threshold_line = self.ax.axhline(self.threshold, linestyle='--',
                                              color='tab:red', label='Contact threshold')
        self.ax.legend(loc='upper right')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        footer = ttk.Frame(outer)
        footer.pack(fill='x', pady=(8, 0))
        ttk.Label(footer, text='Plot window (seconds):').pack(side='left')
        self.window_var = tk.StringVar(value='30')
        ttk.Combobox(footer, textvariable=self.window_var, width=7, state='readonly',
                     values=['10', '30', '60', '120', '300']).pack(side='left', padx=8)
        ttk.Label(footer, text='Contact heuristic: 0 < area ≤ threshold; not verified physical contact.').pack(side='right')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(150, self.update_ui)

    def apply_threshold(self):
        try:
            val = float(self.threshold_var.get())
            if not math.isfinite(val) or val <= 0:
                raise ValueError
        except ValueError:
            self.status_var.set('Enter a finite positive threshold.')
            return
        if self.pending_set is not None:
            self.status_var.set('Waiting for previous update...')
            return
        if not self.node.client.services_are_ready():
            self.status_var.set(f'Parameter service unavailable: {TOUCH_NODE}')
            return
        self.pending_set = (val, self.node.client.set_parameters([
            Parameter(PARAMETER, value=val)
        ]))
        self.status_var.set('Applying threshold...')
        self.apply_button.configure(state='disabled')

    def check_parameters(self):
        if self.pending_set is not None:
            val, future = self.pending_set
            if future.done():
                try:
                    result = future.result()
                    if result and all(r.successful for r in result):
                        self.threshold = val
                        self.status_var.set(f'Applied {val:g} px² to {TOUCH_NODE}')
                    else:
                        reasons = ', '.join(r.reason for r in result if not r.successful)
                        self.status_var.set('Update rejected: ' + reasons)
                except Exception as exc:
                    self.status_var.set(f'Update failed: {exc}')
                self.pending_set = None
                self.apply_button.configure(state='normal')
        if self.pending_get is not None and self.pending_get.done():
            try:
                result = self.pending_get.result()
                if result and result.values:
                    v = result.values[0]
                    val = float(v.double_value if v.type == 3 else v.integer_value)
                    if math.isfinite(val) and val > 0:
                        self.threshold = val
                        if self.pending_set is None:
                            self.threshold_var.set(f'{val:g}')
                            self.status_var.set(f'Connected to {TOUCH_NODE}')
            except Exception as exc:
                self.status_var.set(f'Parameter read failed: {exc}')
            self.pending_get = None
        now = time.monotonic()
        if (self.pending_get is None and self.pending_set is None
                and now - self.last_param_check > 5.0
                and self.node.client.services_are_ready()):
            self.last_param_check = now
            self.pending_get = self.node.client.get_parameters([PARAMETER])

    def update_ui(self):
        if not self.running:
            return
        self.check_parameters()
        with self.node.lock:
            image = self.node.latest_image
            area = self.node.latest_area
            touch = self.node.latest_touch
            last_rx = self.node.last_area_rx
            times = list(self.node.times)
            areas = list(self.node.areas)
        now = time.monotonic()
        stale = last_rx is None or now - last_rx > 1.0
        self.area_var.set('Spot area: N/A' if stale or not math.isfinite(area)
                          else f'Spot area: {area:.1f} px²')
        self.touch_var.set('Tissue touched: N/A' if stale or touch is None
                           else f'Tissue touched: {"TRUE" if touch else "FALSE"}')

        if image is not None and now - self.last_image_update > 0.08:
            self.last_image_update = now
            width = max(200, self.image_label.winfo_width() - 8)
            height = max(200, self.image_label.winfo_height() - 8)
            pil = PILImage.fromarray(image)
            pil.thumbnail((width, height), PILImage.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil)
            self.image_label.configure(image=photo, text='')
            self.image_label.image = photo

        if now - self.last_plot > 0.3:
            self.last_plot = now
            window = float(self.window_var.get())
            pairs = [(t - now, a) for t, a in zip(times, areas)
                     if t >= now - window and math.isfinite(a)]
            if pairs:
                x, y = zip(*pairs)
                self.line.set_data(x, y)
                top = max(max(y), self.threshold) * 1.15
            else:
                self.line.set_data([], [])
                top = self.threshold * 1.2
            self.threshold_line.set_ydata([self.threshold, self.threshold])
            self.ax.set_xlim(-window, 0)
            self.ax.set_ylim(0, max(1, top))
            self.canvas.draw_idle()
        self.root.after(100, self.update_ui)

    def close(self):
        self.running = False
        self.root.destroy()


def main(args=None):
    rclpy.init(args=args)
    node = MonitorNode()
    root = tk.Tk()
    gui = MonitorGUI(root, node)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        root.mainloop()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()


if __name__ == '__main__':
    main()
