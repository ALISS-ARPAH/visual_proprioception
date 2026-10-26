#!/usr/bin/env python3
"""Area-driven ROS2 logger for spot area, triangulated depth, and CTR tip pose.

Every area message produces one CSV row. Distance messages are matched by
callback arrival time (Float64 topics have no image timestamp). A short buffer
allows the corresponding distance and tip pose to arrive before recording missing values.
Tip pose is matched by callback arrival time and expressed in its published frame.
"""
import csv
import math
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseWithCovarianceStamped


class LaserDepthAreaLogger(Node):
    def __init__(self):
        super().__init__('laser_depth_area_logger')
        self.declare_parameter('distance_topic', '/laser/tissue_distance')
        self.declare_parameter('area_topic', '/laser/spot/area_px2')
        self.declare_parameter('tip_pose_topic', '/ves_smoother/right/tip_pose')
        self.declare_parameter('max_pose_dt_sec', 0.20)
        self.declare_parameter('csv_path', '~/laser_calibration_data/depth_area.csv')
        self.declare_parameter('max_pair_dt_sec', 0.20)
        self.declare_parameter('max_queue_age_sec', 2.0)
        self.declare_parameter('log_invalid_pairs', True)  # Kept for CLI compatibility; area rows always logged.
        self.declare_parameter('plot_path', '')
        self.declare_parameter('time_plot_path', '')
        self.declare_parameter('plot_idle_sec', 3.0)
        self.declare_parameter('laser_origin_tip_m', [0.0, 0.0, 0.004])
        self.declare_parameter('laser_direction_tip', [0.0, 0.0, 1.0])
        self.declare_parameter('segment_gap_sec', 0.6)
        self.declare_parameter('segment_min_samples', 8)
        self.declare_parameter('z_plot_path', '')
        self.declare_parameter('segment_csv_path', '')

        self.max_dt = float(self.get_parameter('max_pair_dt_sec').value)
        self.max_age = float(self.get_parameter('max_queue_age_sec').value)
        self.plot_idle_sec = float(self.get_parameter('plot_idle_sec').value)
        self.max_pose_dt = float(self.get_parameter('max_pose_dt_sec').value)
        self.laser_offset = np.asarray(self.get_parameter('laser_origin_tip_m').value, dtype=float)
        self.laser_axis = np.asarray(self.get_parameter('laser_direction_tip').value, dtype=float)
        if self.laser_offset.shape != (3,) or self.laser_axis.shape != (3,):
            raise ValueError('Laser offset and direction must be 3-element vectors')
        axis_norm = np.linalg.norm(self.laser_axis)
        if not np.isfinite(axis_norm) or axis_norm < 1e-12:
            raise ValueError('laser_direction_tip must be nonzero and finite')
        self.laser_axis /= axis_norm
        self.segment_gap = float(self.get_parameter('segment_gap_sec').value)
        self.segment_min_samples = int(self.get_parameter('segment_min_samples').value)
        if self.max_pose_dt <= 0:
            raise ValueError('max_pose_dt_sec must be positive')
        if self.max_dt <= 0 or self.max_age < self.max_dt:
            raise ValueError('Require max_pair_dt_sec > 0 and max_queue_age_sec >= max_pair_dt_sec')

        filename = Path(str(self.get_parameter('csv_path').value)).expanduser()
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.csv_columns = [
            'wall_time_unix_sec', 'ros_time_sec', 'distance_m',
            'distance_mm', 'spot_area_px2', 'pair_arrival_dt_sec',
            'valid_distance', 'valid_area',
            'tip_x_m', 'tip_y_m', 'tip_z_m',
            'tip_qx', 'tip_qy', 'tip_qz', 'tip_qw',
            'pose_arrival_dt_sec', 'valid_pose',
            'laser_origin_x_m', 'laser_origin_y_m', 'laser_origin_z_m',
            'laser_direction_x', 'laser_direction_y', 'laser_direction_z'
        ]
        # Avoid silently appending new-schema rows to an old CSV.
        if filename.exists() and filename.stat().st_size > 0:
            with filename.open('r', newline='') as existing:
                header = next(csv.reader(existing), [])
            if header != self.csv_columns:
                raise ValueError(
                    f'Existing CSV has a different header: {filename}. '
                    'Use a new csv_path (recommended) or rename the old file.')
        self.file = filename.open('a', newline='', buffering=1)
        self.writer = csv.writer(self.file)
        if filename.stat().st_size == 0:
            self.writer.writerow(self.csv_columns)

        plot_setting = str(self.get_parameter('plot_path').value).strip()
        self.plot_path = (Path(plot_setting).expanduser() if plot_setting
                          else filename.with_name(filename.stem + '_plot.png'))
        time_setting = str(self.get_parameter('time_plot_path').value).strip()
        self.time_plot_path = (Path(time_setting).expanduser() if time_setting
                               else filename.with_name(filename.stem + '_time_plot.png'))
        z_setting = str(self.get_parameter('z_plot_path').value).strip()
        self.z_plot_path = (Path(z_setting).expanduser() if z_setting
                            else filename.with_name(filename.stem + '_area_vs_z.png'))
        seg_setting = str(self.get_parameter('segment_csv_path').value).strip()
        self.segment_csv_path = (Path(seg_setting).expanduser() if seg_setting
                                 else filename.with_name(filename.stem + '_segment_correlations.csv'))
        self.z_plot_path.parent.mkdir(parents=True, exist_ok=True)
        self.segment_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        self.time_plot_path.parent.mkdir(parents=True, exist_ok=True)

        self.start_monotonic = time.monotonic()
        self.pending_areas = deque()  # (monotonic arrival, unix time, ROS time, area)
        self.distances = deque()      # (monotonic arrival, distance_m)
        self.poses = deque()          # (monotonic arrival, (x,y,z,qx,qy,qz,qw))
        self.session_time_points = []  # (elapsed time, distance_mm or nan, area)
        self.session_points = []       # valid distance-area pairs only
        self.session_emitter_points = []  # (elapsed, emitter xyz, direction xyz)
        self.session_pose_points = []  # (elapsed time, x, y, z), nan when missing
        self.last_area_time = None
        self.last_plot_count = 0
        self.row_count = 0

        self.create_subscription(Float64, str(self.get_parameter('distance_topic').value),
                                 self.on_distance, 50)
        self.create_subscription(Float64, str(self.get_parameter('area_topic').value),
                                 self.on_area, 50)
        self.create_subscription(
            PoseWithCovarianceStamped, str(self.get_parameter('tip_pose_topic').value),
            self.on_pose, 50)
        self.create_timer(0.05, self.maintenance)
        self.get_logger().info(
            f'Area-driven logging to {filename}; distance pairing tolerance '
            f'{self.max_dt:.3f} s. Every area message will be logged.')

    @staticmethod
    def laser_from_tip(pose, offset, axis):
        """Rotate laser extrinsics from the tip frame into the pose frame."""
        x, y, z, qx, qy, qz, qw = pose
        q = np.array([qx, qy, qz, qw], dtype=float)
        q_norm = np.linalg.norm(q)
        if not np.isfinite(q_norm) or q_norm < 1e-12:
            return (float('nan'),) * 6
        q /= q_norm
        qv = q[:3]
        w = q[3]

        def rotate(v):
            t = 2.0 * np.cross(qv, v)
            return v + w * t + np.cross(qv, t)

        origin = np.asarray([x, y, z], dtype=float) + rotate(offset)
        direction = rotate(axis)
        direction /= np.linalg.norm(direction)
        return tuple(origin) + tuple(direction)

    def on_distance(self, msg):
        self.distances.append((time.monotonic(), float(msg.data)))
        self.flush_ready()

    def on_pose(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        self.poses.append((time.monotonic(), values))

    def on_area(self, msg):
        self.pending_areas.append((
            time.monotonic(), time.time(),
            self.get_clock().now().nanoseconds / 1e9, float(msg.data)
        ))
        self.last_area_time = time.monotonic()
        self.flush_ready()

    def flush_ready(self, force=False):
        now = time.monotonic()
        # Wait for a potential later-arriving distance before finalizing an area.
        while self.pending_areas and (force or now - self.pending_areas[0][0] >= self.max_dt):
            ta, wall_sec, ros_sec, area = self.pending_areas.popleft()
            candidates = [(abs(td - ta), idx) for idx, (td, _) in enumerate(self.distances)
                          if abs(td - ta) <= self.max_dt]
            if candidates:
                delta, idx = min(candidates)
                _, depth = self.distances[idx]
                del self.distances[idx]  # A distance message is used only once.
            else:
                delta, depth = float('nan'), float('nan')

            # Pose can be reused across nearby camera frames (typically lower-rate).
            pose_candidates = [
                (abs(tp - ta), values) for tp, values in self.poses
                if abs(tp - ta) <= self.max_pose_dt
            ]
            if pose_candidates:
                pose_delta, pose_values = min(pose_candidates, key=lambda v: v[0])
                valid_pose = all(math.isfinite(v) for v in pose_values)
            else:
                pose_delta = float('nan')
                pose_values = (float('nan'),) * 7
                valid_pose = False
            if not valid_pose:
                pose_values = (float('nan'),) * 7
            laser_values = (self.laser_from_tip(pose_values, self.laser_offset,
                                               self.laser_axis) if valid_pose
                            else (float('nan'),) * 6)

            valid_depth = math.isfinite(depth) and depth > 0
            valid_area = math.isfinite(area) and area > 0
            depth_mm = depth * 1000 if valid_depth else float('nan')
            depth_m = depth if valid_depth else float('nan')
            self.writer.writerow([
                f'{wall_sec:.6f}', f'{ros_sec:.6f}',
                f'{depth_m:.9g}', f'{depth_mm:.9g}', f'{area:.9g}',
                f'{delta:.6f}', int(valid_depth), int(valid_area),
                *(f'{v:.9g}' for v in pose_values),
                f'{pose_delta:.6f}', int(valid_pose),
                *(f'{v:.9g}' for v in laser_values)
            ])
            self.row_count += 1
            elapsed = ta - self.start_monotonic
            self.session_time_points.append((elapsed, depth_mm, area))
            self.session_pose_points.append((elapsed, *pose_values[:3]))
            self.session_emitter_points.append((elapsed, *laser_values))
            if valid_depth and valid_area:
                self.session_points.append((depth_mm, area))
            if self.row_count % 50 == 0:
                self.get_logger().info(
                    f'{self.row_count} area rows; valid distance-area pairs: '
                    f'{len(self.session_points)}')

        # Discard stale distances that cannot match future area arrivals.
        while self.distances and now - self.distances[0][0] > self.max_age:
            self.distances.popleft()
        while self.poses and now - self.poses[0][0] > max(self.max_age, self.max_pose_dt * 2):
            self.poses.popleft()

    def maintenance(self):
        self.flush_ready()
        if (self.plot_idle_sec > 0 and self.last_area_time is not None
                and self.row_count > self.last_plot_count
                and time.monotonic() - self.last_area_time >= self.plot_idle_sec):
            self.save_plots()

    def save_plots(self):
        if not self.session_time_points:
            return
        if self.session_points:
            depths, areas = zip(*self.session_points)
            fig, ax = plt.subplots(figsize=(8, 5.5))
            ax.scatter(depths, areas, s=15, alpha=0.65)
            ax.set_xlabel('Geometric laser-to-tissue distance (mm)')
            ax.set_ylabel('Laser spot area (pixels²)')
            ax.set_title(f'Spot area vs. triangulated distance (n={len(depths)})')
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            try:
                fig.savefig(self.plot_path, dpi=180)
                self.get_logger().info(f'Saved scatter plot: {self.plot_path}')
            except OSError as exc:
                self.get_logger().error(f'Could not save scatter plot: {exc}')
            finally:
                plt.close(fig)
        else:
            self.get_logger().warn('No valid distance-area pairs; scatter plot not generated.')

        elapsed, depths, areas = zip(*self.session_time_points)
        fig, (ax_depth, ax_area, ax_pose) = plt.subplots(
            3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
        ax_depth.plot(elapsed, depths, '.', markersize=4)
        ax_depth.set_ylabel('Geometric distance (mm)')
        ax_depth.set_title(f'Geometric distance and laser spot area vs. time '
                           f'(n={len(elapsed)} area measurements)')
        ax_depth.grid(True, alpha=0.25)
        ax_area.plot(elapsed, areas, '.', markersize=4)
        ax_area.set_ylabel('Laser spot area (pixels²)')
        ax_area.grid(True, alpha=0.25)
        pose_elapsed, xs, ys, zs = zip(*self.session_pose_points)
        for vals, label in ((xs, 'X'), (ys, 'Y'), (zs, 'Z')):
            ax_pose.plot(pose_elapsed, vals, '.', markersize=3, label=label)
        ax_pose.set_ylabel('CTR tip position (m)')
        ax_pose.set_xlabel('Elapsed logger time (s)')
        ax_pose.legend(loc='best')
        ax_pose.grid(True, alpha=0.25)
        try:
            fig.savefig(self.time_plot_path, dpi=180)
            self.get_logger().info(f'Saved time plot: {self.time_plot_path}')
        except OSError as exc:
            self.get_logger().error(f'Could not save time plot: {exc}')
        finally:
            plt.close(fig)
        self.save_z_and_segments()
        self.last_plot_count = self.row_count

    def save_z_and_segments(self):
        """Plot area vs. tip Z colored by time, and summarize continuous segments.

        Segments are separated by arrival-time gaps, missing pose, or nonpositive
        spot area. Correlations are descriptive, not ground-truth depth validation.
        """
        data = np.asarray([
            (t, area, z, ez, depth)
            for (t, depth, area), (_, _, _, z), (_, _, _, ez, _, _, _) in zip(
                self.session_time_points, self.session_pose_points,
                self.session_emitter_points)
        ], dtype=float)
        if data.size == 0:
            return
        times, areas, zs, emitter_zs, depths = data.T
        good = np.isfinite(times) & np.isfinite(areas) & (areas > 0) & np.isfinite(zs)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, vals, xlabel in (
            (axs[0], zs, 'CTR tip Z (m)'),
            (axs[1], emitter_zs, 'Laser emitter Z (m)')
        ):
            ok = good & np.isfinite(vals)
            if np.any(ok):
                pts = ax.scatter(vals[ok] * 1000, areas[ok], c=times[ok],
                                 cmap='viridis', s=15, alpha=0.8)
                fig.colorbar(pts, ax=ax, label='Elapsed logger time (s)')
            ax.set_xlabel(xlabel.replace('(m)', '(mm)'))
            ax.set_ylabel('Laser spot area (pixels²)')
            ax.grid(True, alpha=0.25)
        fig.suptitle('Spot area vs. CTR Z, colored by time')
        try:
            fig.savefig(self.z_plot_path, dpi=180)
            self.get_logger().info(f'Saved Z scatter plot: {self.z_plot_path}')
        except OSError as exc:
            self.get_logger().error(f'Could not save Z scatter plot: {exc}')
        finally:
            plt.close(fig)

        rows = []
        # Only contiguous valid-area/pose samples, and no time jumps.
        groups = []
        current = []
        for i in range(len(times)):
            if not good[i]:
                if current:
                    groups.append(current)
                    current = []
                continue
            if current and times[i] - times[current[-1]] > self.segment_gap:
                groups.append(current)
                current = []
            current.append(i)
        if current:
            groups.append(current)

        for seg_id, inds in enumerate(groups):
            if len(inds) < self.segment_min_samples:
                continue
            idx = np.asarray(inds, dtype=int)
            a = areas[idx]
            z = zs[idx] * 1000
            e = emitter_zs[idx] * 1000
            d = depths[idx]

            def corr(x, y):
                ok = np.isfinite(x) & np.isfinite(y)
                if np.count_nonzero(ok) < 3:
                    return float('nan'), int(np.count_nonzero(ok))
                xx, yy = x[ok], y[ok]
                if np.std(xx) < 1e-10 or np.std(yy) < 1e-10:
                    return float('nan'), int(np.count_nonzero(ok))
                return float(np.corrcoef(xx, yy)[0, 1]), int(np.count_nonzero(ok))

            r_z, _ = corr(z, a)
            r_e, _ = corr(e, a)
            r_d, n_depth = corr(d, a)
            rows.append([seg_id, f'{times[idx[0]]:.4f}',
                         f'{times[idx[-1]]:.4f}', len(idx), n_depth,
                         f'{r_z:.6g}', f'{r_e:.6g}', f'{r_d:.6g}',
                         f'{np.min(a):.6g}', f'{np.max(a):.6g}',
                         f'{np.min(z):.6g}', f'{np.max(z):.6g}'])
        try:
            with self.segment_csv_path.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['segment_id', 'start_time_sec', 'end_time_sec',
                                 'n_area_pose', 'n_valid_depth',
                                 'pearson_area_tip_z', 'pearson_area_emitter_z',
                                 'pearson_area_geom_depth', 'min_area_px2',
                                 'max_area_px2', 'min_tip_z_mm', 'max_tip_z_mm'])
                writer.writerows(rows)
            self.get_logger().info(
                f'Saved {len(rows)} segment correlations: {self.segment_csv_path}')
        except OSError as exc:
            self.get_logger().error(f'Could not save segment correlations: {exc}')

    def destroy_node(self):
        try:
            self.flush_ready(force=True)
            if self.row_count > self.last_plot_count:
                self.save_plots()
            self.file.flush()
            self.file.close()
        finally:
            return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaserDepthAreaLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
