"""
track.py

Defines the Track abstraction (waypoints, drawing, progress computation).
The track is defined primarily by a list of waypoints forming a closed circuit.
"""

from __future__ import annotations

from typing import List, Tuple

import pygame

from physics import vec2, distance
from settings import TRACK_WAYPOINTS, TRACK_WIDTH, START_LINE_RADIUS, GREEN, GRAY


class Track:
    """
    Represents a fixed circuit track.

    Key responsibilities:
    - hold waypoint list (center line of track)
    - render track (thick polyline)
    - compute "progress" for a car (distance along lap)
    - detect when car crosses the start/finish line (lap counting)
    """

    def __init__(self, waypoints: List[Tuple[int, int]] | None = None) -> None:
        # Use provided waypoints or default global.
        if waypoints is None:
            waypoints = TRACK_WAYPOINTS

        # Convert waypoints into Vector2 objects.
        self.waypoints = [vec2(x, y) for x, y in waypoints]
        if len(self.waypoints) < 2:
            raise ValueError("Track needs at least 2 waypoints.")

        # Pre-compute segment lengths and cumulative distances for efficient progress calc.
        self.segment_lengths: List[float] = []
        self.cumulative_lengths: List[float] = [0.0]

        total = 0.0
        for i in range(len(self.waypoints)):
            a = self.waypoints[i]
            b = self.waypoints[(i + 1) % len(self.waypoints)]
            seg_len = (b - a).length()
            self.segment_lengths.append(seg_len)
            total += seg_len
            self.cumulative_lengths.append(total)

        self.total_length = total

        # Define start/finish as around waypoint 0
        self.start_pos = self.waypoints[0]
        self.start_radius = START_LINE_RADIUS

    # -------------------------------------------------------------------------
    # Progress & Lap helpers
    # -------------------------------------------------------------------------

    def project_position_onto_track(self, position: pygame.math.Vector2) -> float:
        """
        Approximate the distance along the track for a given world position.

        We search all segments and find the closest projection on segment,
        then compute 'distance along lap' using pre-computed cumulative lengths.
        """
        best_dist = float("inf")
        best_progress = 0.0

        for i in range(len(self.waypoints)):
            a = self.waypoints[i]
            b = self.waypoints[(i + 1) % len(self.waypoints)]

            segment = b - a
            seg_len_sq = segment.length_squared()
            if seg_len_sq <= 1e-6:
                continue

            t = (position - a).dot(segment) / seg_len_sq
            # Clamp to [0,1]
            t_clamped = max(0.0, min(1.0, t))
            projection = a + segment * t_clamped

            d = (projection - position).length()
            if d < best_dist:
                best_dist = d
                segment_progress = self.cumulative_lengths[i] + segment.length() * t_clamped
                best_progress = segment_progress

        # Normalize such that progress in [0, total_length)
        best_progress %= self.total_length
        return best_progress

    def check_lap_crossing(self,
                           old_position: pygame.math.Vector2,
                           new_position: pygame.math.Vector2) -> bool:
        """
        Returns True if a car moved across the start/finish region between old and new positions.
        We define a circular region around waypoint 0 as the "start/finish".
        """
        old_dist = distance(old_position, self.start_pos)
        new_dist = distance(new_position, self.start_pos)

        # A naive rule: if we moved from outside the start circle to inside, we count as crossing.
        return old_dist > self.start_radius and new_dist <= self.start_radius

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draw the track onto the surface.
        We render:
        - a green background (grass)
        - a thick gray polyline following the waypoints (asphalt)
        - a highlighted start/finish circle
        """
        # Fill background with grass color.
        surface.fill(GREEN)

        # Draw track center line as thick polyline.
        if len(self.waypoints) >= 2:
            pygame.draw.lines(
                surface,
                GRAY,
                closed=True,
                points=[(p.x, p.y) for p in self.waypoints],
                width=TRACK_WIDTH,
            )

        # Start/finish indicator.
        pygame.draw.circle(
            surface,
            (255, 255, 255),
            (int(self.start_pos.x), int(self.start_pos.y)),
            self.start_radius,
            2,
        )
