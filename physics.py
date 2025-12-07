"""
physics.py

Lightweight physics helper utilities for the racing game.
We keep math-y stuff here so Car/Track/Game stay focused on game logic.
"""

from __future__ import annotations

import math
from typing import Tuple

import pygame


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value between [minimum, maximum]."""
    return max(minimum, min(maximum, value))


def vec2(x: float, y: float) -> pygame.math.Vector2:
    """Convenience helper around pygame Vector2."""
    return pygame.math.Vector2(x, y)


def rotate_point(point: pygame.math.Vector2,
                 origin: pygame.math.Vector2,
                 angle_degrees: float) -> pygame.math.Vector2:
    """
    Rotate a point around an origin by some angle in degrees.
    Positive angles rotate counter-clockwise.
    """
    translated = point - origin
    rotated = translated.rotate(-angle_degrees)  # pygame rotates clockwise for positive
    return rotated + origin


def get_forward_vector(angle_degrees: float) -> pygame.math.Vector2:
    """
    Compute forward vector for the car given its heading angle.
    We treat 0 degrees as pointing to the right, and positive angles turning left.
    """
    return pygame.math.Vector2(1, 0).rotate(-angle_degrees)


def get_right_vector(angle_degrees: float) -> pygame.math.Vector2:
    """Right vector is simply forward rotated by -90 degrees."""
    return pygame.math.Vector2(1, 0).rotate(-angle_degrees - 90)


def apply_friction(velocity: pygame.math.Vector2,
                   rolling_friction: float,
                   air_friction: float,
                   dt: float) -> pygame.math.Vector2:
    """
    Apply simple friction model.
    - rolling friction: constant deceleration
    - air friction: proportional to current speed
    """
    speed = velocity.length()
    if speed <= 1e-3:
        return vec2(0, 0)

    # Direction of motion
    direction = velocity.normalize()

    # Magnitude of friction force
    friction_force = rolling_friction + air_friction * speed

    # Delta-v = F * dt (mass is implicitly 1)
    dv = friction_force * dt

    # If dv exceeds current speed, we just stop.
    if dv >= speed:
        return vec2(0, 0)

    return direction * (speed - dv)


def polygon_from_center_rotation(center: pygame.math.Vector2,
                                 length: float,
                                 width: float,
                                 angle_degrees: float) -> list[pygame.math.Vector2]:
    """
    Construct the four corners of a rotated rectangle (the car's footprint) given center,
    length, width, and heading in degrees.
    """
    half_l = length / 2
    half_w = width / 2

    # Unrotated corners relative to center.
    corners = [
        vec2(+half_l, -half_w),  # front-right
        vec2(+half_l, +half_w),  # front-left
        vec2(-half_l, +half_w),  # rear-left
        vec2(-half_l, -half_w),  # rear-right
    ]

    # Build rotated polygon.
    poly = []
    for corner in corners:
        rotated = corner.rotate(-angle_degrees)
        poly.append(center + rotated)

    return poly


def polygons_intersect(poly_a: list[pygame.math.Vector2],
                       poly_b: list[pygame.math.Vector2]) -> bool:
    """
    Use Separating Axis Theorem (SAT) to test if two convex polygons intersect.
    This is overkill for rectangles but showcases more serious collision detection.
    """
    def edges(poly):
        return [poly[(i + 1) % len(poly)] - poly[i] for i in range(len(poly))]

    def perpendicular(edge):
        # A perpendicular vector for edge (x, y) is (y, -x).
        return pygame.math.Vector2(edge.y, -edge.x)

    def project(poly, axis):
        # Project all vertices on axis and get min/max scalar.
        dots = [v.dot(axis) for v in poly]
        return min(dots), max(dots)

    def overlap(min_a, max_a, min_b, max_b):
        return (min_b <= max_a) and (min_a <= max_b)

    for poly in (poly_a, poly_b):
        for edge in edges(poly):
            axis = perpendicular(edge)
            if axis.length_squared() <= 1e-6:
                continue
            axis = axis.normalize()

            min_a, max_a = project(poly_a, axis)
            min_b, max_b = project(poly_b, axis)
            if not overlap(min_a, max_a, min_b, max_b):
                # Found a separating axis -> no intersection.
                return False

    return True


def reflect_velocity_on_collision(velocity: pygame.math.Vector2,
                                  normal: pygame.math.Vector2,
                                  damping: float) -> pygame.math.Vector2:
    """
    Reflect a velocity vector about a collision normal, with damping
    to simulate loss of kinetic energy.
    """
    if normal.length_squared() <= 1e-6:
        return -velocity * damping

    n = normal.normalize()
    # Reflect: v' = v - 2 (v ⋅ n) n
    reflected = velocity - 2 * velocity.dot(n) * n
    return reflected * damping


def distance(p1: pygame.math.Vector2, p2: pygame.math.Vector2) -> float:
    """Simple distance helper."""
    return (p1 - p2).length()


def angle_to_target(forward: pygame.math.Vector2,
                    to_target: pygame.math.Vector2) -> float:
    """
    Signed angle (in degrees) from the forward vector to the vector pointing to target.
    Positive means target is to the "left", negative -> "right".
    """
    if forward.length_squared() <= 1e-6 or to_target.length_squared() <= 1e-6:
        return 0.0
    forward_n = forward.normalize()
    target_n = to_target.normalize()
    angle = forward_n.angle_to(target_n)  # pygame returns positive CCW
    return angle
