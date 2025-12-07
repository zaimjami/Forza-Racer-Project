"""
car.py

Defines the Car entity and its control abstraction.
This file focuses on *how a car behaves*, independent of game loop & AI.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from physics import (
    vec2,
    get_forward_vector,
    get_right_vector,
    apply_friction,
    polygon_from_center_rotation,
)


@dataclass
class CarControl:
    """
    High-level control signals for a car.

    Values are normalized:
        throttle in [0, 1]
        brake    in [0, 1]
        steer    in [-1, 1] (negative = left, positive = right)
    """
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0


class Car:
    """
    A simple rigid-body car approximation for a top-down racer.

    It models:
    - linear velocity & acceleration
    - rotation / steering
    - friction
    - collisions via a rotated rectangle footprint
    """

    def __init__(
        self,
        name: str,
        color: tuple[int, int, int],
        position: tuple[float, float],
        heading_degrees: float,
        length: float,
        width: float,
        max_engine_force: float,
        max_brake_force: float,
        max_steer_angle: float,
        max_speed: float,
        rolling_friction: float,
        air_friction: float,
    ) -> None:
        self.name = name
        self.color = color

        # Kinematic state
        self.position = vec2(*position)
        self.velocity = vec2(0, 0)
        self.heading = heading_degrees  # in degrees

        # Car shape / physics
        self.length = length
        self.width = width
        self.max_engine_force = max_engine_force
        self.max_brake_force = max_brake_force
        self.max_steer_angle = max_steer_angle
        self.max_speed = max_speed
        self.rolling_friction = rolling_friction
        self.air_friction = air_friction

        # Lap / progress tracking
        self.current_lap = 0
        self.distance_along_track = 0.0
        self.best_lap_time = None
        self.current_lap_time = 0.0

        # Utility / flags
        self.finished = False

    # -------------------------------------------------------------------------
    # Core update
    # -------------------------------------------------------------------------

    def update(self, control: CarControl, dt: float) -> None:
        """
        Integrate car dynamics for one frame.

        The control object describes high-level intent (throttle, brake, steer),
        and this function turns it into forces & kinematics.
        """
        if self.finished:
            # If the car finished the race, we can just smoothly apply friction.
            self.velocity = apply_friction(
                self.velocity, self.rolling_friction, self.air_friction, dt
            )
            self.position += self.velocity * dt
            return

        # 1. Compute forward vector for this heading.
        forward = get_forward_vector(self.heading)

        # 2. Engine force along forward direction.
        throttle_force = control.throttle * self.max_engine_force

        # 3. Brake is applied opposite to current velocity, not necessarily forward.
        #    This is a bit more realistic than just anti-forward.
        if self.velocity.length_squared() > 1e-6:
            brake_dir = -self.velocity.normalize()
        else:
            brake_dir = vec2(0, 0)
        brake_force = control.brake * self.max_brake_force

        # 4. Build resulting acceleration.
        acceleration = forward * throttle_force + brake_dir * brake_force

        # Mass is assumed to be 1 for simplicity, so F = ma -> a = F.
        # Update velocity.
        self.velocity += acceleration * dt

        # 5. Clamp max speed.
        speed = self.velocity.length()
        if speed > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)

        # 6. Apply friction (rolling + air).
        self.velocity = apply_friction(
            self.velocity, self.rolling_friction, self.air_friction, dt
        )

        # 7. Steering: more steering effect at higher speeds, less when slow.
        steer_angle_deg = control.steer * self.max_steer_angle
        # Steering strength scales with speed / max_speed.
        steer_strength = (self.velocity.length() / max(self.max_speed, 1e-3)) * 1.5
        self.heading += steer_angle_deg * steer_strength * dt

        # 8. Integrate position.
        self.position += self.velocity * dt

        # 9. Lap time update (actual lap detection handled by Track/Game).
        self.current_lap_time += dt

    # -------------------------------------------------------------------------
    # Geometry & collision helpers
    # -------------------------------------------------------------------------

    def get_polygon(self) -> list[pygame.math.Vector2]:
        """
        Returns the world-space polygon (4 vertices) for car's footprint.
        This is used for drawing and for polygon-level collision detection.
        """
        return polygon_from_center_rotation(self.position, self.length, self.width, self.heading)

    def get_bounding_rect(self) -> pygame.Rect:
        """
        Axis-aligned bounding box around the rotated car.
        This is a cheap approximation for broad-phase collision.
        """
        poly = self.get_polygon()
        xs = [p.x for p in poly]
        ys = [p.y for p in poly]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def reset(self, position: tuple[float, float], heading_degrees: float) -> None:
        """
        Reset the car state to a known position & orientation.
        Used for restarting races or when car goes off-track badly.
        """
        self.position = vec2(*position)
        self.velocity = vec2(0, 0)
        self.heading = heading_degrees

        self.current_lap = 0
        self.distance_along_track = 0.0
        self.best_lap_time = None
        self.current_lap_time = 0.0
        self.finished = False

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draw the car onto the given surface.
        To keep it simple, we draw a rotated rectangle plus a small "front" marker.
        """
        poly = self.get_polygon()

        # Fill car body.
        pygame.draw.polygon(surface, self.color, [(p.x, p.y) for p in poly])

        # Draw a small triangle or line at the front to indicate direction.
        # We'll use the midpoint of the front edge.
        front_mid = (poly[0] + poly[1]) / 2
        forward = get_forward_vector(self.heading)
        nose = front_mid + forward * 10

        pygame.draw.line(surface, (255, 255, 255), front_mid, nose, 2)
