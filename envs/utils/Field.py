import pygame
from rsoccer_gym.Entities import Field as CommonField

from envs.utils.Point import Point

class Field():
    def __init__(self, field: CommonField):
        self.length = field.length
        self.width = field.width
        self.penalty_length = field.penalty_length
        self.penalty_width = field.penalty_width
        self.goal_width = field.goal_width
        self.goal_depth = field.goal_depth
        self.ball_radius = field.ball_radius
        self.rbt_distance_center_kicker = field.rbt_distance_center_kicker
        self.rbt_kicker_thickness = field.rbt_kicker_thickness
        self.rbt_kicker_width = field.rbt_kicker_width
        self.rbt_wheel0_angle = field.rbt_wheel0_angle
        self.rbt_wheel1_angle = field.rbt_wheel1_angle
        self.rbt_wheel2_angle = field.rbt_wheel2_angle
        self.rbt_wheel3_angle = field.rbt_wheel3_angle
        self.rbt_radius = field.rbt_radius
        self.rbt_wheel_radius = field.rbt_wheel_radius
        self.rbt_motor_max_rpm = field.rbt_motor_max_rpm

        # New attributes
        self.right_goal_outside_bottom = Point(self.length / 2, self.goal_width / 2)
        self.right_goal_outside_top = Point(self.length / 2, -self.goal_width / 2)
        self.right_goal_outside_center = Point(self.length / 2, 0)
        self.top_center = Point(0, -self.width / 2)
        self.bottom_center = Point(0, self.width / 2)

        self.left_goal_outside_bottom = Point(-self.length / 2, self.goal_width / 2)
        self.left_goal_outside_top = Point(-self.length / 2, -self.goal_width / 2)
        self.left_goal_outside_center = Point(-self.length / 2, 0)

    def render_points(self, window_surface, field_renderer):
        def pos_transform(pos_x, pos_y):
            return (
                int(pos_x * field_renderer.scale + field_renderer.center_x),
                int(pos_y * field_renderer.scale + field_renderer.center_y),
            )
        
        pygame.font.init() 
        font = pygame.font.SysFont('Comic Sans MS', 30)
        
        points = [
            ["RGOB", self.right_goal_outside_bottom],
            ["RGOT", self.right_goal_outside_top],
            ["RGOC", self.right_goal_outside_center],
            ["TC", self.top_center],
            ["BC", self.bottom_center],
            ["LGOB", self.left_goal_outside_bottom],
            ["LGOT", self.left_goal_outside_top],
            ["LGOC", self.left_goal_outside_center]
        ]

        for point in points:
            txt = font.render(point[0], True, (255, 0, 0))
            window_surface.blit(txt, pos_transform(point[1].x, point[1].y - 0.05))
            pygame.draw.circle(
                window_surface,
                (0, 0, 0),
                pos_transform(point[1].x, point[1].y),
                5,
            )