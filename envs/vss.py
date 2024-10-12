import math
import random
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import gymnasium as gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.Utils import KDTree

from envs.utils.Field import Field
from envs.utils.Point import Point


class VSSAttackerEnv(VSSBaseEnv):
    """This environment controls a single robot in a VSS soccer League 3v3 match


    Description:
    Observation:
        Type: Box(46)
        Normalized Bounds to [-1.25, 1.25]
        Num             Observation normalized
        0               Ball X
        1               Ball Y
        2               Ball Vx
        3               Ball Vy
        4 + (7 * i)     id i Blue Robot X
        5 + (7 * i)     id i Blue Robot Y
        6 + (7 * i)     id i Blue Robot sin(theta)
        7 + (7 * i)     id i Blue Robot cos(theta)
        8 + (7 * i)     id i Blue Robot Vx
        9  + (7 * i)    id i Blue Robot Vy
        10 + (7 * i)    id i Blue Robot v_theta
        25 + (5 * i)    id i Yellow Robot X
        26 + (5 * i)    id i Yellow Robot Y
        27 + (5 * i)    id i Yellow Robot sin(theta)
        28 + (5 * i)    id i Yellow Robot cos(theta)
        29 + (5 * i)    d i Yellow Robot Vx
        30 + (7 * i)    id i Yellow Robot Vy
        31 + (7 * i)    id i Yellow Robot v_theta
    Actions:
        Type: Box(2, )
        Num     Action
        0       id 0 Blue Left Wheel Speed  (%)
        1       id 0 Blue Right Wheel Speed (%)
    Reward:
        Sum of Rewards:
            Goal
            Ball Potential Gradient
            Move to Ball
            Energy Penalty
    Starting State:
        Randomized Robots and Ball initial Position
    Episode Termination:
        5 minutes match time
    """

    def __init__(self, render_mode=None, max_steps=1200):
        super().__init__(
            field_type=0,
            n_robots_blue=3,
            n_robots_yellow=3,
            time_step=0.025,
            render_mode=render_mode,
        )

        self.max_steps = max_steps

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS, high=self.NORM_BOUNDS, shape=(46,), dtype=np.float32
        )

        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_shaping_total = None
        self.v_wheel_deadzone = 0.05

        self.ou_actions = []
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )
        
        MAX_WHEEL_SPEED = 50 # 100 rad/s
        self.max_v = MAX_WHEEL_SPEED * self.field.rbt_wheel_radius

        self.field = Field(self.field)

    def reset(self, *, seed=None, options=None):
        self.actions = None
        self.reward_shaping_total = None
        self.previous_ball_potential = None
        for ou in self.ou_actions:
            ou.reset()
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        return observation, reward, terminated, truncated, self.reward_shaping_total

    def _frame_to_observations(self):
        observation = []

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(np.sin(np.deg2rad(self.frame.robots_blue[i].theta)))
            observation.append(np.cos(np.deg2rad(self.frame.robots_blue[i].theta)))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
            observation.append(np.sin(np.deg2rad(self.frame.robots_yellow[i].theta)))
            observation.append(np.cos(np.deg2rad(self.frame.robots_yellow[i].theta)))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
            observation.append(self.norm_w(self.frame.robots_yellow[i].v_theta))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        self.actions[0] = actions
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0, v_wheel1=v_wheel1))

        # Send random commands to the other robots
        for i in range(1, self.n_robots_blue):
            actions = self.ou_actions[i].sample()
            self.actions[i] = actions
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(
                Robot(yellow=False, id=i, v_wheel0=v_wheel0, v_wheel1=v_wheel1)
            )
        for i in range(self.n_robots_yellow):
            actions = self.ou_actions[self.n_robots_blue + i].sample()
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(
                Robot(yellow=True, id=i, v_wheel0=v_wheel0, v_wheel1=v_wheel1)
            )

        return commands

    def _calculate_reward_and_done(self):
        reward = 0
        goal = False
        w_move = 0.2
        w_ball_grad = 0.8
        w_energy = 2e-4
        w_goal = 10
        w_ang_vel = 0.2
        w_obs = 0.2
        w_ball_to_goal = 0.8
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                "goal_score": 0,
                "move": 0,
                "ball_grad": 0,
                "energy": 0,
                "goals_blue": 0,
                "goals_yellow": 0,
                "ball_to_goal": 0,
                # "ang_vel": 0,
                # "obstacle": 0,
            }

        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            self.reward_shaping_total["goal_score"] += 1
            self.reward_shaping_total["goals_blue"] += 1
            reward = w_goal
            goal = True
        elif self.frame.ball.x < -(self.field.length / 2):
            self.reward_shaping_total["goal_score"] -= 1
            self.reward_shaping_total["goals_yellow"] += 1
            reward = -w_goal
            goal = True
        else:
            if self.last_frame is not None:
                # Calculate ball potential
                grad_ball_potential = self._ball_grad()
                # Calculate Move ball
                move_reward = self._move_reward()
                # Calculate Energy penalty
                energy_penalty = self._energy_penalty()
                # # Calculate High Angular Velocity Penalty
                # angular_vel_penalty = self._high_angular_velocity_penalty()
                # # Calculate Obstacle Penalty
                # obstacle_reward = self._obstacle_reward()

                ball_to_goal_reward = self._ball_towards_goal_reward()
                ball_to_goal_penalty = self._ball_towards_goal_reward(enemy_goal=False)
                total_ball_to_goal_reward = ball_to_goal_reward - ball_to_goal_penalty

                reward = (
                    w_move * move_reward
                    + w_ball_grad * grad_ball_potential
                    + w_energy * energy_penalty
                    + w_ball_to_goal * total_ball_to_goal_reward
                    # + w_ang_vel * angular_vel_penalty
                    # + w_obs * obstacle_reward
                )

                # print("Reward: ", reward)
                # print("Move to Ball Reward: ", move_reward * w_move)
                # print("Ball Grad Reward: ", grad_ball_potential * w_ball_grad)
                # print("Energy Penalty: ", energy_penalty * w_energy)
                # print("Angular Vel Penalty: ", angular_vel_penalty * w_ang_vel)
                # print("Ball to Goal Reward: ", total_ball_to_goal_reward * w_ball_to_goal)

                self.reward_shaping_total["move"] += w_move * move_reward
                self.reward_shaping_total["ball_grad"] += (
                    w_ball_grad * grad_ball_potential
                )
                self.reward_shaping_total["energy"] += w_energy * energy_penalty
                self.reward_shaping_total["ball_to_goal"] += w_ball_to_goal * total_ball_to_goal_reward
                # self.reward_shaping_total["ang_vel"] += w_ang_vel * angular_vel_penalty
                # self.reward_shaping_total["obstacle"] += obstacle_reward

        return reward, goal
    
    def __is_ball_towards_goal(self, ball: Point, ball_velocity: Point, goal_outside_bottom: Point, goal_outside_top: Point):
        bottom_verser_x = (goal_outside_bottom.x - ball.x) / \
                          (ball.dist_to(goal_outside_bottom) + 1e-9)
        bottom_verser_y = (goal_outside_bottom.y - ball.y) / \
                          (ball.dist_to(goal_outside_bottom) + 1e-9)
        bottom_verser = Point(bottom_verser_x, bottom_verser_y)

        top_verser_x = (goal_outside_top.x - ball.x) / \
                       (ball.dist_to(goal_outside_top) + 1e-9)
        top_verser_y = (goal_outside_top.y - ball.y) / \
                       (ball.dist_to(goal_outside_top) + 1e-9)
        top_verser = Point(top_verser_x, top_verser_y)

        vel_verser = ball_velocity * (1 / (ball_velocity.length() + 1e-9))

        if vel_verser.dot(bottom_verser) > bottom_verser.dot(top_verser) and \
           vel_verser.dot(top_verser) > bottom_verser.dot(top_verser):
            return True
        else:
            return False
        
    def _ball_towards_goal_reward(self, enemy_goal=True):
        ball = Point(self.frame.ball.x, self.frame.ball.y)
        ball_velocity = Point(self.frame.ball.v_x, self.frame.ball.v_y)

        if not enemy_goal:
            goal_outside_bottom = self.field.left_goal_outside_bottom
            goal_outside_top = self.field.left_goal_outside_top
        else:
            goal_outside_bottom = self.field.right_goal_outside_bottom
            goal_outside_top = self.field.right_goal_outside_top

        if self.__is_ball_towards_goal(ball, ball_velocity, goal_outside_bottom, goal_outside_top):
            return 10 * ball_velocity.length()
        else:
            return -1 * ball_velocity.length()

    def _get_initial_positions_frame(self):
        """Returns the position of each robot and ball for the initial frame"""
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x():
            return random.uniform(-field_half_length + 0.1, field_half_length - 0.1)

        def y():
            return random.uniform(-field_half_width + 0.1, field_half_width - 0.1)

        def theta():
            return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=x(), y=y())

        min_dist = 0.1

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        left_wheel_speed = actions[0] * self.max_v
        right_wheel_speed = actions[1] * self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed, right_wheel_speed

    def _ball_grad(self):
        """Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        """
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0) + self.field.goal_depth

        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100

        dist_1 = -math.sqrt(dx_a**2 + 2 * dy**2)
        dist_2 = math.sqrt(dx_d**2 + 2 * dy**2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            diff = ball_potential - self.previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.time_step, -5.0, 5.0)

        self.previous_ball_potential = ball_potential

        return grad_ball_potential

    def _move_reward(self):
        """Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        """

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        robot = np.array([self.frame.robots_blue[0].x, self.frame.robots_blue[0].y])
        robot_vel = np.array(
            [self.frame.robots_blue[0].v_x, self.frame.robots_blue[0].v_y]
        )
        robot_ball = ball - robot
        robot_ball = robot_ball / np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward

    def _energy_penalty(self):
        """Calculates the energy penalty"""

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = -(en_penalty_1 + en_penalty_2)
        return energy_penalty

    def _high_angular_velocity_penalty(self):
        """Calculates the high angular velocity penalty
        
            If the robot is rotating too fast and the ball is far away, the robot is penalized.
            This logic is to handle the case where the robot is rotating in place and not moving towards the ball.

            The penalty is calculated based on the value of the Hyperbolic tangent function, 
            which returns more negative rewards when the ball is far away and positive rewards when the ball is close.
            This function has the y-axis limits based on the angular velocity of the robot.
        """

        angular_vel_penalty = 0

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        dist_to_ball = math.sqrt((ball.x - robot.x) ** 2 + (ball.y - robot.y) ** 2)
        linear_vel = math.sqrt(robot.v_x ** 2 + robot.v_y ** 2)
        angular_vel = abs(np.deg2rad(robot.v_theta))
        rbt_axis = self.field.rbt_radius * 2

        if dist_to_ball > rbt_axis:
            robot_axis = self.field.rbt_radius * 2
            angular_vel_penalty = np.tanh(robot_axis - dist_to_ball) * angular_vel

            if angular_vel_penalty > 0:
                angular_vel_penalty = 0

        ball_vel_norm = math.sqrt(ball.v_x ** 2 + ball.v_y ** 2)
        if ball.v_x > 0:
            angular_vel_penalty += ball_vel_norm
        else:
            angular_vel_penalty -= ball_vel_norm

        return angular_vel_penalty

    def _check_collision(self):
        for i in range(len(self.frame.robots_yellow)):
            obstacle_pos = np.array(
                [
                    self.frame.robots_yellow[i].x,
                    self.frame.robots_yellow[i].y,
                ]
            )
            agent_pos = np.array(
                (
                    self.frame.robots_blue[0].x,
                    self.frame.robots_blue[0].y,
                )
            )
            dist = np.linalg.norm(agent_pos - obstacle_pos)
            if dist < 0.2:
                return True
        return False

    def _obstacle_reward(self):
        reward = 0
        agent_pos = np.array(
            (
                self.frame.robots_blue[0].x,
                self.frame.robots_blue[0].y,
            )
        )
        for i in range(len(self.frame.robots_yellow)):
            obstacle_pos = np.array(
                [
                    self.frame.robots_yellow[i].x,
                    self.frame.robots_yellow[i].y,
                ]
            )
            dist = np.linalg.norm(agent_pos - obstacle_pos)
            std = 1
            exponential = np.exp((-0.5) * (dist / std) ** 2)
            gaussian = exponential / (std * np.sqrt(2 * np.pi))
            reward -= gaussian
        return reward
    
    def _render(self):
        super()._render()
        # self.field.render_points(self.window_surface, self.field_renderer)