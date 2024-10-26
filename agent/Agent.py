from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment
from strategy.Assignment import pass_reciever_selector
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance





    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
        return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation (optional)
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()

    def think_and_send(self):
        
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass


        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""



        



    # def select_skill(self,strategyData):
    #     #--------------------------------------- 2. Decide action

    #     drawer = self.world.draw
        
    #     path_draw_options = self.path_manager.draw_options

    #     target = (15,0) # Opponents Goal

    #     #------------------------------------------------------
    #     #Role Assignment
    #     if strategyData.active_player_unum == strategyData.robot_model.unum: # I am the active player 
    #         drawer.annotation((0,10.5), "Role Assignment Phase" , drawer.Color.yellow, "status")
    #     else:
    #         drawer.clear("status")

    #     formation_positions = GenerateBasicFormation()
    #     point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
    #     strategyData.my_desired_position = point_preferences[strategyData.player_unum]
    #     strategyData.my_desried_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.my_desired_position)

    #     drawer.line(strategyData.mypos, strategyData.my_desired_position, 2,drawer.Color.blue,"target line")

    #     if not strategyData.IsFormationReady(point_preferences):
    #         return self.move(strategyData.my_desired_position, orientation=strategyData.my_desried_orientation)
    #     #else:
    #     #     return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)


    
    #     #------------------------------------------------------
    #     #Pass Selector
    #     if strategyData.active_player_unum == strategyData.robot_model.unum: # I am the active player 
    #         drawer.annotation((0,10.5), "Pass Selector Phase" , drawer.Color.yellow, "status")
    #     else:
    #         drawer.clear_player()



    #     if strategyData.active_player_unum == strategyData.robot_model.unum: # I am the active player 
    #         target = pass_reciever_selector(strategyData.player_unum, strategyData.teammate_positions, (15,0))
    #         drawer.line(strategyData.mypos, target, 2,drawer.Color.red,"pass line")
    #         return self.kickTarget(strategyData,strategyData.mypos,target)
    #     else:
    #         drawer.clear("pass line")
    #         return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)

def select_skill(self, world):
    # Check if this player is the closest to the ball
    if self.active_player_unum == self.player_unum:
        # Determine if there is a closer teammate to the goal
        closest_teammate_to_goal = self.get_closest_teammate_to_goal(world)
        if closest_teammate_to_goal and closest_teammate_to_goal != self.player_unum:
            # Pass if a teammate is better positioned
            target_pos = self.teammate_positions[closest_teammate_to_goal - 1]
            return self.kick_to_target(target_pos)
        else:
            # Otherwise, attempt a direct goal kick
            return self.kick_to_target((15.05, 0))  # Opponent's goal position
    else:
        # Move to a formation position or ball's position if formation is not ready
        if not self.IsFormationReady(self.point_preferences):
            self.my_desired_position = self.point_preferences[self.player_unum]
        else:
            self.my_desired_position = self.ball_2d  # Move towards the ball
        return self.move_to_position(self.my_desired_position)

def get_closest_teammate_to_goal(self, world):
    # Find the teammate closest to the opponent's goal
    goal_pos = (15.05, 0)  # Assume opponent goal position
    min_dist = float('inf')
    closest_teammate = None

    for i, teammate_pos in enumerate(self.teammate_positions):
        if teammate_pos is not None and i != self.player_unum - 1:
            dist = np.linalg.norm(np.array(teammate_pos) - np.array(goal_pos))
            if dist < min_dist:
                min_dist = dist
                closest_teammate = i + 1
    return closest_teammate

def move_to_position(self, target_position):
    direction = self.GetDirectionRelativeToMyPositionAndTarget(target_position)
    return f"move({direction})"

def kick_to_target(self, target_position):
    direction = self.GetDirectionRelativeToMyPositionAndTarget(target_position)
    return f"kick({direction})"

            





        # if strategyData.PM == self.world.M_GAME_OVER:
        #     pass
        # elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
        #     self.beam()
        # elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
        #     self.beam(True) # avoid center circle
        # elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
        #     self.state = 0 if behavior.execute("Get_Up") else 1 # return to normal state if get up behavior has finished
        # elif strategyData.PM == self.world.M_OUR_KICKOFF:
        #     if strategyData.robot_model.unum == 9:
        #         self.kick(120,3) # no need to change the state when PM is not Play On
        #     else:
        #         self.move(self.init_pos, orientation=strategyData.ball_dir) # walk in place
        # elif strategyData.PM == self.world.M_THEIR_KICKOFF:
        #     self.move(self.init_pos, orientation=strategyData.ball_dir) # walk in place
        # elif strategyData.active_player_unum != strategyData.robot_model.unum: # I am not the active player
        #     if strategyData.robot_model.unum == 1: # I am the goalkeeper
        #         self.move(self.init_pos, orientation=strategyData.ball_dir) # walk in place 
        #     else:
        #         # compute basic formation position based on ball position
        #         new_x = max(0.5,(strategyData.ball_2d[0]+15)/15) * (self.init_pos[0]+15) - 15
        #         if strategyData.min_teammate_ball_dist < strategyData.min_opponent_ball_dist:
        #             new_x = min(new_x + 3.5, 13) # advance if team has possession
        #         self.move((new_x,self.init_pos[1]), orientation=strategyData.ball_dir, priority_unums=[strategyData.active_player_unum])

        # else: # I am the active player
        #     path_draw_options(enable_obstacles=True, enable_path=True, use_team_drawing_channel=True) # enable path drawings for active player (ignored if self.enable_draw is False)

            
        #     enable_pass_command = (strategyData.PM == self.world.M_PLAY_ON and strategyData.ball_2d[0]<6)

        #     if strategyData.robot_model.unum == 1 and strategyData.PM_GROUP == self.world.MG_THEIR_KICK: # goalkeeper during their kick
        #         self.move(self.init_pos, orientation=strategyData.ball_dir) # walk in place 
        #     if strategyData.PM == self.world.M_OUR_CORNER_KICK:
        #         self.kick( -np.sign(strategyData.ball_2d[1])*95, 5.5) # kick the ball into the space in front of the opponent's goal
        #         # no need to change the state when PM is not Play On
        #     elif strategyData.min_opponent_ball_dist + 0.5 < strategyData.min_teammate_ball_dist: # defend if opponent is considerably closer to the ball
        #         if self.state == 2: # commit to kick while aborting
        #             self.state = 0 if self.kick(abort=True) else 2
        #         else: # move towards ball, but position myself between ball and our goal
        #             self.move(strategyData.slow_ball_pos + M.normalize_vec((-16,0) - strategyData.slow_ball_pos) * 0.2, is_aggressive=True)
        #     else:
        #         self.state = 0 if self.kick(strategyData.goal_dir,9,False,enable_pass_command) else 2

        #     path_draw_options(enable_obstacles=False, enable_path=False) # disable path drawings


    

    #--------------------------------------- Fat proxy auxiliary methods


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")