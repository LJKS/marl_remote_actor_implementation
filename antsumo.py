import pybullet as p
import os
import time
import pybullet_data
import numpy as np
import math

STRENGTH = 250*4 #as motivated by pybullet implementation
MIN_DIST = 1
MAX_DIST = 2
MIN_ARENA = 2
MAX_ARENA = 3
START_HEIGHT = .5
MAX_STEPS = 8000
MOTOR_COST = -0.0
KING_OF_THE_HILL_REWARD = 0.001
MIN_HEIGHT = 0.27 # as motivated by pybulletgym
GRAVITY = -9.8
WON_REWARD = 10
LOST_REWARD = -10
DRAW_REWARD = -10
ACTION_SIZE = 8
LINK_OBS_LIST_IDX = [4,5]


class Ant_Sumo_Gym():
    def __init__(self, opponent, mode=p.DIRECT):
        self.opponent = opponent
        self.mode = mode
        self.arena_radius = np.random.uniform(low=MIN_ARENA,high=MAX_ARENA)
        self.clientId = p.connect(mode)
        self.gravity = GRAVITY
        self.rad = None #Set by genearte random start
        p.setGravity(gravX=0,gravY=0,gravZ=self.gravity, physicsClientId=self.clientId)
        p.setTimeStep(1/60, physicsClientId=self.clientId)
        full_path = 'robotics_files/ant.xml'
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        ground = p.loadURDF("plane_transparent.urdf", useMaximalCoordinates=True, physicsClientId=self.clientId)
        player_robotpos, player_robotorient, opponent_robotpos, opponent_robotorient = self.generate_random_start()
        self.player_robot = self.Ant(clientId=self.clientId, path=full_path, position=player_robotpos, orientation=player_robotorient)
        self.player_robot.generate_observation()
        self.opponent_robot = self.Ant(clientId=self.clientId, path=full_path, position=opponent_robotpos, orientation=opponent_robotorient)
        self.arena = self.create_arena()
        self.action_size=8
        self.current_step = 0
        self.agent_won_flag = 0
        self.max_steps = MAX_STEPS
        self.motor_cost = MOTOR_COST
        self.min_height = MIN_HEIGHT
        self.draw_reward = DRAW_REWARD
        self.won_reward = WON_REWARD
        self.lost_reward = LOST_REWARD
        self.king_of_the_hill_reward = KING_OF_THE_HILL_REWARD
        self.reset()

        """for i in range(20000):
            #if i < 40:
            #    p.applyExternalForce(objectUniqueId=self.robot[0], linkIndex=-1, forceObj=[0,0,10000],posObj=[0,0,0], flags=p.WORLD_FRAME)
            p.stepSimulation(self.clientId)
            print(p.getBasePositionAndOrientation(bodyUniqueId=self.player_robot.robotId, physicsClientId=self.clientId))
            #time.sleep(1/240)
            time.sleep(0.05)
            print('obs')
            self.player_robot.generate_observation()
            random_action1 = (np.random.rand(8) - .5)*2
            random_action2 = (np.random.rand(8) - .5)*2
            #p.resetBasePositionAndOrientation(bodyUniqueId=self.player_robot.robotId, posObj=player_robotpos, ornObj=[0,0,i/50,1], physicsClientId=self.clientId)
            self.player_robot.set_action(random_action1)
            self.opponent_robot.set_action(random_action2)
            #self.player_robot.set_action(np.ones(8))
            #self.opponent_robot.set_action(np.ones(8))"""


    def reset(self):
        p.stepSimulation(self.clientId)
        player_robotpos, player_robotorient, opponent_robotpos, opponent_robotorient = self.generate_random_start()
        self.player_robot.set_position(player_robotpos, player_robotorient)
        self.opponent_robot.set_position(opponent_robotpos, opponent_robotorient)
        return self.generate_player_observation()

    def step(self, action):
        action=np.squeeze(action)
        opponent_action, _ = self.opponent.act(self.generate_opponent_observation())
        opponent_action=np.squeeze(opponent_action)
        self.opponent_robot.set_action(opponent_action)
        self.player_robot.set_action(action)
        p.stepSimulation(self.clientId)

        done_flag = False
        reward = 0
        opponent_dist, opponent_height = self.opponent_robot.get_dist_to_origin_and_height()
        if (opponent_dist > self.arena_radius) or (opponent_height < self.min_height):
            done_flag = True
            reward += self.won_reward
        player_dist, player_height = self.player_robot.get_dist_to_origin_and_height()
        if (player_dist > self.arena_radius) or (player_height < self.min_height):
            done_flag = True
            reward += self.lost_reward
        if (self.current_step > self.max_steps) and not done_flag:
            reward += self.draw_reward
            done_flag = True

        reward += self.support_reward(player_dist) + self.motor_reward(action)
        reward = float(reward)
        self.current_step += 1
        self.agent_won_flag += reward
        return self.generate_player_observation(), reward, done_flag, None

    def support_reward(self, dist_to_middle):
        relativ_middelity = 1-(dist_to_middle/self.arena_radius)
        king_of_the_hill_influence =relativ_middelity*self.king_of_the_hill_reward
        return king_of_the_hill_influence

    def motor_reward(self, action):
        return np.sum(self.motor_cost*action)

    def generate_player_observation(self):
        obs = np.asarray(self.player_robot.generate_observation() + self.opponent_robot.generate_observation() + self.generate_common_observation())
        return np.reshape(obs, (1,-1))

    def generate_opponent_observation(self):
        obs = np.asarray(self.opponent_robot.generate_observation() + self.player_robot.generate_observation() + self.generate_common_observation())
        return np.reshape(obs, (1,-1))


    def generate_common_observation(self):
        return [self.current_step, self.arena_radius]

    def get_observation_shape(self):
        return self.generate_player_observation().shape

    def get_action_size(self):
        return self.action_size

    def agent_won(self):
        return self.agent_won_flag



    class Ant():
        def __init__(self, clientId, path, position, orientation):
            self.clientId = clientId
            self.robotId = p.loadMJCF(path, physicsClientId=self.clientId, flags=p.URDF_USE_SELF_COLLISION|p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)[0]
            p.resetBasePositionAndOrientation(bodyUniqueId=self.robotId, posObj=position, ornObj=orientation, physicsClientId=self.clientId)
            self.setCollision()
            self.get_motor_joints()
            self.strength = STRENGTH
            self.motor_joints, self.named_motor_joints, self.joints = self.get_motor_joints()
            #print(self.motor_joints, self.named_motor_joints)

        def set_position(self, position, orientation):
            p.resetBasePositionAndOrientation(bodyUniqueId=self.robotId, posObj=position, ornObj=orientation, physicsClientId=self.clientId)

        def setCollision(self):
            for joint in range(p.getNumJoints(bodyUniqueId=self.robotId, physicsClientId=self.clientId)):
                p.setCollisionFilterGroupMask(bodyUniqueId=self.robotId, linkIndexA=joint, collisionFilterGroup=1, collisionFilterMask=1)

        def get_motor_joints(self):
            motor_joints = []
            named_motor_joints = []
            joints = []

            for i in range(p.getNumJoints(bodyUniqueId=self.robotId, physicsClientId=self.clientId)):
                info = p.getJointInfo(bodyUniqueId=self.robotId, jointIndex=i,physicsClientId=self.clientId)
                # info[2] describes the joint, 0 is a torque motor
                if info[2] == 0:
                    motor_joints.append(info[0])
                    named_motor_joints.append(str(info[1]))
                joints.append(info[0])
            return motor_joints, named_motor_joints, joints



        def set_action(self, action):
            action=np.clip(action, -1,1)
            scaled_actions = [ac*self.strength for ac in action.tolist()]
            p.setJointMotorControlArray(bodyIndex=self.robotId, jointIndices=self.motor_joints, controlMode=p.TORQUE_CONTROL, forces=scaled_actions)

        def get_dist_to_origin_and_height(self):
            pos, _ = p.getBasePositionAndOrientation(self.robotId, self.clientId)
            dist = math.sqrt((pos[0]**2) + (pos[1]**2))
            height = pos[2]
            return dist, height

        def generate_observation(self):
            obs_list = []
            for obs in p.getBasePositionAndOrientation(bodyUniqueId=self.robotId, physicsClientId=self.clientId):
                obs_list = obs_list + list(obs)
            for joint in self.joints:
                joint_obs_list = p.getLinkState(self.robotId, joint, self.clientId)
                for sub_obs in [joint_obs_list[idx] for idx in LINK_OBS_LIST_IDX]:
                    obs_list = obs_list + list(sub_obs)
            return obs_list

    def generate_random_start(self):
        rad = np.random.rand(1)*np.pi*2
        random_scale = np.random.uniform(MIN_DIST, MAX_DIST)
        start_vec_1 = [np.cos(rad)[0]*random_scale,np.sin(rad)[0]*random_scale]
        start_vec_2 = [i*-1 for i in start_vec_1]
        start_vec_1.append(START_HEIGHT)
        start_vec_2.append(START_HEIGHT)
        orientation_1 = p.getQuaternionFromEuler(eulerAngles=[0,0,rad], physicsClientId=self.clientId)
        orientation_2 = p.getQuaternionFromEuler(eulerAngles=[0,0,rad+np.pi], physicsClientId=self.clientId)
        return start_vec_1, orientation_1, start_vec_2, orientation_2

    def create_arena(self):
        arenaVis = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=self.arena_radius, length=0.01,visualFramePosition=[0,0,0], visualFrameOrientation=[0,0,0,1], rgbaColor=[0.5,0.5,0.5,0.5], physicsClientId=self.clientId)
        arena = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=arenaVis, basePosition=[0,0,0.01])
        p.setCollisionFilterGroupMask(bodyUniqueId=arena, linkIndexA=0, collisionFilterGroup=0, collisionFilterMask=0)
        return arena



class Dummy:
    def __init__(self):
        pass
    def act(self, dummy):
        return (np.random.rand(8) - .5)*2, None
if __name__ == '__main__':
    opponent = Dummy()
    test=Ant_Sumo_Gym(opponent, mode=p.GUI)
    test.reset()
    test2=Ant_Sumo_Gym(opponent)
    test3=Ant_Sumo_Gym(opponent)
    for _ in range(10000):
        print(test.step((np.random.rand(8) - .5)*2)[2])
