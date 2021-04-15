from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
import sys
from models.DeepQLearner import *
from models.DeepQLearner2 import *
from builtins import range
from builtins import object
from textworld.logic import Action, Rule, Placeholder, Predicate, Proposition, Signature, State, Variable
from MalmoLogicState import *
from constants import *
from models.Agent import Agent
from PIL import Image
import imagehash as iHash
import MalmoPython
import json
import logging
import os
import random
import sys
import time
import torch
import torch.nn as nn
import cv2
if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk

world_bounds = ((-2, -2), (7, 13))

class CameraDQNAgent(Agent):
    """Deep Q-learning agent for discrete state/action spaces."""

    def __init__(self, agentHost=None):
        self.logger = logging.getLogger(__name__)
        self.verbose = agentHost.verbose
        self.no_training = agentHost.no_training
        if self.verbose: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))


        self.host = agentHost
        logicState = agentHost.state
        self.move_actions = ["movenorth 1", "moveeast 1", "movewest 1"]
        (x1, y1, z1), (x2, y2, z2) = logicState.world_bounds.roundPosition()

        # self.dyna_rate = 30
        self.learner = DeepQLearner2(
            input_size = 31,
            num_actions=len(self.move_actions) + len(logicState.actions),
            discount_factor = 0.9, # 0.9
            epsilon = 0.99, 
            epsilon_decay = 0.99,# 0.99
            epsilon_min = 0.01, # 0.01
            learning_rate = 0.0005, # 0.0005
            dyna_rate = 50, 
            clip = 0.3,
            load_path='cache/camera_dqn.pkl',
            save_path='cache/camera_dqn.pkl',
            camera=True,
            verbose=self.verbose)

        # set vars for draw_QTable method
        self.canvas = None
        self.root = None

        # set vars for collecting output and logging numbers in logs folder
        self.cumulative_rewards = []
        tstr = time.strftime("%Y%m%d-%H%M%S")
        self.logFile = 'CameraDQNAgent_Rewards-' + tstr + '.txt'
        self.lossFile = 'CameraDQNAgent_Losses-' + tstr + '.txt'
        self.saveSnapshots = True
        self.maxReward = -1000

    def updateGrammar(self, agentHost):
        self.host = agentHost

    def getActionSpace(self):
        return self.host.getApplicableActions()

    def getObservations(self, world_state):
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)
        if not u'XPos' in obs or not u'ZPos' in obs:
            raise RuntimeError("Incomplete observation received: %s" % obs_text)
        else:
            return obs

    def addObservations(self, observation):
        observation = observation or self.host.state
        self.host.updateLogicState(observation)

    def setState(self, world_state):
        self.host.updateLogicState(world_state)

    def train(self):
        raise NotImplementedError

    def processFrame(self, pixels, width, height, state):

        # converts byte arr into image
        embedding = np.array(pixels).reshape(height, width, 3)
        # flips channel to give rgb image instead of blue scaled
        embedding = np.flip(embedding, -1)
        if self.saveSnapshots and self.verbose: # save image
            curr_hash = iHash.average_hash(Image.fromarray(embedding))
            cv2.imwrite('./images/' + str(curr_hash) + '-' + time.strftime("%Y%m%d-%H%M%S") + '.jpg', embedding)
        embedding = np.moveaxis(embedding, -1, 0)
        return (embedding, state)

    def queryActions(self, frame, world_state, current_reward ):
        """
        take 1 action in response to the current world state
        
        UNUSED METHOD
        """

        # Set malmo system to current world state
        self.setState(world_state)
        try:
            obs = self.getObservations(world_state) # get and print most recent observation
        except RuntimeError as e:
            self.logger.error(e)
            return 0
        self.logger.debug("\n Full Observation: " + str(obs))

        current_state = self.host.state.getStateEmbedding(includePos=False)
        logicalActions = self.host.state.getApplicableActions()
        allActions = self.move_actions + logicalActions
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_state, float(obs[u'XPos']), float(obs[u'ZPos'])))
        current_state = self.processFrame(frame.pixels, frame.width, frame.height, current_state)

        # update Q values
        if self.prev_state is not None and self.prev_action is not None:\
            action = self.learner.query(current_state, current_reward)
        else:
            action = self.learner.querysetstate(current_state)

        if self.verbose: self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) )

        self.logger.info("Taking q action: %s" % allActions[action % len(allActions)])

        # try to send the selected action, only update prev_s if this succeeds
        try:
            self.host.sendCommand(allActions[action % len(allActions)], is_logical = action % len(allActions) >= len(self.move_actions))
            self.prev_state = current_state
            self.prev_action = action

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

        return allActions[action], current_reward

    def act(self, frame, world_state, current_reward ):
        """take 1 action in response to the current world state"""

        # Set malmo system to current world state
        self.setState(world_state)
        try:
            obs = self.getObservations(world_state) # get and print most recent observation
        except RuntimeError as e:
            self.logger.error(e)
            return 0
        self.logger.debug("\n Full Observation: " + str(obs))

        current_state = self.host.state.getStateEmbedding() # param includePos=False deleted for method definition in MalmoLogicState.py
        logicalActions = self.host.state.getApplicableActions()
        # if len(logicalActions) > 0:
        #     print(logicalActions[0])
        allActions = self.move_actions + logicalActions
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_state, float(obs[u'XPos']), float(obs[u'ZPos'])))
        current_state = self.processFrame(frame.pixels, frame.width, frame.height, current_state)
        #current_state.append(frame.pixels)

        # update Q values
        if self.prev_state is None or self.prev_action is None or self.no_training: # changed and to or (double check)
            action = self.learner.querysetstate(current_state)
        else:
            action = self.learner.query(current_state, current_reward)
            self.learner.run_dyna()

        if self.verbose: self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) )
        
        # if action % len(allActions) > 3:
            # print(allActions[action % len(allActions)])
        self.logger.info("Taking q action: %s" % allActions[action % len(allActions)])

        # try to send the selected action, only update prev_s if this succeeds
        try:
            self.host.sendCommand(allActions[action % len(allActions)], is_logical = action % len(allActions) >= len(self.move_actions))
            self.prev_state = current_state
            self.prev_action = action

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

        return current_reward

    def run(self):
        """run the agent on the world"""

        total_reward = 0

        self.prev_state = None
        self.prev_action = None

        is_first_action = True

        # main loop:
        world_state = self.host.getWorldState()
        while world_state.is_mission_running:

            current_reward = 0

            if is_first_action:
                self.host.resetState()
                # wait until have received action valid observation
                while True:
                    time.sleep(0.1)
                    world_state = self.host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_reward += reward.getValue()
                    current_reward += self.host.rewardValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}" and world_state.number_of_video_frames_since_last_state > 0:
                        frame = world_state.video_frames[-1]
                        total_reward += self.act(frame, world_state, current_reward)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_reward == 0:
                    time.sleep(0.1)
                    world_state = self.host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_reward += reward.getValue()
                    current_reward += self.host.rewardValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = self.host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_reward += reward.getValue()
                    current_reward += self.host.rewardValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}" and world_state.number_of_video_frames_since_last_state > 0:
                        frame = world_state.video_frames[-1]
                        total_reward += self.act(frame, world_state, current_reward)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % current_reward)
        total_reward += current_reward

        # update Q values
        # final_s = (np.array(frame.pixels).reshape(3, frame.height, frame.width), self.host.state.getStateEmbedding())
        # self.learner.query(final_s, current_reward)

        # double check if this improves accuracy
        # if self.counter%self.dyna_rate == 0 and not self.no_training:
        #     if self.verbose: print("Running dyna replay...")
        # self.learner.run_dyna()

        self.learner.updateEps()
        
        if self.verbose: self.drawQ()
        self.cumulative_rewards.append(total_reward)

        plt.clf()
        plt.plot(self.cumulative_rewards)
        plt.xlabel("Iterations")
        plt.ylabel("Rewards")
        plt.title("Running track of rewards")
        plt.savefig(f"./graphs/CameraDQNRunning.png")
        if (len(self.cumulative_rewards) > 2):
            if (self.cumulative_rewards[-1] > self.maxReward):
                # self.learner.save()
                self.logOutput()
                self.maxReward = self.cumulative_rewards[-1]
                print(f"Saved Model with reward {self.cumulative_rewards[-1]}")
                print()
        return total_reward

    def drawQ( self, curr_x=None, curr_y=None ):
        scale = 40
        world_x = 6
        world_y = 14
        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
            self.canvas.grid()
            self.root.update()
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        curr_radius = 0.2
        action_positions = [ ( 0.5, action_inset ), ( 0.5, 1-action_inset ), ( action_inset, 0.5 ), ( 1-action_inset, 0.5 ) ]
        # (NSWE to match action order)
        min_value = -20
        max_value = 20
        suffixes = ["000:01", "000:11", "000:10", "000:00"]
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d|" % (x,y)
                self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")
                for action in range(4):
                    #if not s in self.q_table:
                    #    continue
                    values = []
                    #for suf in suffixes:
                    #    if s + suf in self.q_table:
                    #        values.append(self.q_table[s + suf][action])
                    if len(values) == 0:
                        continue
                    value = float(sum(values)) / len(values)
                    color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
                    color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                    color_string = '#%02x%02x%02x' % (255-color, color, 0)
                    self.canvas.create_oval( (x + action_positions[action][0] - action_radius ) *scale,
                                             (y + action_positions[action][1] - action_radius ) *scale,
                                             (x + action_positions[action][0] + action_radius ) *scale,
                                             (y + action_positions[action][1] + action_radius ) *scale,
                                             outline=color_string, fill=color_string )
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale,
                                     (curr_y + 0.5 - curr_radius ) * scale,
                                     (curr_x + 0.5 + curr_radius ) * scale,
                                     (curr_y + 0.5 + curr_radius ) * scale,
                                     outline="#fff", fill="#fff" )
        self.root.update()

    def logOutput(self):
        self.learner.save()
        with open(os.path.join('logs', self.logFile), 'w') as f:
            for item in self.cumulative_rewards:
                f.write("%s\n" % item)
        with open(os.path.join('logs', self.lossFile), 'w') as f:
            for item in self.learner.losses:
                f.write("%s\n" % item)