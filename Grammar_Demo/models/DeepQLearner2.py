import numpy as np
import random as rand
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime
#from collections import deque

class NeuralNet(nn.Module):
    def __init__(self, input_size=31, hidden_size=20, num_classes=2):
        super(NeuralNet, self).__init__()

        self.fc0 = nn.Linear(input_size, 31)
        self.activate0 = nn.LeakyReLU()
        self.fc1 = nn.Linear(31, 31)
        self.activate1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=0.05)
        self.fc2 = nn.Linear(31, 23)
        self.activate2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p=0.05)
        self.fc3 = nn.Linear(23, 15)
        self.activate3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(p=0.05)
        self.fc4 = nn.Linear(15, num_classes)

    def forward(self, x):
        # return self.nn_no_dropout(x)
        out = self.fc0(x)
        out = self.activate0(out)
        out = self.fc1(x)
        out = self.activate1(out)
        #out = self.dropout1(out)
        out = self.fc2(out)
        out = self.activate2(out)
        #out = self.dropout2(out)
        out = self.fc3(out)
        out = self.activate3(out)
        #out = self.dropout3(out)
        out = self.fc4(out)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class CNN(nn.Module):
    def __init__(self, input_size=31, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(720 + input_size, 128)
        self.activate1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.apply(weights_init)

    def forward(self, x, y):
        # Convolution
        out = self.pool1(F.leaky_relu(self.conv1(x)))
        out = self.pool2(F.leaky_relu(self.conv2(out)))
        out = torch.flatten(out, 1)
        out = torch.cat((out, y), 1)
        # Feedforward
        out = self.fc1(out)
        out = self.activate1(out)
        out = self.fc2(out)
        return out

class DeepQLearner2(object):

    def __init__(self, \
        input_size = 31, \
        num_actions = 2, \
        discount_factor = 0.9, \
        epsilon = 0.2, \
        epsilon_decay = 0.99, \
        epsilon_min = 0.01, \
        clip = 1, \
        learning_rate = 0.0002, \
        dyna_rate = 10, \
        load_path = None, \
        save_path = None, \
        camera = False, \
        verbose = False):

        self.verbose = verbose
        self.input_size = input_size
        self.num_actions = num_actions
        self.state = [0] * input_size
        self.action = 0
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.clip = clip
        self.max_samples = 5000
        self.replay_sample = dyna_rate
        self.counter = 0
        self.update_rate = 5
        self.camera = camera
        self.learning_rate = learning_rate
        self.samples = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not self.camera:
            self.policy_net = NeuralNet(input_size = input_size, num_classes=num_actions).to(self.device)
            self.target_net = NeuralNet(input_size = input_size, num_classes=num_actions).to(self.device)
        else:
            self.policy_net = CNN(input_size = input_size, num_classes=num_actions).to(self.device)
            self.target_net = CNN(input_size = input_size, num_classes=num_actions).to(self.device)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        #self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=learning_rate)

        # Load saved model
        self.save_path = save_path

        if load_path != None and os.path.exists(save_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.target_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded Model!")
        self.target_net.eval()

        self.losses = []

    def author(self):
        return 'kxiao36'

    def updateEps(self):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        return

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action

        Specifically called for the start state
        """
        self.policy_net.eval()
        with torch.no_grad():
            self.state = s
            if self.camera:
                output = self.policy_net(torch.Tensor([s[0]]).to(self.device), torch.Tensor([s[1]]).to(self.device))
            else:
                output = self.policy_net(torch.Tensor([s]).to(self.device))
            output_Q, output_action = torch.max(output.data, 1)
            self.action = output_action[0].item() if self.epsilon < rand.random() else rand.randint(0, self.num_actions - 1)
            return self.action

    def query(self,next_state,reward):
        """
        @summary: Update the Q table and return an action
        @param next_state: The new state
        @param reward: The reward
        @returns: The selected action
        """
        # add sample to array and pop if greater than max
        self.samples.append([self.state, self.action, next_state, reward])
        self.counter = (self.counter + 1) % self.update_rate
        while len(self.samples) > self.max_samples:
            self.samples.pop(0)
        # reward += 5 if not self.camera else 0

        # query target net for action and calculate expected reward
        self.policy_net.eval()
        with torch.no_grad():
            if self.camera:
                next_output = self.policy_net(torch.Tensor([next_state[0]]).to(self.device), torch.Tensor([next_state[1]]).to(self.device))
            else:
                next_output = self.policy_net(torch.Tensor([next_state]).to(self.device))
            
            if self.verbose: print("model output: ", next_output)
            next_output_Q, next_output_action = torch.max(next_output.data, 1)
            next_action = next_output_action[0].item()
            # expected_reward = reward + self.discount_factor * next_output_Q[0].item()

        # set 
        # self.policy_net.train()
        # if self.camera:
        #     output = self.policy_net(torch.Tensor([self.state[0]]).to(self.device), torch.Tensor([self.state[1]]).to(self.device))
        # else:
        #     output = self.policy_net(torch.Tensor([self.state]).to(self.device))
        # output_Q, output_action = torch.max(output.data, 1)
        # label = torch.Tensor()
        # label.data = output.clone()
        # label[0][self.action] = expected_reward
        # loss = self.criterion(output, label)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.losses.append(loss.item())
        # nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip)
        self.state = next_state
        self.action = next_action if self.epsilon < rand.random() else rand.randint(0, self.num_actions - 1)
        # self.epsilon *= self.epsilon_decay

        if self.verbose: print("s =", next_state,"a =",self.action,"reward =",reward)
        return self.action

    def run_dyna(self):
        """
        @summary: Runs dyna on saved samples
        @returns: The loss for the epoch
        """
        if len(self.samples) < self.replay_sample or not self.counter == 0:
            return
        s_list = []
        a_list = []
        s_prime_list = []
        r_list = []
        # if self.camera:
        s_state_list = []
        s_prime_state_list = []

        # probs = [np.exp(x[-1]) for x in self.samples]
        # total = sum(probs)
        # probs = [x / total for x in probs]
        # sample_indices = np.random.choice(np.arange(len(probs)), self.replay_sample, p=probs, replace=False).tolist()
        # curr_samples = [self.samples[i] for i in sample_indices]
        curr_samples = rand.sample(self.samples, self.replay_sample)
        for sample in curr_samples:
            [s, a, next_state, reward] = sample
            if self.camera:
                if type(s[0]) is int:
                    s_list.append(torch.zeros(3, 96, 172).float()) # same shape as video producer in grammar demo xml file
                    s_state_list.append(torch.Tensor(s).float())
                else:
                    s_list.append(torch.from_numpy(s[0].copy()).float()) 
                    s_state_list.append(torch.Tensor(s[1]).float()) 
                a_list.append(a)
                s_prime_list.append(next_state[0])
                r_list.append(reward)
                s_prime_state_list.append(next_state[1])
            else:
                s_list.append(s)
                a_list.append(a)
                s_prime_list.append(next_state)
                r_list.append(reward)
        # sample_set = {} Sorting for the samples commented out temporarily 
        # for sample in reversed(self.samples):
        #     sample_set[str(sample)] = sample
        # self.samples = list(sample_set.values())
        # self.samples.sort(key = lambda x: x[3])

        # self.policy_net.eval()
        # with torch.no_grad():
        #     if self.camera:
        #         next_output = self.policy_net(torch.Tensor(s_prime_list).to(self.device), torch.Tensor(s_prime_state_list).to(self.device))
        #     else:
        #         next_output = self.policy_net(torch.Tensor(s_prime_list).to(self.device))
        #     next_output_Q, next_output_action = torch.max(next_output.data, 1)
        #     expected_reward = torch.Tensor(r_list).to(self.device) + self.discount_factor * next_output_Q

        self.policy_net.train()
        self.target_net.eval()
        if self.camera:
            output = self.policy_net(torch.stack(s_list).to(self.device), torch.stack(s_state_list).to(self.device))
            next_output = self.target_net(torch.Tensor(s_prime_list).to(self.device), torch.Tensor(s_prime_state_list).to(self.device)) 
        else:
            output = self.policy_net(torch.Tensor(s_list).to(self.device))
            next_output = self.target_net(torch.Tensor(s_prime_list).to(self.device))

        # print(output.shape)
        # print(a_list)
        # print(torch.Tensor(a_list).shape)
        # output_Q, output_action = torch.max(output.data, 1)
        output_Q = torch.gather(output, 1, torch.Tensor(a_list).unsqueeze(-1).type(torch.LongTensor))

        next_output_Q, next_output_action = torch.max(next_output.data, 1)
        expected_reward = torch.Tensor(r_list).to(self.device) + self.discount_factor * next_output_Q
        # label = torch.Tensor()
        # label.data = output.clone()
        # for i in next_output_action:
        # next_output[:, torch.Tensor(a_list)] = expected_reward
        # print(output.shape)
        # print(expected_reward.shape)
        # print(next_output.shape)
        # print(output_Q.shape)
        loss = self.criterion(output_Q, expected_reward.unsqueeze(-1))
        # loss.requires_grad = True
        self.optimizer.zero_grad()
        loss.backward()
        # print("Before clip")
        # for p in self.policy_net.parameters():
        #     print(p.grad.norm())
        # print()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip)
        
        # print("After clip")
        # for p in self.policy_net.parameters():
        #     print(p.grad.norm())
        # print()
        self.optimizer.step()
        # print(loss.shape)
        self.losses.append(loss.item())
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            # target_param.data.copy_(0.001*local_param.data + 0.999*target_param.data)
            target_param.data.copy_(local_param.data)
        return loss.item()

    
    def query_dyna(self,next_state,reward):
        """
        UNUSED METHOD

        @summary: Does NOT update the Q table and returns an action
        @param next_state: The new state
        @param reward: The reward
        @returns: The selected action
        """
        if self.epsilon < rand.random():
            self.policy_net.eval()
            with torch.no_grad():
                if self.camera or [self.state, self.action, next_state, reward] not in self.samples:
                    self.samples.append([self.state, self.action, next_state, reward])
                while len(self.samples) > self.max_samples:
                    self.samples.pop()
                next_output = self.policy_net(torch.Tensor([next_state]).to(self.device))
                next_output_Q, next_output_action = torch.max(next_output.data, 1)
                self.action = next_output_action[0].item()
        else:
            self.action = rand.randint(0, self.num_actions - 1)
        self.state = next_state
        return self.action

    def clear_dyna(self):
        self.samples = []

    def save(self):
        currentTime = datetime.now().strftime("%m%d_%H%M%S")
        agentType = "CameraDQN" if self.camera else "DQN"
        save_path = f"cache/{agentType}_{currentTime}_model.pkl"
        
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path)
        print(f"Saved Model At {currentTime}")

    def plot_loss(self):
        # print(len(self.losses))
        # print(self.losses)
        plt.clf()
        prices_length = 10
        ravgs = [sum(self.losses[i:i+prices_length])/prices_length for i in range(len(self.losses)-prices_length+1)]
        currentTime = datetime.now().strftime("%m%d_%H%M")
        agentType = "CameraDQN" if self.camera else "DQN"

        plt.plot(ravgs)
        plt.xlabel("Iteration Number")
        plt.ylabel("Average Loss")
        plt.title("Average of loss across " + str(prices_length) + " iterations")
        if self.verbose: plt.show()
        plt.savefig(f"./graphs/{agentType}_{currentTime}_Avg_Loss_Graph.png")
        plt.clf()
        
        plt.plot(self.losses)
        plt.xlabel("Iteration Number")
        plt.ylabel("Loss")
        plt.title("Overall Loss")
        if self.verbose: plt.show()
        plt.savefig(f"./graphs/{agentType}_{currentTime}_Overall_Loss_Graph.png")
        plt.clf()

