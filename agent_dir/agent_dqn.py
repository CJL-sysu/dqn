import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent
from torch.nn import Sequential, Linear, ReLU


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        ##################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = Sequential(
            Linear(self.input_size, self.hidden_size),
            ReLU(),
            Linear(self.hidden_size, self.output_size)
        )

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        ##################
        return self.model(inputs)


class ReplayBuffer:
    class Data:
        def __init__(self, buffer_size):
            self.observation = np.zeros((buffer_size, 4))
            self.action = np.zeros(buffer_size)
            self.reward = np.zeros(buffer_size)
            self.observation_ = np.zeros((buffer_size, 4))
    def __init__(self, buffer_size):
        ##################
        # YOUR CODE HERE #
        ##################
        self.data = self.Data(buffer_size)
        self.buffer_size = buffer_size
        self.idx = 0

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        ##################
        if self.idx >= self.buffer_size:
            return self.buffer_size
        else:
            return self.idx

    def push(self, *transition):
        ##################
        # YOUR CODE HERE #
        ##################
        observation, action, reward, observation_ = transition

        idx = self.idx % self.buffer_size
        self.data.observation[idx] = observation
        self.data.action[idx] = action
        self.data.reward[idx] = reward
        self.data.observation_[idx] = observation_
        self.idx += 1

    def sample(self, batch_size):
        ##################
        # YOUR CODE HERE #
        ##################
        if self.__len__() == 0:
            return None
        if self.__len__() < batch_size:
            random_indices = np.random.choice(
                self.__len__(), batch_size, replace=True
            )  # 可能选到重复的
        else:
            random_indices = np.random.choice(
                self.__len__(), batch_size, replace=False
            )  # 不会选到重复的

        observation = self.data.observation[random_indices]
        action = self.data.action[random_indices]
        reward = self.data.reward[random_indices]
        observation_ = self.data.observation_[random_indices]
        return observation, action, reward, observation_

    def clean(self):
        ##################
        # YOUR CODE HERE #
        ##################
        self.idx = 0


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        ##################
        # parse args
        np.random.seed(args.seed)
        self.args = args
        self.env = env
        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.test = args.test
        self.lr = args.lr
        self.gamma = args.gamma
        self.n_frames = args.n_frames
        self.render = args.render
        self.target_update_freq = args.target_update_freq
        self.min_train_size = args.min_train_size
        self.epsilon = args.epsilon
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.batch_size = args.batch_size
        self.writer = SummaryWriter(log_dir=args.log_dir)
        if not os.path.exists(args.log_dir):
            print(f"path {args.log_dir} not exists, now touch it")
            os.makedirs(args.log_dir)
        # Prepare memory
        self.memory = ReplayBuffer(args.memory_size)
        # build net
        self.q_net = QNetwork(
            self.state_dim,
            self.args.hidden_size,
            self.action_dim,
        ).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss().to(self.device)
        self.update_count = 0

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        sample = self.memory.sample(self.batch_size)
        observation, action, reward, observation_ = sample
        # ToTensor
        observation = torch.tensor(observation, dtype = torch.float).to(self.device)
        action = (
            torch.tensor(action, dtype=torch.float).view(-1, 1).to(self.device)
        )  # .view(-1, 1)将action转列向量
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(self.device)
        observation_ = torch.tensor(observation_, dtype = torch.float).to(self.device)
        # train
        q_values = self.q_net(observation)
        target_q_values = self.target_net(observation_).max(1)[0].view(-1, 1)
        q_targets = reward + self.gamma * target_q_values
        loss = self.loss_fn(q_values, q_targets)
        loss_avg = torch.mean(loss)
        #print(f"avg loss = {loss_avg}")
        self.writer.add_scalar("loss", loss_avg, self.update_count)  # tensorboardX
        self.optimizer.zero_grad()  
        loss.backward()  # 反向传播更新参数
        self.optimizer.step()
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.update_count += 1

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        ##################
        # YOUR CODE HERE #
        ##################
        x, x_dot, theta, theta_dot = observation
        if not test and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = np.hstack([x, x_dot, theta, theta_dot])
            state = torch.tensor(state, dtype = torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        for episode in range(self.args.n_frames):
            observation = self.env.reset()
            total_reward = 0
            while True:
                if self.render:
                    self.env.render()
                action = self.make_action(observation, self.test)
                observation_, reward, done, info = self.env.step(action)
                total_reward += reward
                # x, x_dot, theta, theta_dot = observation_
                self.memory.push(
                    observation,
                    action,
                    reward,
                    observation_,
                )
                if len(self.memory) > self.min_train_size:
                    self.train()
                if done:
                    print(f"episode {episode} total reward = {total_reward}")
                    self.writer.add_scalar("total_reward", total_reward, episode)
                    break
