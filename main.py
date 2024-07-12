import argparse
import gym
from argument import dqn_arguments, pg_arguments


def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    parser.add_argument('--train_dqn', default=True, type=bool, help='whether train DQN')

    parser = dqn_arguments(parser)
    # parser = pg_arguments(parser)
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        return agent.run()

    if args.train_dqn:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        return agent.run()

from plot import plot_vectors
if __name__ == '__main__':
    args = parse()
    arr = []
    for i in range(3):
        print(f"--------第{i}轮训练开始---------")
        args.seed += i
        a = run(args)
        arr.append(a)
    print(arr)
    plot_vectors(arr[0], arr[1], arr[2], args.seed - 2, args.seed - 1, args.seed)
    # run(args)
