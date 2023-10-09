from collections import defaultdict
import numpy as np
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting

class Sarsa(AbstractSolver):
    def __init__(self, env, eval_env, options):
        assert str(env.observation_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete state spaces"
        )
        assert str(env.action_space).startswith("Discrete") or str(
            env.action_space
        ).startswith("Tuple(Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def train_episode(self):
        """
        Run one episode of the SARSA algorithm: On-policy TD control.

        Use:
            self.env: OpenAI environment.
            self.epsilon_greedy_action(state): returns an epsilon greedy action
            self.options.steps: number of steps per episode
            self.options.gamma: Gamma discount factor.
            self.options.alpha: TD learning rate.
            self.Q[state][action]: q value for ('state', 'action')
            self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.

        """

        # Reset the environment
        state, _ = self.env.reset()
        #init for the first action
        action = np.argmax(self.epsilon_greedy_action(state))

        for _ in range(self.options.steps):
            next_state, reward, done, _, = self.step(action)
            next_action = np.argmax(self.epsilon_greedy_action(next_state))
            # SARSA update rule
            target = reward + self.options.gamma * self.Q[next_state][next_action]
            self.Q[state][action] += self.options.alpha * (target - self.Q[state][action])
            state = next_state
            action = next_action

            if done:
                break

    def __str__(self):
        return "Sarsa"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.

        Returns:
            A function that takes a state as input and returns a greedy action.
        """

        def policy_fn(state):
            return np.argmax(self.Q[state])

        return policy_fn

    def epsilon_greedy_action(self, state):
        """
        Return an epsilon-greedy action based on the current Q-values and
        epsilon.

        Use:
            self.env.action_space.n: the size of the action space
            np.argmax(self.Q[state]): action with the highest q value
        Returns:
            The selected action.
        """
        action_probs = np.ones(self.env.action_space.n) * self.options.epsilon / self.env.action_space.n
        action_probs[np.argmax(self.Q[state])] = 1 - self.options.epsilon + self.options.epsilon / self.env.action_space.n
        return action_probs
    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)
