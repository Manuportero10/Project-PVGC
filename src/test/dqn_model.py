from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import adam_v2 as Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from simpleRL_bot import SimpleRLPlayer
from heuristic_bot import HeuristicPlayer

heuristic_bot = HeuristicPlayer(battle_format="gen9randombattle", start_timer_on_battle_start=True) 
train_env = SimpleRLPlayer(battle_format="gen9randombattle", 
                           start_timer_on_battle_start=True,
                           start_challenging=True,
                           opponent=heuristic_bot
                           )

def main():
    # Create a model
    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape # (1,) is the batch size that the model expects in input.

    # Create model
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    dqn.fit(train_env, nb_steps=10000)
    train_env.close()

if __name__ == "__main__":
    main()