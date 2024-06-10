import gym , warnings
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from statistics import median, mean
from collections import Counter
from tqdm import tqdm
import pickle

LR = 1e-3
#env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
env.reset()



goal_steps = 5000
score_requirement = 100
initial_games = 100

# Suprimir avisos de DeprecationWarning
#warnings.filterwarnings("ignore", category=DeprecationWarning)

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(200):
         
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            if done:
                break
                
#some_random_games_first()


def initial_population():
    model = tf.keras.models.load_model('deucerto.keras')
    training_data = []
    scores = []
    accepted_scores = []
    
    for _ in tqdm(range(initial_games)):
        score = 0
        game_memory = []
        prev_observation = []
        
        for _ in range(goal_steps):
            if len(prev_observation) == 0:
                action = random.randrange(0, 2)
            else:
                action = np.argmax(model.predict(prev_observation.reshape(-1, len(prev_observation), 1), verbose=0)[0])
            
            observation, reward, done, truncated, info = env.step(action)
            
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done or truncated: break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                else:
                    output = [1, 0]
                training_data.append([data[0], output])
                
        env.reset()
        scores.append(score)
        
    with open('train_data.pkl', 'wb') as file:
            pickle.dump(training_data, file)
    # training_data_save = np.array(training_data)
    # np.save('saved.npy', training_data_save)
    
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data

def neural_network_model(input_size):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(input_size, 1)))
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LR), metrics=['accuracy'])

    
    return model

def train_model(training_data, model=None):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = np.array([i[1] for i in training_data])
    
    if model is None:
        model = neural_network_model(input_size=len(X[0]))
    
    model.fit(X, y, epochs=20, batch_size=64, verbose=1)
    return model

training_data = initial_population()
model = train_model(training_data)






# import gym
# from gym.envs.classic_control import CartPoleEnv , utils
# from gym.utils import seeding


# class CustomCartPoleEnv(CartPoleEnv):
#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
        
#         if options is not None:
#             low = options.get('low', [-0.05, -0.05, -0.05, -0.05])
#             high = options.get('high', [0.05, 0.05, 0.05, 0.05])
#         else:
#             low, high = [-0.05, -0.05, -0.05, -0.05], [0.05, 0.05, 0.05, 0.05]
        
#         self.state = self.np_random.uniform(low=low, high=high, size=(4,))
#         self.steps_beyond_terminated = None

#         if self.render_mode == "human":
#             self.render()
#         return np.array(self.state, dtype=np.float32), {}

# # Uso do ambiente customizado
# env = CustomCartPoleEnv(render_mode="human")
# options = {
#     'low': [-2, -1, -0.2, -2],
#     'high': [2, 1, 0.2, 2]
# }
# obs, info = env.reset(options=options)  # Limites customizados para cada estado





# Carregar o modelo
#model = tf.keras.models.load_model('deucerto.keras')


#working jsut fine up to here

env.close()
env = gym.make("CartPole-v1", render_mode="human")
env.reset()


scores = []
choices = []
for each_game in range(5):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    
    for _ in range(goal_steps):
        

        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1), verbose=0)[0])

        choices.append(action)
                
        new_observation, reward, done, truncated, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        
        if done : break

    scores.append(score)

print('Average Score:', sum(scores) / len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
print(score_requirement)




    
















