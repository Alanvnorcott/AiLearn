import random
import tensorflow as tf
import numpy as np
import time

fruit_to_color = {
    "apple": "red",
    "banana": "yellow",
    "orange": "orange",
    "blueberries": "blue",
    "grapes": "green"
}


fruits = list(fruit_to_color.keys())
colors = list(set(fruit_to_color.values()))
num_fruits = len(fruits)
num_colors = len(colors)

fruit_to_index = {fruit: i for i, fruit in enumerate(fruits)}
color_to_index = {color: i for i, color in enumerate(colors)}


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_fruits, 10),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_colors, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Generate training data
X_train = np.array([fruit_to_index[fruit] for fruit in fruits])
y_train = np.array([color_to_index[fruit_to_color[fruit]] for fruit in fruits])

# Train the neural network
model.fit(X_train, y_train, epochs=50)

# Q-learning agent
Q = np.zeros((len(fruits), len(colors)))
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1

# Game part of sim
options = ["orange", "yellow", "red", "green", "blue"]
questions = ["apple", "banana", "orange", "blueberries", "grapes"]

for episode in range(1000):
    score = 0
    random.shuffle(questions)

    for fruit in questions:
        fruit_index = fruit_to_index[fruit]
        fruit_input = np.array([fruit_index])

        # Choose action using epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, len(colors) - 1)  # Explore
        else:
            action_index = np.argmax(Q[fruit_index])  # Exploit

        predicted_color = colors[action_index]

        # Automatically input the predicted color
        user_answer = predicted_color.lower()

        if user_answer == fruit_to_color[fruit]:
            reward = 1
            score += 1
        else:
            reward = -1
            score -= 1

        # Update Q-table using Q-learning update rule
        Q[fruit_index, action_index] += learning_rate * (
                    reward + discount_factor * np.max(Q[fruit_index]) - Q[fruit_index, action_index])

    print(f"Episode {episode + 1} - Score: {score}")

# letting the AI play
while True:
    score = 0
    random.shuffle(questions)

    for fruit in questions:
        fruit_index = fruit_to_index[fruit]
        action_index = np.argmax(Q[fruit_index])
        predicted_color = colors[action_index]

        print(f"What color is {fruit}?")
        print("Options:")
        print(", ".join(options))

        # Automatically input the predicted color
        user_answer = predicted_color.lower()

        if user_answer == fruit_to_color[fruit]:
            print("Correct!")
            score += 1
        else:
            print(f"Sorry, the correct color for {fruit} is {fruit_to_color[fruit]}.")

        time.sleep(2)

    print(f"\nGame Over! Your final score is: {score}/{len(questions)}")
    play_again = input("Do you want to play again? (yes/no): ").lower()
    if play_again != "yes":
        break
