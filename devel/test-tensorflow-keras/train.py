import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the training data
conversations = [
    ("Hi", "Hello! How can I help you?"),
    ("What's your name?", "My name is Chatbot."),
    ("How are you?", "I'm fine, thank you!"),
    ("Goodbye", "Goodbye! Have a nice day!")
]

# Separate questions and answers
questions = [conv[0] for conv in conversations]
answers = [conv[1] for conv in conversations]

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
vocab_size = len(tokenizer.word_index) + 1

# Convert text data to sequences
question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

# Pad sequences to make them equal length
max_sequence_length = max(len(seq) for seq in question_sequences + answer_sequences)
padded_question_sequences = pad_sequences(question_sequences, maxlen=max_sequence_length, padding='post')
padded_answer_sequences = pad_sequences(answer_sequences, maxlen=max_sequence_length, padding='post')

# Define the model architecture
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 100, input_length=max_sequence_length),
    keras.layers.Bidirectional(keras.layers.LSTM(256)),
    keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_question_sequences, padded_answer_sequences, epochs=100)

# Save the tokenizer and model
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as json_file:
    json_file.write(tokenizer_json)

model.save("chatbot_model.h5")
