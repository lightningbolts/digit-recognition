import tensorflow as tf

# Load the model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train)
y_train = tf.keras.utils.to_categorical(y_train)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

y_test = tf.keras.utils.to_categorical(y_test)

loss, accuracy = model.evaluate(x_test, y_test)

print(f'Loss: {loss}, Accuracy: {accuracy}')

model.save('handwrittendigit.keras')