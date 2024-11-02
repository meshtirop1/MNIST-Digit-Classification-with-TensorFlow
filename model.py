import tensorflow as tf
import hy_param

# Input placeholders using tf.keras.Input
x = tf.keras.Input(shape=(hy_param.num_input,), name="input_x")
y = tf.keras.Input(shape=(hy_param.num_classes,), name="input_y")

# Model definition
input_layer = x
dense_layer_1 = tf.keras.layers.Dense(hy_param.num_hidden_1, activation='relu')(input_layer)
dense_layer_2 = tf.keras.layers.Dense(hy_param.num_hidden_2, activation='relu')(dense_layer_1)
output_layer = tf.keras.layers.Dense(hy_param.num_classes, activation='softmax')(dense_layer_2)

# Defining the model
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model with loss and optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hy_param.learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Use model.summary() to see the architecture
model.summary()
