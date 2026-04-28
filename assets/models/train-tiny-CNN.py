import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# ------------------- Data Generators -------------------
def get_generator(directory, batch_size=32, target_size=(24, 24), augment=False):
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,      # small touches only
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False   # eyes are not symmetric left-right in this context
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
        directory,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

# Use your existing folder structure
train_gen = get_generator('data/train', batch_size=32, augment=True)   # mild aug on train
valid_gen = get_generator('data/val',   batch_size=32, augment=False)

steps_per_epoch = len(train_gen)
validation_steps = len(valid_gen)

print(f"Training steps: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")
print(f"Class mapping: {train_gen.class_indices}")   # ← Tell me this output!

# ------------------- Model (same tiny CNN) -------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=15,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1
)

# Save
os.makedirs('models', exist_ok=True)
model.save('models/cnnCat2.h5')
print("✅ Model trained and saved to models/cnnCat2.h5")



"""
this is the model results:
Epoch 1/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 996s 625ms/step - accuracy: 0.8937 - loss: 0.2612 - val_accuracy: 0.9635 - val_loss: 0.0952
Epoch 2/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 75s 47ms/step - accuracy: 0.9482 - loss: 0.1400 - val_accuracy: 0.9688 - val_loss: 0.0868
Epoch 3/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 68s 42ms/step - accuracy: 0.9599 - loss: 0.1097 - val_accuracy: 0.9780 - val_loss: 0.0582
Epoch 4/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 70s 44ms/step - accuracy: 0.9653 - loss: 0.0959 - val_accuracy: 0.9733 - val_loss: 0.0746
Epoch 5/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 68s 43ms/step - accuracy: 0.9685 - loss: 0.0878 - val_accuracy: 0.9805 - val_loss: 0.0534
Epoch 6/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 70s 44ms/step - accuracy: 0.9708 - loss: 0.0807 - val_accuracy: 0.9804 - val_loss: 0.0528
Epoch 7/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 69s 43ms/step - accuracy: 0.9732 - loss: 0.0752 - val_accuracy: 0.9847 - val_loss: 0.0425
Epoch 8/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 71s 45ms/step - accuracy: 0.9739 - loss: 0.0731 - val_accuracy: 0.9834 - val_loss: 0.0460
Epoch 9/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 69s 44ms/step - accuracy: 0.9744 - loss: 0.0702 - val_accuracy: 0.9843 - val_loss: 0.0460
Epoch 10/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 71s 45ms/step - accuracy: 0.9757 - loss: 0.0686 - val_accuracy: 0.9867 - val_loss: 0.0387
Epoch 11/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 68s 43ms/step - accuracy: 0.9768 - loss: 0.0665 - val_accuracy: 0.9816 - val_loss: 0.0496
Epoch 12/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 69s 43ms/step - accuracy: 0.9772 - loss: 0.0640 - val_accuracy: 0.9859 - val_loss: 0.0394
Epoch 13/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 70s 44ms/step - accuracy: 0.9775 - loss: 0.0636 - val_accuracy: 0.9863 - val_loss: 0.0380
Epoch 14/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 69s 43ms/step - accuracy: 0.9784 - loss: 0.0598 - val_accuracy: 0.9875 - val_loss: 0.0365
Epoch 15/15
1592/1592 ━━━━━━━━━━━━━━━━━━━━ 72s 45ms/step - accuracy: 0.9790 - loss: 0.0580 - val_accuracy: 0.9870 - val_loss: 0.0362
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
✅ Model trained and saved to models/cnnCat2.h5
"""