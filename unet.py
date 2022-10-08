import tensorflow as tf 
from tensorflow_examples.models.pix2pix import pix2pix

base_model = tf.keras.applications.MobileNetV2(
    input_shape=[128,128,3],
    include_top=False
)

layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu',
    'block_16_project'
]

layers = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input,outputs=layers)
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512,3),
    pix2pix.upsample(256,3),
    pix2pix.upsample(128,3),
    pix2pix.upsample(64,3)
]

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128,128,3])
    x = inputs

    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack,skips):

        x = up(x)

        concat = tf.keras.layers.Concatenate()

        x = concat([x,skip])
    
    last = tf.keras.layers.Conv2DTranspose(
        output_channels,3,strides=2,padding='same'
    )

    x = last(x)

    return tf.keras.Model(inputs=inputs,outputs=x)
    


