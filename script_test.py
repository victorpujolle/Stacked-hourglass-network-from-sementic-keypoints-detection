"""

"""

import tensorflow as tf




def _conv(inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv'):
    """ Spatial Convolution (CONV2D)
    Args:
        inputs			: Input Tensor (Data Type : NHWC)
        filters		: Number of filters (channels)
        kernel_size	: Size of kernel
        strides		: Stride
        pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
        name			: Name of the block
    Returns:
        conv			: Output Tensor (Convolved Input)
    """
    with tf.name_scope(name):
        # Kernel for convolution, Xavier Initialisation
        kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
            [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
        conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')

        return conv


lay0 = tf.placeholder(dtype=tf.float32, shape=[None,256,256,1])

lay1 = _conv(
    inputs=lay0,
    filters=64
)

print(lay1.shape)

