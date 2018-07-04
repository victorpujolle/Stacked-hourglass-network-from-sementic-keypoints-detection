import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os
from flip_gradient import flip_gradient


class HourglassModel:

    """
    Hourglass model created for keypoint detection
    """

    def __init__(self, nFeat=256, nStack=2, nModules=1, nLow=6, outputDim=8, batch_size=4, drop_rate=0.2,
                 lear_rate=2.5e-4, decay=0.96, decay_step=2000, dataset=None, training=True, w_summary=True,
                 logdir_train=None, logdir_test=None, w_loss=False,
                 name='hourglass', joints=['fru','frd','flu','fld','bru','brd','blu','bld']):

        """ Initializer
         nStack              : number of stacks (stage/Hourglass modules)
         nFeat               : number of feature channels on conv layers
         nLow                : number of downsampling (pooling) per module
         outputDim           : number of output Dimension (16 for MPII)
         batch_size          : size of training/testing Batch
         dro_rate            : Rate of neurons disabling for Dropout Layers
         lear_rate           : Learning Rate starting value
         decay               : Learning Rate Exponential Decay (decay in ]0,1], 1 for constant learning rate)
         decay_step          : Step to apply decay
         dataset             : Dataset (class DataGenerator)
         training            : (bool) True for training / False for prediction
         w_summary           : (bool) True/False for summary of weight (to visualize in Tensorboard)
         name                : name of the model
         """

        self.nStack = nStack
        self.nFeat = nFeat
        self.nModules = nModules
        self.outDim = outputDim
        self.batchSize = batch_size
        self.training = training
        self.w_summary = w_summary
        self.dropout_rate = drop_rate
        self.learning_rate = lear_rate
        self.decay = decay
        self.name = name
        self.decay_step = decay_step
        self.nLow = nLow
        self.dataset = dataset
        self.cpu = '/cpu:0'
        self.gpu = '/gpu:0'
        self.logdir_train = logdir_train
        self.logdir_test = logdir_test
        self.joints = joints


    def generate_model(self):
        """
        Create the complete graph
        """

        startTime = time.time()

        with tf.name_scope('input'):

            # placefolder of the input image : shape = (?, 256, 256, 256)
            self.img = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img')

            # ground truth placeholder : shape = (?, nStacks, 64, 64, outDim)
            self.gtMaps = tf.placeholder(dtype=tf.float32, shape=(None, self.nStack, 64, 64, self.outDim))

            # ground truth of the domain : shape = (?)
            self.gtDomain = tf.placeholder(dtype=tf.float32, shape=(None))

        inputTime = time.time()
        print('---Inputs : Done (' + str(int(abs(inputTime - startTime))) + ' sec.)')

        with tf.name_scope('graph'):
        # output and output_domain
            self.output, self.domain = self._graph_hourglass(self.img)

        graphTime = time.time()
        print('---Graph : Done (' + str(int(abs(graphTime - inputTime))) + ' sec.)')

        with tf.name_scope('loss'):

            hm_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.output,
                labels=self.gtMaps
            )
            domain_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.domain,
                labels=self.gtDomain
            )

            # gamma implemantation
            #gamma = 0.1
            #hm_loss = (hm_loss - (1 - gamma) * (1 - tf.reshape(self.gtDomain, [4, 1, 1, 1, 1])) * hm_loss) * (1 + gamma) / (2 * gamma)
            #domain_loss = (domain_loss - (1 - gamma) * (1 - self.gtDomain) * domain_loss) * (1 + gamma) / (2 * gamma)

            self.loss = tf.reduce_mean(hm_loss, name='cross_entropy_heatmap_loss') + tf.reduce_mean(domain_loss, name='cross_entropy_domain_loss')

        lossTime = time.time()
        print('---Loss : Done (' + str(int(abs(graphTime - lossTime))) + ' sec.)')


        with tf.name_scope('accuracy'):
            self._accuracy_computation()

        accurTime = time.time()
        print('---Acc : Done (' + str(int(abs(accurTime - lossTime))) + ' sec.)')

        with tf.name_scope('steps'):
            self.train_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.name_scope('lear_rate'):
            self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay, staircase=True, name='learning_rate')

        lrTime = time.time()
        print('---LR : Done (' + str(int(abs(accurTime - lrTime))) + ' sec.)')


        with tf.name_scope('rmsprop'):
        # the choice of the optimizer
            self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)


        optimTime =time.time()
        print('---Optim : Done (' + str(int(abs(optimTime - lrTime))) + ' sec.)')

        with tf.name_scope('minimizer'):
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)

        minimTime = time.time()
        print('---Minimizer : Done (' + str(int(abs(optimTime - minimTime))) + ' sec.)')

        # initialisation of the variables
        self.init = tf.global_variables_initializer()
        initTime = time.time()
        print('---Init : Done (' + str(int(abs(initTime - minimTime))) + ' sec.)')


        with tf.name_scope('training'):
            tf.summary.scalar('loss', self.loss, collections=['train'])
            tf.summary.scalar('learning_rate', self.lr, collections=['train'])

        with tf.name_scope('summary'):
            for i in range(len(self.joints)):
                tf.summary.scalar(self.joints[i], self.joint_accur[i], collections=['train', 'test'])


        self.train_op = tf.summary.merge_all('train')
        self.test_op = tf.summary.merge_all('test')
        self.weight_op = tf.summary.merge_all('weight')

        endTime = time.time()
        print('Model created (' + str(int(abs(endTime - startTime))) + ' sec.)')
        del endTime, startTime, initTime, optimTime, minimTime, lrTime, accurTime, lossTime, graphTime, inputTime


    def restore(self, load=None):
        """ Restore a pretrained model
        Model to load (None if training from scratch) (see README for further information)
        """
        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                self._init_session()
                self._define_saver_summary(summary=False)

                if load is not None:
                    print('Loading Trained Model')
                    t = time.time()

                    self.saver.restore(self.Session, load)

                    print('Model Loaded (', time.time() - t, ' sec.)')

                else:
                    print('Please give a Model in args (see README for further information)')


    def _init_session(self):
        """ Initialize Session
        """
        print('Session initialization')
        t_start = time.time()
        self.Session = tf.Session()
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')


    def _train(self, nEpochs=10, epochSize=1000, saveStep=500, validIter=10):
        """
        training function (todo)
        """

        with tf.name_scope('train'):
            self.generator = self.dataset._aux_generator(self.batchSize, self.nStack, normalize=True, sample_set='train')
            self.valid_gen = self.dataset._aux_generator(self.batchSize, self.nStack, normalize=True, sample_set='valid')

            startTime = time.time()
            self.resume = {}

            self.resume['accur'] = []
            self.resume['loss'] = []
            self.resume['err'] = []

            for epoch in range(nEpochs):

                epochstartTime = time.time()
                avg_cost = 0.
                cost = 0.
                print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')

                # Training Set
                for i in range(epochSize):

                    # DISPLAY PROGRESS BAR
                    percent = ((i + 1) / epochSize) * 100
                    num = np.int(20 * percent / 100)
                    tToEpoch = int((time.time() - epochstartTime) * (100 - percent) / (percent))
                    sys.stdout.write('\r Train: {0}>'.format("=" * num) + "{0}>".format(" " * (20 - num)) + ' || ' + str(epoch) + '/' + str(nEpochs) + ' Epochs || ' + str(percent)[:4] + '%' + ' -cost: ' + str(cost)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
                    sys.stdout.flush()

                    img_train, gt_train, weight_train, gt_domain = next(self.generator)

                    if i % saveStep == 0:
                        _, c, summary = self.Session.run(
                            [self.train_rmsprop, self.loss, self.train_op],
                            feed_dict={self.img: img_train, self.gtMaps: gt_train,
                            self.gtDomain: gt_domain}
                        )

                        # Save summary (Loss + Accuracy)
                        self.train_summary.add_summary(summary, epoch * epochSize + i)
                        self.train_summary.flush()

                    else:
                        _, c, = self.Session.run(
                            [self.train_rmsprop, self.loss],
                            feed_dict={self.img: img_train, self.gtMaps: gt_train,
                            self.gtDomain: gt_domain}
                        )

                    cost += c
                    avg_cost += c / epochSize

                epochfinishTime = time.time()

                weight_summary = self.Session.run(
                    self.weight_op,
                    {self.img: img_train, self.gtMaps: gt_train, self.gtDomain: gt_domain}
                )

                self.train_summary.add_summary(weight_summary, epoch)
                self.train_summary.flush()

                print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(((epochfinishTime - epochstartTime) / epochSize))[:4] + ' sec.')

                with tf.name_scope('save'):
                    self.saver.save(self.Session, os.path.join(os.getcwd(), str(self.name + '_' + str(epoch + 1))))

                self.resume['loss'].append(cost)

                # Validation Set
                accuracy_array = np.array([0.0] * len(self.joint_accur))

                for i in range(validIter):
                    img_valid, gt_valid, w_valid, gt_domain_valid = next(self.valid_gen)

                    accuracy_pred = self.Session.run(
                        self.joint_accur,
                        feed_dict={self.img: img_valid, self.gtMaps: gt_valid,self.gtDomain: gt_domain_valid}
                    )

                    accuracy_array += np.array(accuracy_pred, dtype=np.float32) / validIter

                print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%')

                self.resume['accur'].append(accuracy_pred)
                self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
                valid_summary = self.Session.run(
                    self.test_op,
                    feed_dict={self.img: img_valid, self.gtMaps: gt_valid, self.gtDomain: gt_domain_valid})
                self.test_summary.add_summary(valid_summary, epoch)
                self.test_summary.flush()

            print('Training Done')
            print('Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(nEpochs * epochSize * self.batchSize))
            print('  Final Loss: ' + str(cost) + '\n' + '  Relative Loss: ' + str(100 * self.resume['loss'][-1] / (self.resume['loss'][0] + 0.1)) + '%')
            print('  Relative Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) + '%')
            print('  Training Time: ' + str(datetime.timedelta(seconds=time.time() - startTime)))


    def _accuracy_computation(self):
        """Compute the accuracy tensor
        """
        self.joint_accur = []
        for i in range(len(self.joints)):
            self.joint_accur.append(self._accur(
                self.output[:, self.nStack - 1, :, :, i],
                self.gtMaps[:, self.nStack - 1, :, :, i],
                self.batchSize)
            )


    def _define_saver_summary(self, summary=True):
        """ Create Summary and Saver
        Args:
            logdir_train        : Path to train summary directory
            logdir_test        : Path to test summary directory
        """
        if (self.logdir_train == None) or (self.logdir_test == None):
            raise ValueError('Train/Test directory not assigned')
        else:
            with tf.device(self.cpu):
                self.saver = tf.train.Saver()
            if summary:
                with tf.device(self.gpu):
                    self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
                    self.test_summary = tf.summary.FileWriter(self.logdir_test)
                    # self.weight_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())


    def _init_weight(self):
        """ Initialize weights
        """
        print('Session initialization')
        self.Session = tf.Session()
        t_start = time.time()
        self.Session.run(self.init)
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')


    def _accur(self, pred, gtMap, num_image):
        """ Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
            returns one minus the mean distance.
            Args:
                pred        : Prediction Batch (shape = num_image x 64 x 64)
                gtMaps        : Ground Truth Batch (shape = num_image x 64 x 64)
                num_image     : (int) Number of images in batch
            Returns:
                (float)
        """
        err = tf.to_float(0)
        for i in range(num_image):
            err = tf.add(err, self._compute_err(pred[i], gtMap[i]))
        return tf.subtract(tf.to_float(1), err / num_image)


    def _compute_err(self, u, v):
        """ Given 2 tensors compute the euclidean distance (L2) between maxima locations
        Args:
            u        : 2D - Tensor (Height x Width : 64x64 )
            v        : 2D - Tensor (Height x Width : 64x64 )
        Returns:
            (float) : Distance (in [0,1])
        """
        u_x, u_y = self._argmax(u)
        v_x, v_y = self._argmax(v)
        return tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y)))


    def _argmax(self, tensor):
        """ ArgMax
        Args:
            tensor    : 2D - Tensor (Height x Width : 64x64 )
        Returns:
            arg        : Tuple of max position
        """
        resh = tf.reshape(tensor, [-1])
        argmax = tf.argmax(resh, 0)
        return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])


    def _graph_hourglass(self, inputs):
        """Create the Network
            Args:
                inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
        """
        with tf.name_scope('model'):
            with tf.name_scope('preprocessing'):
                # Input Dim : nbImages x 256 x 256 x 3
                pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad_1')

                # Dim pad1 : nbImages x 260 x 260 x 3
                conv1 = self._conv_bn_relu(pad1, filters=64, kernel_size=6, strides=2, name='conv_256_to_128')

                # Dim conv1 : nbImages x 128 x 128 x 64
                r1 = self._residual(conv1, numOut=128, name='r1')

                # Dim pad1 : nbImages x 128 x 128 x 128
                pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')

                # Dim pool1 : nbImages x 64 x 64 x 128
                r2 = self._residual(pool1, numOut=int(self.nFeat / 2), name='r2')
                r3 = self._residual(r2, numOut=self.nFeat, name='r3')

            # Storage Table
            hg = [None] * self.nStack   # hourglass net
            ll = [None] * self.nStack   # convolutional net
            ll_ = [None] * self.nStack  # convolutional net
            drop = [None] * self.nStack # dropout layer
            out = [None] * self.nStack  # output layer
            out_ = [None] * self.nStack # output layer
            sum_ = [None] * self.nStack # summation of out_, ll and sum[i-1]
            domain = [None] * self.nStack

            with tf.name_scope('stacks'):
                with tf.name_scope('stack_0'):
                # initialization of the first stack

                    hg[0] = self._hourglass(r3, self.nLow, self.nFeat, name='hourglass')
                    drop[0] = tf.layers.dropout(hg[0], rate=self.dropout_rate, training=self.training, name='dropout')

                    ll[0] = self._conv_bn_relu(drop[0], filters=self.nFeat, kernel_size=1, strides=1, pad='VALID', name='conv')
                    ll_[0] = self._conv(ll[0], filters=self.nFeat, kernel_size=1, strides=1, pad='VALID', name='ll')

                    out[0] = self._conv(ll[0], filters=self.outDim, kernel_size=1, strides=1, pad='VALID', name='out')
                    domain[0] = self._conv(ll[0], 1, ll[0].get_shape().as_list()[1], 1, 'VALID','out')

                    out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')
                    sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')

                    for i in range(1,self.nStack - 1):
                        with tf.name_scope('stack_' + str(i)):
                        # construction of all the stacks

                            hg[i] = self._hourglass(sum_[i-1], self.nLow, self.nFeat, name='hourglass')
                            drop[i] = tf.layers.dropout(hg[i], rate=self.dropout_rate, training=self.training, name='dropout')

                            ll[i] = self._conv_bn_relu(drop[i], filters=self.nFeat, kernel_size=1, strides=1, pad='VALID', name='conv')
                            ll_[i] = self._conv(ll[i], filters=self.nFeat, kernel_size=1, strides=1, pad='VALID', name='ll')

                            out[i] = self._conv(ll[i], filters=self.outDim, kernel_size=1, strides=1, pad='VALID', name='out')
                            domain[i] = self._conv(ll[i], 1, ll[i].get_shape().as_list()[1], 1, 'VALID', 'out')

                            out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
                            sum_[i] = tf.add_n([out_[i], sum[i - 1], ll_[0]], name='merge')

                with tf.name_scope('stack_' + str(self.nStack - 1)):
                # last stack

                    hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
                    drop[self.nStack - 1] = tf.layers.dropout(hg[self.nStack - 1], rate=self.dropout_rate, training=self.training, name='dropout')

                    ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack - 1], self.nFeat, 1, 1, 'VALID', 'conv')

                    out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim, 1, 1, 'VALID', 'out')
                    domain[self.nStack - 1] = self._conv(ll[self.nStack - 1], 1, ll[self.nStack - 1].get_shape().as_list()[1], 1, 'VALID', 'out')

            with tf.name_scope('domain_classifier'):
            # domain classifier
                stack_out = tf.layers.flatten(tf.stack(out, axis=1))
                flipped = flip_gradient(stack_out)

                dense = tf.layers.dense(
                    inputs=flipped,
                    units=512
                )

                dropout = tf.layers.dropout(
                    inputs=dense,
                    rate=0.4
                )

                domain_logits= tf.nn.sigmoid(tf.contrib.layers.fully_connected(
                    inputs=dropout,
                    num_outputs=1,
                ))

            # return of the heatmap and the domain
            #return tf.stack(out, axis=1, name='final_output'), tf.contrib.layers.fully_connected(flip_gradient(tf.stack(domain)), num_outputs=1)
            return tf.stack(out, axis=1, name='final_output'), domain_logits


    def record_training(self, record):
        """ Record Training Data and Export them in CSV file
        Args:
            record        : record dictionnary
        """
        out_file = open(self.name + '_train_record.csv', 'w')
        for line in range(len(record['accur'])):
            out_string = ''
            labels = [record['loss'][line]] + [record['err'][line]] + record['accur'][line]
            for label in labels:
                out_string += str(label) + ', '
            out_string += '\n'
            out_file.write(out_string)
        out_file.close()
        print('Training Record Saved')


    def training_init(self, nEpochs=10, epochSize=1000, saveStep=500, dataset=None, load=None):
        """ Initialize the training
        Args:
            nEpochs        : Number of Epochs to train
            epochSize        : Size of one Epoch
            saveStep        : Step to save 'train' summary (has to be lower than epochSize)
            dataset        : Data Generator (see generator.py)
            load            : Model to load (None if training from scratch) (see README for further information)
        """
        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                self._init_weight()
                self._define_saver_summary()
                if load is not None:
                    self.saver.restore(self.Session, load)
                self._train(nEpochs, epochSize, saveStep, validIter=10)


    def _hourglass(self, inputs, n, numOut, name='hourglass'):
        """ Hourglass Module
        Args:
            inputs    : Input Tensor
            n        : Number of downsampling step
            numOut    : Number of Output Features (channels)
            name    : Name of the block
        """
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')

            return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')


    def _conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu'):
        """ Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
        Args:
            inputs            : Input Tensor (Data Type : NHWC)
            filters        : Number of filters (channels)
            kernel_size    : Size of kernel
            strides        : Stride
            pad                : Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name            : Name of the block
        Returns:
            norm            : Output Tensor
        """
        with tf.name_scope(name):
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding='VALID', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=self.training)
            if self.w_summary:
                with tf.device(self.cpu):
                    tf.summary.histogram('weights_summary', kernel, collections=['weight'])
            return norm


    def _residual(self, inputs, numOut, name='residual_block'):
        """ Residual Unit
        Args:
            inputs    : Input Tensor
            numOut    : Number of Output Features (channels)
            name    : Name of the block
        """
        with tf.name_scope(name):
            convb = self._conv_block(inputs, numOut)
            skipl = self._skip_layer(inputs, numOut)

            return tf.add_n([convb, skipl], name='res_block')


    def _conv_block(self, inputs, numOut, name='conv_block'):
        """ Convolutional Block
        Args:
            inputs    : Input Tensor
            numOut    : Desired output number of channel
            name    : Name of the block
        Returns:
            conv_3    : Output Tensor
        """
        with tf.name_scope(name):
            with tf.name_scope('norm_1'):
                norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,is_training=self.training)
                conv_1 = self._conv(norm_1, int(numOut / 2), kernel_size=1, strides=1, pad='VALID', name='conv')

            with tf.name_scope('norm_2'):
                norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,is_training=self.training)
                pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                conv_2 = self._conv(pad, int(numOut / 2), kernel_size=3, strides=1, pad='VALID', name='conv')

            with tf.name_scope('norm_3'):
                norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,is_training=self.training)
                conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad='VALID', name='conv')

            return conv_3


    def _skip_layer(self, inputs, numOut, name='skip_layer'):
        """ Skip Layer
        Args:
            inputs    : Input Tensor
            numOut    : Desired output number of channel
            name    : Name of the bloc
        Returns:
            Tensor of shape (None, inputs.height, inputs.width, numOut)
        """
        with tf.name_scope(name):
            if inputs.get_shape().as_list()[3] == numOut:
                return inputs
            else:
                conv = self._conv(inputs, numOut, kernel_size=1, strides=1, name='conv')
                return conv


    def _conv(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv'):
        """ Spatial Convolution (CONV2D)
        Args:
            inputs            : Input Tensor (Data Type : NHWC)
            filters        : Number of filters (channels)
            kernel_size    : Size of kernel
            strides        : Stride
            pad                : Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name            : Name of the block
        Returns:
            conv            : Output Tensor (Convolved Input)
        """
        with tf.name_scope(name):
            # Kernel for convolution, Xavier Initialisation
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')

            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')

            if self.w_summary:
                with tf.device(self.cpu):
                    tf.summary.histogram('weights_summary', kernel, collections=['weight'])

            return conv








