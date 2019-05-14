import tensorflow as tf
import numpy as np
import collections
from collections import namedtuple
import utils
import vgg19

LAYER_INFO = namedtuple('LayerInfo', ['names', 'weights'])

CONTENT_LAYER_INFO = LAYER_INFO(names=['conv4_2'],
                                weights=[1.0])
STYLE_LAYER_INFO = LAYER_INFO(names=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
                              weights=[.2,.2,.2,.2,.2])


class StyleTransfer:

    def __init__(self, sess, content_img, style_img, net):

        # hyperparameter
        self.alpha = 1e-3
        self.beta = 1

        self.net = net
        self.sess = sess
        self.init_img = np.random.normal(size=content_img.shape, scale=np.std(content_img))

        # preprocess images
        self.content_img_prep = self.net.preprocess(content_img)
        self.style_img_prep = self.net.preprocess(style_img)
        self.init_img_prep = self.net.preprocess(self.init_img)

        # layers' info
        self.content_layer_info = dict(zip(CONTENT_LAYER_INFO.names, CONTENT_LAYER_INFO.weights))
        self.style_layer_info = dict(zip(STYLE_LAYER_INFO.names, STYLE_LAYER_INFO.weights))

        # build graph for style transfer
        self._build_graph()

    def _build_graph(self):

        # output initialization
        self.output_img = tf.Variable(self.init_img, trainable=True, dtype=tf.float32)

        # graph input
        self.content_inp = tf.placeholder(tf.float32,
                                          shape=self.content_img_prep.shape,
                                          name='content_input')
        self.style_inp = tf.placeholder(tf.float32,
                                        shape=self.style_img_prep.shape,
                                        name='style_input')
        # content's layers feature vectors
        self.content_fvs = self.net.feed_forward(self.content_inp, scope='content')

        # style's layers features and its gram matrice
        self.style_fvs = self.net.feed_forward(self.style_img_prep, scope='style')
        self.style_grams = {}
        for name in self.style_layer_info.keys():
            self.style_grams[name] = self._gram_matrix(self.style_fvs[name])

        # output's feature vectors
        self.output_fvs = self.net.feed_forward(self.output_img, scope='mixed')

        # compute loss
        content_loss = 0
        style_loss = 0
        for layer_name in self.output_fvs:
            # compute content loss
            if layer_name in self.content_layer_info.keys():
                content_fv = self.content_fvs[layer_name]
                output_fv = self.output_fvs[layer_name]
                weight = self.content_layer_info[layer_name]

                content_loss += weight * tf.reduce_sum(tf.pow((content_fv-output_fv), 2)) / 2

            # compute content loss
            elif layer_name in self.style_layer_info.keys():
                style_gram = self.style_grams[layer_name]
                output_fv = self.output_fvs[layer_name]
                output_gram = self._gram_matrix(output_fv)
                weight = self.style_layer_info[layer_name]

                _, h, w, d = output_fv.get_shape()  # first return value is batch size (must be one)
                N = h.value * w.value       # product of width and height
                M = d.value                 # number of filters

                style_loss += weight * (1. / (4 * N ** 2 * M ** 2)) * \
                              tf.reduce_sum(tf.pow((style_gram-output_gram), 2))

        # total loss
        self.content_loss = content_loss
        self.style_loss = style_loss
        self.total_loss = self.alpha * content_loss + self.beta * style_loss


    def get_output(self, max_iter=1000):

        # define optimizer
        global _iter
        _iter = 0

        def loss_callback(total_loss, content_loss, style_loss):
            global _iter
            print(f'iter: {_iter}, total_loss={total_loss}, content_loss={content_loss}, style_loss={style_loss}')
            _iter +=1

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.total_loss,
            method = 'L-BFGS-B',
            options={'maxiter': max_iter, 'disp': True}
        )

        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # optimize
        print(f'start to optimize with max_iter={max_iter}')
        optimizer.minimize(self.sess,
                           feed_dict={self.content_inp: self.content_img_prep,
                                      self.style_inp: self.style_img_prep
                                      },
                           fetches=[self.total_loss, self.content_loss, self.style_loss],
                           loss_calback=loss_callback)

        # get final result
        final_output_img = self.sess.run(self.output_img)
        final_output_img = np.clip(self.net.undo_preprocess(final_output_img), 0.0, 255.0)

        return final_output_img

    def _gram_matrix(self, tensor):

        # reshape tensor to flatten to channel-direction)
        #   before = [n_batch, width, height, n_channel)
        #   after = [n_batch, n_channel]

        tensor = tf.reshape(tensor, shape=[-1, tensor.get_shape()[3]])

        # gram is dot-product of all combinations of feature-channel
        gram = tf.matmul(tf.transpose(tensor), tensor)

        return gram

"""add one dim for batch"""
# VGG19 requires input dimension to be (batch, height, width, channel)
def add_one_dim(image):
    shape = (1,) + image.shape
    return np.reshape(image, shape)


def remove_one_dim(batched_image):
    shape = batched_image.shape
    return np.reshape(batched_image, shape[1:])


def main():

    # initiate VGG19 model
    model_file_path = '../pre_trained_model/imagenet-vgg-verydeep-19.mat'
    vgg_net = vgg19.VGG19(model_file_path)

    # prepare input images
    content_img = utils.load_image('./images/gyeongbokgung.jpg')
    style_img = utils.load_image('./images/the_scream.jpg')

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    st = StyleTransfer(sess=sess,
                       content_img=add_one_dim(content_img),
                       style_img=add_one_dim(style_img),
                       net=vgg_net
                       )
    output_img = st.get_output(max_iter=10)
    output_img = remove_one_dim(output_img)

    # save result
    utils.save_image(output_img, 'output.jpg')

    utils.plot_images(content_img,style_img, output_img)

    sess.close()

def main2():
    def print_loss(loss_evaled, vector_evaled):
        print(loss_evaled, vector_evaled)

    vector = tf.Variable([7., 7.], 'vector')
    loss = tf.reduce_sum(tf.square(vector))

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
        loss, method='L-BFGS-B',
        options={'maxiter': 100})

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        optimizer.minimize(session,
                           loss_callback=print_loss,
                           fetches=[loss, vector])
        print(vector.eval())

if __name__ == '__main__':
    main()
