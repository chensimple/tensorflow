import tornado.ioloop
import tornado.web
import tornado.websocket
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_cnn as mnist_interence
EVAL_INTERVAL_SECS = 10
BATCH_SIZE = 100

mnist = input_data.read_data_sets('./mni_data', one_hot=True)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("main.html")

    def post(self):
        pass


def cat(img):
    new = []
    for i in range(len(img)):
        if (i+1) % 4 == 0:
            pass
        else:
            new.append(img[i])
    return new


class ChatSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        pass

    def on_close(self):
        pass

    def on_message(self, message):
        img = message.split(",")  # 3136 RGBA
        # 将字符串转为整型
        for i in range(len(img)):
            img[i] = int(img[i])

        # 将RGBA转换为RGB
        img_rgb = cat(img)
        # 将一维矩阵转换为28*28*3的三维矩阵
        new_img = np.array(img_rgb, dtype=np.uint8).reshape((28, 28, 3))
        # 输入图片数据的张量占位符
        img_ = tf.placeholder(tf.uint8, shape=[28, 28, 3])
        # 转换为float32
        img_float = tf.image.convert_image_dtype(img_, dtype=tf.float32)
        # 灰度化
        img_gray = tf.image.rgb_to_grayscale(img_float)
        e = show_img(img_gray, img_, new_img)
        num = get_num(e)
        self.write_message(str(num))


def show_img(iimg_gray, iimg_, iimg_data):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        show = sess.run(iimg_gray, feed_dict={iimg_: iimg_data})
        plt.imshow(show[:, :, 0], cmap='gray_r')
        # plt.show()
    return show


def get_num(e):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x-input')
        reshape_xs = np.reshape(e, (-1, 28, 28, 1))
        y = mnist_interence.interence(x, False, None)
        result = tf.argmax(y, 1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('model')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                key = sess.run(result, feed_dict={x: reshape_xs})
                print(key[0])
                return key[0]


if __name__ == "__main__":

    application = tornado.web.Application([
        (r"/", MainHandler),
        (r"/echo", ChatSocketHandler)
    ],
        static_path=os.path.join(os.path.dirname(__file__), 'static'))
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
