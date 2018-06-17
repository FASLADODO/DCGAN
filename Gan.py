from keras.layers import Conv2D,BatchNormalization,UpSampling2D,Dense,Dropout,Flatten,MaxPool2D,Activation,Reshape
import tensorflow as tf 
from keras import backend as K
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential,Model
from keras.activations import elu 
from keras.losses import binary_crossentropy

K.set_learning_phase(1)


sess=tf.Session()
K.set_session(sess)

discriminator=Sequential()
discriminator.add(Conv2D(filters=128,input_shape=(28,28,1),kernel_size=(3,3),strides=1,padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(Activation('tanh'))
discriminator.add(MaxPool2D())
discriminator.add(Dropout(0.4))

discriminator.add(Conv2D(filters=128,kernel_size=4,strides=1,padding='same'))
discriminator.add(Activation('tanh'))
discriminator.add(BatchNormalization())
discriminator.add(MaxPool2D())
discriminator.add(Dropout(0.6))

discriminator.add(Conv2D(filters=256,kernel_size=3,strides=1,padding='valid',activation='relu'))
discriminator.add(MaxPool2D())
discriminator.add(Flatten())


discriminator.add(Dropout(0.3))
discriminator.add(Dense(100,activation='relu'))

discriminator.add(Dense(1,activation='sigmoid'))



generator=Sequential()

generator.add(Dense(7*7*200,activation='tanh',input_shape=(100,)))
generator.add(Reshape((7,7,200)))
generator.add(UpSampling2D())

generator.add(Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='same'))
generator.add(BatchNormalization())
generator.add(Activation('tanh'))

generator.add(UpSampling2D())
generator.add(Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='same'))
generator.add(BatchNormalization())
generator.add(Activation('tanh'))

generator.add(Conv2D(filters=1,kernel_size=3,strides=1,padding='same'))


noise=K.placeholder(shape=(None,100),dtype=K.floatx())
images=K.placeholder(shape=(None,28,28,1),dtype=K.floatx())

generator_output=generator(noise)


gen_loss=tf.reduce_mean(binary_crossentropy(y_pred=discriminator(generator_output),y_true=K.ones_like(discriminator(generator_output))))

disc_fake_loss=tf.reduce_mean(binary_crossentropy(y_pred=discriminator(generator_output),y_true=K.zeros_like(discriminator(generator_output))))
disc_real_loss=tf.reduce_mean(binary_crossentropy(y_pred=discriminator(images),y_true=K.ones_like(discriminator(images))))
disc_loss=disc_fake_loss+disc_real_loss


trainD=tf.train.AdamOptimizer(0.0001).minimize(disc_loss)
trainG=tf.train.AdamOptimizer(0.0002).minimize(gen_loss)


init=tf.global_variables_initializer()
sess.run(init)

batch_size=32
iterations=data.train.num_examples//batch_size


for i in range(iterations):

	discriminator.trainable=True 
	generator.trainable=False

	noise_vector=np.random.normal(size=(batch_size,100))
	image_batch,_=data.train.next_batch(batch_size)
	image_batch=image_batch.reshape(-1,28,28,1)

	feed_dict={noise:noise_vector,images:image_batch} 


	for _ in range(10):
		sess.run(trainD,feed_dict=feed_dict)
		print('Discriminator loss ={}'.format(sess.run(disc_loss,feed_dict=feed_dict)))
	
	discriminator.trainable=False 
	generator.trainable=True
	sess.run(trainG,feed_dict=feed_dict)
	print('Generator loss = {}'.format(sess.run(gen_loss,feed_dict=feed_dict)))





