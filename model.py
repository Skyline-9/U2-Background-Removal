import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D


class ConvBlock(tf.keras.layers.Layer):
	"""Convolution 2D Block with a stride of 1 and same padding.

	Attributes:
		out_ch: Integer - Number of channels produced by the convolution
		dirate: Integer or tuple/list of 2 integers - Dilation rate to use for dilated convolution
	"""

	def __init__(self, out_ch=3, dirate=1):
		super(ConvBlock, self).__init__()

		# Layers
		self.conv = Conv2D(out_ch, (3, 3), strides=1, padding='same', dilation_rate=dirate)
		self.bn = BatchNormalization()
		self.relu = ReLU()

	def call(self, inputs):
		# Forward pass
		hx = inputs

		x = self.conv(hx)
		x = self.bn(x)
		x = self.relu(x)

		return x


class RSU_7(tf.keras.layers.Layer):
	"""RSU Block 7 (En_1 and De_1)

	Attributes:
		mid_ch: Integer - Number of mid channels
		out_ch: Integer - Number of channels produced by the convolution
	"""

	def __init__(self, mid_ch=12, out_ch=3):
		super(RSU_7, self).__init__()

		self.conv_b0 = ConvBlock(out_ch, dirate=1)

		# Conv_b1 to Conv_b5 are downsample x1/2
		self.conv_b1 = ConvBlock(mid_ch, dirate=1)
		self.pool1 = MaxPool2D(2, strides=(2, 2))

		self.conv_b2 = ConvBlock(mid_ch, dirate=1)
		self.pool2 = MaxPool2D(2, strides=(2, 2))

		self.conv_b3 = ConvBlock(mid_ch, dirate=1)
		self.pool3 = MaxPool2D(2, strides=(2, 2))

		self.conv_b4 = ConvBlock(mid_ch, dirate=1)
		self.pool4 = MaxPool2D(2, strides=(2, 2))

		self.conv_b5 = ConvBlock(mid_ch, dirate=1)
		self.pool5 = MaxPool2D(2, strides=(2, 2))

		# Downsample but no pool
		self.conv_b6 = ConvBlock(mid_ch, dirate=1)

		# Conv Block but Dilation=2
		self.conv_b7 = ConvBlock(mid_ch, dirate=2)

		# Same thing but with upsampling
		self.conv_b6_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b5_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b4_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b3_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b2_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_5 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b1_u = ConvBlock(out_ch, dirate=1)
		self.upsample_6 = UpSampling2D(size=(2, 2), interpolation='bilinear')

	def call(self, inputs):
		# Forward pass
		hx = inputs
		hxin = self.conv_b0(hx)

		# Downsampling
		hx1 = self.conv_b1(hxin)
		hx = self.pool1(hx1)

		hx2 = self.conv_b2(hx)
		hx = self.pool2(hx2)

		hx3 = self.conv_b3(hx)
		hx = self.pool3(hx3)

		hx4 = self.conv_b4(hx)
		hx = self.pool4(hx4)

		hx5 = self.conv_b5(hx)
		hx = self.pool5(hx5)

		hx6 = self.conv_b6(hx)

		hx7 = self.conv_b7(hx6)

		# Upsampling
		hx6u = self.conv_b6_u(tf.concat([hx7, hx6], axis=3))
		hx6u_up = self.upsample_5(hx6u)

		hx5u = self.conv_b5_u(tf.concat([hx6u_up, hx5], axis=3))
		hx5u_up = self.upsample_4(hx5u)

		hx4u = self.conv_b4_u(tf.concat([hx5u_up, hx4], axis=3))
		hx4u_up = self.upsample_3(hx4u)

		hx3u = self.conv_b3_u(tf.concat([hx4u_up, hx3], axis=3))
		hx3u_up = self.upsample_2(hx3u)

		hx2u = self.conv_b2_u(tf.concat([hx3u_up, hx2], axis=3))
		hx2u_up = self.upsample_1(hx2u)

		hx1u = self.conv_b1_u(tf.concat([hx2u_up, hx1], axis=3))

		return hx1u + hxin


class RSU_6(tf.keras.layers.Layer):
	"""RSU Block 6 (En_2 and De_2)

	Attributes:
		mid_ch: Integer - Number of mid channels
		out_ch: Integer - Number of channels produced by the convolution
	"""

	def __init__(self, mid_ch=12, out_ch=3):
		super(RSU_6, self).__init__()
		self.conv_b0 = ConvBlock(out_ch, dirate=1)

		self.conv_b1 = ConvBlock(mid_ch, dirate=1)
		self.pool1 = MaxPool2D(2, strides=(2, 2))

		self.conv_b2 = ConvBlock(mid_ch, dirate=1)
		self.pool2 = MaxPool2D(2, strides=(2, 2))

		self.conv_b3 = ConvBlock(mid_ch, dirate=1)
		self.pool3 = MaxPool2D(2, strides=(2, 2))

		self.conv_b4 = ConvBlock(mid_ch, dirate=1)
		self.pool4 = MaxPool2D(2, strides=(2, 2))

		self.conv_b5 = ConvBlock(mid_ch, dirate=1)
		self.pool5 = MaxPool2D(2, strides=(2, 2))

		self.conv_b6 = ConvBlock(mid_ch, dirate=2)

		# Upsample
		self.conv_b5_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b4_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b3_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b2_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b1_u = ConvBlock(out_ch, dirate=1)
		self.upsample_5 = UpSampling2D(size=(2, 2), interpolation='bilinear')

	def call(self, inputs):
		hx = inputs
		hxin = self.conv_b0(hx)

		# Downsampling
		hx1 = self.conv_b1(hxin)
		hx = self.pool1(hx1)

		hx2 = self.conv_b2(hx)
		hx = self.pool2(hx2)

		hx3 = self.conv_b3(hx)
		hx = self.pool3(hx3)

		hx4 = self.conv_b4(hx)
		hx = self.pool4(hx4)

		hx5 = self.conv_b5(hx)

		hx6 = self.conv_b6(hx5)

		# Upsampling
		hx5d = self.conv_b5_u(tf.concat([hx6, hx5], axis=3))
		hx5dup = self.upsample_4(hx5d)

		hx4d = self.conv_b4_u(tf.concat([hx5dup, hx4], axis=3))
		hx4dup = self.upsample_3(hx4d)

		hx3d = self.conv_b3_u(tf.concat([hx4dup, hx3], axis=3))
		hx3dup = self.upsample_2(hx3d)

		hx2d = self.conv_b2_u(tf.concat([hx3dup, hx2], axis=3))
		hx2dup = self.upsample_1(hx2d)

		hx1d = self.conv_b1_u(tf.concat([hx2dup, hx1], axis=3))

		return hx1d + hxin


class RSU_5(tf.keras.layers.Layer):
	"""RSU Block 5 (En_4 and De_4)

	Attributes:
		mid_ch: Integer - Number of mid channels
		out_ch: Integer - Number of channels produced by the convolution
	"""

	def __init__(self, mid_ch=12, out_ch=3):
		super(RSU_5, self).__init__()
		self.conv_b0 = ConvBlock(out_ch, dirate=1)

		self.conv_b1 = ConvBlock(mid_ch, dirate=1)
		self.pool1 = MaxPool2D(2, strides=(2, 2))

		self.conv_b2 = ConvBlock(mid_ch, dirate=1)
		self.pool2 = MaxPool2D(2, strides=(2, 2))

		self.conv_b3 = ConvBlock(mid_ch, dirate=1)
		self.pool3 = MaxPool2D(2, strides=(2, 2))

		self.conv_b4 = ConvBlock(mid_ch, dirate=1)
		self.pool4 = MaxPool2D(2, strides=(2, 2))

		self.conv_b5 = ConvBlock(mid_ch, dirate=2)

		# Upsample
		self.conv_b4_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b3_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b2_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b1_u = ConvBlock(out_ch, dirate=1)
		self.upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')

	def call(self, inputs):
		# Forward pass
		hx = inputs
		hxin = self.conv_b0(hx)

		hx1 = self.conv_b1(hxin)
		hx = self.pool1(hx1)

		hx2 = self.conv_b2(hx)
		hx = self.pool2(hx2)

		hx3 = self.conv_b3(hx)
		hx = self.pool3(hx3)

		hx4 = self.conv_b4(hx)

		hx5 = self.conv_b5(hx4)

		hx4u = self.conv_b4_u(tf.concat([hx5, hx4], axis=3))
		hx4u_up = self.upsample_3(hx4u)

		hx3u = self.conv_b3_u(tf.concat([hx4u_up, hx3], axis=3))
		hx3u_up = self.upsample_2(hx3u)

		hx2u = self.conv_b2_u(tf.concat([hx3u_up, hx2], axis=3))
		hx2u_up = self.upsample_1(hx2u)

		hx1u = self.conv_b1_u(tf.concat([hx2u_up, hx1], axis=3))

		return hx1u + hxin


class RSU_4(tf.keras.layers.Layer):
	"""RSU Block 4 (En_5 and De_5)

	Attributes:
		mid_ch: Integer - Number of mid channels
		out_ch: Integer - Number of channels produced by the convolution
	"""

	def __init__(self, mid_ch=12, out_ch=3):
		super(RSU_4, self).__init__()
		self.conv_b0 = ConvBlock(out_ch, dirate=1)

		self.conv_b1 = ConvBlock(mid_ch, dirate=1)
		self.pool1 = MaxPool2D(2, strides=(2, 2))

		self.conv_b2 = ConvBlock(mid_ch, dirate=1)
		self.pool2 = MaxPool2D(2, strides=(2, 2))

		self.conv_b3 = ConvBlock(mid_ch, dirate=1)
		self.pool3 = MaxPool2D(2, strides=(2, 2))

		self.conv_b4 = ConvBlock(mid_ch, dirate=2)

		# Upsample
		self.conv_b3_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b2_u = ConvBlock(mid_ch, dirate=1)
		self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.conv_b1_u = ConvBlock(out_ch, dirate=1)
		self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')

	def call(self, inputs):
		# Forward pass
		hx = inputs
		hxin = self.conv_b0(hx)

		hx1 = self.conv_b1(hxin)
		hx = self.pool1(hx1)

		hx2 = self.conv_b2(hx)
		hx = self.pool2(hx2)

		hx3 = self.conv_b3(hx)
		hx4 = self.conv_b4(hx3)

		hx3u = self.conv_b3_u(tf.concat([hx4, hx3], axis=3))
		hx3u_up = self.upsample_2(hx3u)

		hx2u = self.conv_b2_u(tf.concat([hx3u_up, hx2], axis=3))
		hx2u_up = self.upsample_1(hx2u)

		hx1u = self.conv_b1_u(tf.concat([hx2u_up, hx1], axis=3))

		return hx1u + hxin


class RSU_4F(tf.keras.layers.Layer):
	"""RSU Block 4F (En_5, En_6, De_5)

	Attributes:
		mid_ch: Integer - Number of mid channels
		out_ch: Integer - Number of channels produced by the convolution
	"""

	def __init__(self, mid_ch=12, out_ch=3):
		super(RSU_4F, self).__init__()
		self.conv_b0 = ConvBlock(out_ch, dirate=1)
		self.conv_b1 = ConvBlock(mid_ch, dirate=1)

		self.conv_b2 = ConvBlock(mid_ch, dirate=2)
		self.conv_b3 = ConvBlock(mid_ch, dirate=4)

		self.conv_b4 = ConvBlock(mid_ch, dirate=8)

		self.conv_b3_d = ConvBlock(mid_ch, dirate=4)
		self.conv_b2_d = ConvBlock(mid_ch, dirate=2)

		self.conv_b1_d = ConvBlock(out_ch, dirate=1)

	def call(self, inputs):
		# Forward pass
		hx = inputs
		hxin = self.conv_b0(hx)

		hx1 = self.conv_b1(hxin)
		hx2 = self.conv_b2(hx1)
		hx3 = self.conv_b3(hx2)
		hx4 = self.conv_b4(hx3)

		hx3d = self.conv_b3_d(tf.concat([hx4, hx3], axis=3))
		hx2d = self.conv_b2_d(tf.concat([hx3d, hx2], axis=3))
		hx1d = self.conv_b1_d(tf.concat([hx2d, hx1], axis=3))

		return hx1d + hxin


class U2NET(tf.keras.models.Model):
	"""
	Full sized U2 model
	"""

	def __init__(self, out_ch=1):
		super(U2NET, self).__init__()

		# Encoder
		self.stage1 = RSU_7(32, 64)
		self.pool12 = MaxPool2D((2, 2), 2)

		self.stage2 = RSU_6(32, 128)
		self.pool23 = MaxPool2D((2, 2), 2)

		self.stage3 = RSU_5(64, 256)
		self.pool34 = MaxPool2D((2, 2), 2)

		self.stage4 = RSU_4(128, 512)
		self.pool45 = MaxPool2D((2, 2), 2)

		self.stage5 = RSU_4F(256, 512)
		self.pool56 = MaxPool2D((2, 2), 2)

		self.stage6 = RSU_4F(256, 512)

		# Decoder
		self.stage5d = RSU_4F(256, 512)
		self.stage4d = RSU_4(128, 256)
		self.stage3d = RSU_5(64, 128)
		self.stage2d = RSU_6(32, 64)
		self.stage1d = RSU_7(16, 64)

		self.side1 = Conv2D(out_ch, (3, 3), padding='same')
		self.side2 = Conv2D(out_ch, (3, 3), padding='same')
		self.side3 = Conv2D(out_ch, (3, 3), padding='same')
		self.side4 = Conv2D(out_ch, (3, 3), padding='same')
		self.side5 = Conv2D(out_ch, (3, 3), padding='same')
		self.side6 = Conv2D(out_ch, (3, 3), padding='same')

		self.upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
		self.upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
		self.upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
		self.upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')
		self.upsample_5 = UpSampling2D(size=(2, 2), interpolation='bilinear')
		self.upsample_6 = UpSampling2D(size=(2, 2), interpolation='bilinear')

		self.upsample_out_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
		self.upsample_out_3 = UpSampling2D(size=(4, 4), interpolation='bilinear')
		self.upsample_out_4 = UpSampling2D(size=(8, 8), interpolation='bilinear')
		self.upsample_out_5 = UpSampling2D(size=(16, 16), interpolation='bilinear')
		self.upsample_out_6 = UpSampling2D(size=(32, 32), interpolation='bilinear')

		self.outconv = Conv2D(out_ch, (1, 1), padding='same')

	def call(self, inputs):
		hx = inputs

		# Stage 1
		hx1 = self.stage1(hx)
		hx = self.pool12(hx1)

		# Stage 2
		hx2 = self.stage2(hx)
		hx = self.pool23(hx2)

		# Stage 3
		hx3 = self.stage3(hx)
		hx = self.pool34(hx3)

		# Stage 4
		hx4 = self.stage4(hx)
		hx = self.pool45(hx4)

		# Stage 5
		hx5 = self.stage5(hx)
		hx = self.pool56(hx5)

		# stage 6
		hx6 = self.stage6(hx)
		hx6up = self.upsample_1(hx6)
		side6 = self.upsample_out_6(self.side6(hx6))

		# Decoder and side output
		hx5d = self.stage5d(tf.concat([hx6up, hx5], axis=3))
		hx5dup = self.upsample_2(hx5d)
		side5 = self.upsample_out_5(self.side5(hx5d))

		hx4d = self.stage4d(tf.concat([hx5dup, hx4], axis=3))
		hx4dup = self.upsample_3(hx4d)
		side4 = self.upsample_out_4(self.side4(hx4d))

		hx3d = self.stage3d(tf.concat([hx4dup, hx3], axis=3))
		hx3dup = self.upsample_4(hx3d)
		side3 = self.upsample_out_3(self.side3(hx3d))

		hx2d = self.stage2d(tf.concat([hx3dup, hx2], axis=3))
		hx2dup = self.upsample_5(hx2d)
		side2 = self.upsample_out_2(self.side2(hx2d))

		hx1d = self.stage1d(tf.concat([hx2dup, hx1], axis=3))
		side1 = self.side1(hx1d)

		fused_output = self.outconv(tf.concat([side1, side2, side3, side4, side5, side6], axis=3))

		sig = tf.keras.activations.sigmoid
		return tf.stack([sig(fused_output), sig(side1), sig(side2), sig(side3), sig(side4), sig(side5), sig(side6)])
