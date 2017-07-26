from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from keras.applications.vgg19 import decode_predictions

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

host, port = FLAGS.server.split(':')

request = predict_pb2.PredictRequest()
request.model_spec.name = 'vgg19'
request.model_spec.signature_name = 'predict'
request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(img))

channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
result = stub.Predict(request, 10.0)  # 10 secs timeout
to_decode = np.expand_dims(result.outputs['outputs'].float_val, axis=0)
decoded = decode_predictions(to_decode, 5)
print(decoded)