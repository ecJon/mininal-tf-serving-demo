from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


channel = implementations.insecure_channel('0.0.0.0', 8500)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'tfp'
#request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

request.model_spec.signature_name = 'pre_x'

request.inputs['x'].CopyFrom(
        tf.contrib.util.make_tensor_proto([2.], shape=[1]))

result = stub.Predict(request, 5.0)  # 5 seconds


print(result.outputs['y'].float_val)

