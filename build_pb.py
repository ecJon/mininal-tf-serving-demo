import tensorflow as tf

def genData():
    x_list = []
    y_list = []
    for x in range(10000):
        x_list.append(x)
        y_list.append(x*x)

    return x_list, y_list

def testtf(input, output, is_train=False):
    if is_train:
        test = tf.Variable([0.1], dtype=tf.float32, name="test")
    else:
        test = tf.Variable([0.2], dtype=tf.float32, name="test")

    b = tf.Variable([0.2], dtype=tf.float32, name="test")

    # pred = x * test + b

    pred = tf.add(tf.multiply(x, test, name=None), b, name="pred")

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y), name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(cost)

    return test, b, pred, optimizer

x = tf.placeholder(dtype=tf.float32, shape=[1], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[1], name="y")
x_batch, y_batch = genData()
w, b, pred, optimizer = testtf(x, y, is_train=True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    p, _ = sess.run([pred, optimizer], feed_dict={x:[x_batch[i]],y:[y_batch[i]]})
    print(str(p) + ' ' + str(x_batch[i]) + ' ' + str(y_batch[i]))
print(sess.run([w,b]))


builder = tf.saved_model.builder.SavedModelBuilder('./tfp/1')

pre_inputs = tf.saved_model.utils.build_tensor_info(x)

pred_inputs = tf.saved_model.utils.build_tensor_info(pred)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'x': pre_inputs},
        outputs={'y': pred_inputs},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
	"pre_x": prediction_signature
    },
    legacy_init_op=legacy_init_op)

builder.save()


