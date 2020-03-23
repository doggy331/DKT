import tensorflow as tf

# 加载数据
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 查看数据
sample_x = x_train[0]
sample_y = y_train[0]
print(sample_x)
print('----------------')
print(sample_y)

# 参数设置
hiddens = [64,32]
output_size = 10
input_dim = 28
batch_size = 64
drop_out = 0.2

# 制造一个模型
def build_model(hiddens):
    lstm_layers = []
    for idx, hidden_size in enumerate(hiddens):
        lstm_cell = tf.keras.layers.LSTMCell(units=hidden_size,input_shape=(None, input_dim))
        # hidden_layer = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_layer,
        #                                              output_keep_prob=self.keep_prob)
        cell_dropout = tf.nn.RNNCellDropoutWrapper(lstm_cell,state_keep_prob=0.8)
        lstm_layers.append(cell_dropout)
    # self.hidden_cell = tf.nn.rnn_cell.MultiRNNCell(cells=hidden_layers, state_is_tuple=True)
    hidden_layers = tf.keras.layers.RNN(cell=lstm_layers,input_shape=(None, input_dim))
    model = tf.keras.models.Sequential([
                                        hidden_layers,
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(output_size)]
                                        )
    return model

model = build_model(hiddens)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=batch_size,
          epochs=5)

# 查看权重
# print(model.get_weights())