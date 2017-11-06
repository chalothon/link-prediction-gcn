import numpy as np
import networkx as nx
import scipy.sparse as sparse
import pandas as pd
import tensorflow as tf

graph = nx.Graph()

edges = pd.read_csv('../dat/facebook/1912.edges', sep=' ', skiprows=0, header=None, names=['node1','node2'])

nodes = pd.read_csv('../dat/facebook/1912.feat', sep=' ', skiprows=0, header=None, names=range(481))

diff = np.zeros(nodes.shape[0])
diff.fill(2)

for i in range(nodes.shape[0]):
    if nodes[260].iloc[i] == nodes[261].iloc[i]:
        diff[i] = 1
    else:
        diff[i] = 0

nodes = pd.read_csv('../dat/facebook/1912.feat', sep=' ', skiprows=0, header=None, names=range(481))
nodes.drop(range(1,260), axis=1, inplace=True)
nodes.drop(range(261,481), axis=1, inplace=True)

graph.clear()

graph.add_node(1912, gender=0)

for i in range(nodes.shape[0]):
    graph.add_node(nodes[0].iloc[i], gender=nodes[260].iloc[i])

for i in range(edges.shape[0]):
    graph.add_edge(edges['node1'].iloc[i], edges['node2'].iloc[i])
    
for i in list(graph.nodes):
    graph.add_edge(1912, i)

adj_img = sparse.csr_matrix.todense(nx.adjacency_matrix(graph))

oneD = np.ravel(adj_img)

traindat = np.zeros((nodes.shape[0], oneD.shape[0] + 1))

for i in range(nodes.shape[0]):
    traindat[i, 0] = i
    traindat[i, 1:] = oneD

trainlbl = np.zeros(nodes.shape[0])

for i in range(nodes.shape[0]):
    trainlbl[i] = nodes[260].iloc[i]

testdat = traindat[700:,:]
testlbl = np.expand_dims(trainlbl[700:], axis=1)
traindat = traindat[:700,:]
trainlbl = np.expand_dims(trainlbl[:700], axis=1)

# placeholders for I/O
inp = tf.placeholder('float', [None, 571537], name='inp')
otp = tf.placeholder('float', [None, 1], name='otp')

# build weight matrices
w1 = tf.Variable(tf.random_normal([571537,1000], stddev=0.01), name='w1')
w2 = tf.Variable(tf.random_normal([1000,500], stddev=0.01), name='w2')
w3 = tf.Variable(tf.random_normal([500,50], stddev=0.01), name='w3')
w4 = tf.Variable(tf.random_normal([50,1], stddev=0.01), name='w4')

# weight matrix histogram summaries
tf.summary.histogram('w1_summ',w1)
tf.summary.histogram('w2_summ',w2)
tf.summary.histogram('w3_summ',w3)
tf.summary.histogram('w4_summ',w4)

# build model
def model(inp, w1, w2, w3, w4):
	with tf.name_scope('layer1'):
		l1 = tf.nn.relu(tf.matmul(inp, w1))
	with tf.name_scope('layer2'):
		l2 = tf.nn.relu(tf.matmul(l1, w2))
	with tf.name_scope('layer3'):
		l3 = tf.nn.relu(tf.matmul(l2, w3))
	with tf.name_scope('layer4'):
		return tf.matmul(l3, w4)

mod = model(inp, w1, w2, w3, w4)

# build loss function
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mod, labels=otp))
	train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
	tf.summary.scalar('loss', loss)

# validate
with tf.name_scope('validation'):
	correct = tf.equal(tf.argmax(otp, 1), tf.argmax(mod, 1))
	val_op  = tf.reduce_mean(tf.cast(correct, 'float'))
	tf.summary.scalar('accuracy', val_op)

with tf.Session() as sess:
	# write out log
	writer = tf.summary.FileWriter('../dat/logs/mlp', sess.graph)
	merged = tf.summary.merge_all()

	# init
	tf.global_variables_initializer().run()

	# train
	for i in range(10):
		sess.run(train_op, feed_dict={inp: traindat, otp: trainlbl})

		summary, acc = sess.run([merged, val_op],
			feed_dict={inp: testdat, otp: testlbl})
		writer.add_summary(summary, i)

		print(i, acc)

