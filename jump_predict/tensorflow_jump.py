import tensorflow as tf
import csv
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA
random.seed(6)
csv_read = csv.reader(open('./data.csv', 'r'))
people_l = []
ans_x, ans_y, c_x = [], [], []
title = csv_read.next()
title = ['id', 'age', 'exercise', 'competitive', 'height', 'weight', 'gender', 'injury', 'color', 'sleep', 'race', 'jump']
CATEGORICAL_COLUMNS = ["gender", "race", "color"]
CAST = ["id", "sleep", "competitive", "injury"]
for row in csv_read:
    people_l.append(row)
    x = []
    y = []
    cx = []
    for i in range(len(row)):
    	if title[i] in CAST:
    		continue
    	elif title[i] == "jump":
    		y.append(int(row[i]))
    	elif title[i] in CATEGORICAL_COLUMNS:
    		t = [0]*9
    		t[int(row[i])-1] = 1
    		cx.extend(t)
    	else:
    		x.append(float(row[i]))
    #x.append((x[3]**2)/x[2])
    ans_x.append(x)
    ans_y.append(y)
    c_x.append(cx)

# scaler = preprocessing.StandardScaler().fit(ans_x)
# ans_x = scaler.transform(ans_x)
#scaler = preprocessing.StandardScaler().fit(ans_x)
#ans_x = scaler.transform(ans_x)
#mypca = PCA(12)
ans_x = list(ans_x)
#ans_x[0:21] = mypca.fit_transform(ans_x[0:21])
#ans_x[21:] = mypca.transform(ans_x[21:])
ans_x = map(lambda x: list(x[0])+list(x[1]), zip(ans_x, c_x))
pp = zip(ans_x, ans_y)
print ans_x

def cal_loss(pp):
	random.shuffle(pp)
	ans_x, ans_y = zip(*pp)
	
	len_var = len(ans_x[0])
	print len_var
	W = tf.Variable(tf.ones([len_var,1]))
	#W2 = tf.Variable(tf.ones([10,1]))
	#W3 = tf.Variable(tf.ones([10,1]))
	x = tf.placeholder(tf.float32, [None, len_var])
	b = tf.Variable(tf.zeros([1]))
	#h1 = tf.matmul(x, W)
	#h1 = tf.nn.dropout(h1, 0.80)
	#h2 = tf.matmul(h1, W2)
	#h2 = tf.nn.dropout(h2, 0.80)
	sigma = tf.Variable(0.000125462)
	y = tf.matmul(x, W) + b #+  tf.random_normal([1], stddev=sigma, mean = 0)
	y_ = tf.placeholder(tf.float32,[None, 1])
	score = tf.reduce_sum(tf.abs(y-y_))
	lambda_parameter = 0.01
	cross_entropy = tf.reduce_sum(tf.abs(y-y_)) + tf.reduce_sum(tf.abs(W))# + tf.reduce_sum(tf.abs(W2))# + tf.reduce_sum(tf.abs(W3)) #+ lambda_parameter*(tf.nn.l2_loss(W) + tf.nn.l2_loss(b)) #+ 1e10*(tf.abs(sigma) - sigma) + 1e5*tf.abs(sigma)

	op = tf.train.AdamOptimizer(2)
	train_step = op.minimize(cross_entropy)
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	batch_xs, batch_ys = ans_x[0:21], ans_y[0:21]
	t_batch_xs, t_batch_ys = ans_x[21:], ans_y[21:]
	stop_n = 750
	small_error = 1e10
	now_n = 0
	lock = False
	for _ in range(60001):
		n, e = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_:batch_ys})
		if lock and e<small_error:
			print 'end in ', _
			ss, cs = sess.run([score, cross_entropy], {x:batch_xs, y_:batch_ys})      
			ss2, ry = sess.run([score, y], {x:t_batch_xs, y_:t_batch_ys})
			print("Train_score: %s    Test_score: %s cx: %s"%(ss, ss2, cs))
			break
		if e >= small_error:
			now_n += 1
			if now_n > stop_n:
				lock = True
		else:
			small_error = e
			now_n = 0
		if _ % 400==0:                
			ss, cs = sess.run([score, cross_entropy], {x:batch_xs, y_:batch_ys})      
			ss2, ry = sess.run([score, y], {x:t_batch_xs, y_:t_batch_ys})
			print("Train_score: %s    Test_score: %s cx: %s"%(ss, ss2, cs))
	re_y = []
	se_y = []
	for i in range(len(t_batch_ys)):
		if abs(t_batch_ys[i][0] - ry[i]) > 40:
			re_y.append(t_batch_ys[i][0])
		else:
			se_y.append(t_batch_ys[i][0])

	return ss2, re_y, se_y

if __name__ == '__main__':
	lll = 0
	fail = dict()
	success = dict()
	for i in range(100):
		ll, ry, sy=cal_loss(pp)
		lll = lll+ll
		for key in ry:
			fail[key]=fail.get(key, 0) + 1 
		for key in sy:
			success[key]=success.get(key, 0) + 1 
		print "sss:", success.items()
		print "fff:", fail.items()
		print i, lll/5.0/(i+1)


