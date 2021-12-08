import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LinearRegression:

	# Huấn luyện mô hình bằng cách sử dụng kỹ thuật gradient descent
	def fit_grad(self, X, y, numOfIteration=None, learning_rate=1e-6, threshhold=1e-3, show_loss=False):
		dim = X.shape[1]
		self.w = np.ones(dim+1).reshape(-1, 1)
		N = X.shape[0]
		X = np.hstack((np.ones((N, 1)), X))

		# Trường hợp cho biết số lần lặp xác định
		if numOfIteration != None:
			cost = np.zeros((numOfIteration+1,1))
			r = np.dot(X, self.w) - y
			cost[0] = 0.5*np.sum(r*r)
			for epoch in range(1, numOfIteration+1):
				self.w[0] -= learning_rate*np.sum(r)
				for i in range(1, dim+1):
					self.w[i] -= learning_rate * np.sum(np.multiply(r, X[:,i].reshape(-1,1)))
				r = np.dot(X, self.w) - y
				cost[epoch] = 0.5*np.sum(r*r)
		# Trường hợp không cho biết số lần lặp: lặp đến khi thay đổi không đáng kể
		else:
			numOfIteration = 0
			cost = []
			r = np.dot(X, self.w) - y
			cost.append(0.5*np.sum(r*r))
			while True:
				self.w[0] -= learning_rate*np.sum(r)
				for i in range(1, dim+1):
					self.w[i] -= learning_rate * np.sum(np.multiply(r, X[:,i].reshape(-1,1)))
				numOfIteration += 1
				r = np.dot(X, self.w) - y
				cost.append(0.5*np.sum(r*r))
				if cost[-2] - cost[-1] <= threshhold:
					break
		# Biểu diễn giá trị hàm loss trong quá trình huấn luyện
		if show_loss:
			plt.plot(np.arange(0, numOfIteration+1).reshape(-1, 1), cost)
			plt.title('Giá trị hàm loss trong quá trình huấn luyện')
			plt.xlabel('Epoch #')
			plt.ylabel('Loss');
			plt.show()

	# Huấn luyện mô hình bằng phương pháp đại số
	def fit(self, X, y):
		N = X.shape[0]
		X = np.hstack((np.ones((N, 1)), X))
		self.w = np.linalg.pinv(X.T @ X) @ (X.T @ y)
		
	# Dự đoán
	def predict(self, X):
		N = X.shape[0]
		X = np.hstack((np.ones((N, 1)), X))
		return np.dot(X, self.w)

	# Đánh giá mô hình bằng hệ số xác định R2
	def score(self, X, y):
		# r2 = SSR/SST = 1 - SSE/SST
		y_pred = self.predict(X)
		sse = ((y - y_pred) ** 2).sum()
		sst = ((y - y.mean()) ** 2).sum()
		return 1 - sse/sst
	
	# Biểu diễn mô hình lên đồ thị
	def show(self, X, y, labels=['', '', '']):
		dim = X.shape[1]
		if dim == 1:
			plt.scatter(X, y)
			y_hat = self.predict(X)
			N = X.shape[0]
			plt.plot((X[0], X[N-1]),(y_hat[0], y_hat[N-1]), 'r')
			plt.xlabel(labels[0])
			plt.ylabel(labels[1])
			plt.show()
		elif dim == 2:
			m = int(max(X[:, 0].max(), X[:, 1].max()))		
			# create x, y
			xx, yy = np.meshgrid(range(m), range(m))
			zz = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
			zz = self.predict(zz).reshape(m, m)
			# plot the surface
			plt3d = plt.figure().gca(projection='3d')
			plt3d.scatter(X[:,0], X[:,1], y, c='r', marker='o')
			plt3d.plot_surface(xx,yy,zz, color='blue')
			plt3d.set_xlabel(labels[0])
			plt3d.set_ylabel(labels[1])
			plt3d.set_zlabel(labels[2])
			plt.show()
		else:
			print('Chỉ hỗ trợ vẽ mô hình có 1 hoặc 2 tham số đầu vào!')

	# Lưu mô hình
	def save(self, filename='LinearRegression.npy'):
		np.save(filename, self.w)

	# Tải mô hình đã có sẵn
	def load(self, filename='LinearRegression.npy'):
		self.w = np.load(filename)