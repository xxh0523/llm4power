import numpy as np
import torch
import pathlib
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.use('Agg')


def split_equal(total: int, num: int):
	quotient = int(total / num)
	remainder = total % num
	start = list()
	end = list()
	for i in range(num):
		if i < remainder:
			start.append(i*(quotient+1))
			end.append((i+1)*(quotient+1))
		else:
			start.append(i*quotient+remainder)
			end.append((i+1)*quotient+remainder)
	return start, end


class Logger:
    def __init__(self, logfile_path: pathlib.Path, train_log_name=None, test_log_name=None):
        self.training_logfile = None if train_log_name is None else open(logfile_path / train_log_name, 'w')
        self.testing_logfile = None if test_log_name is None else open(logfile_path / test_log_name, 'w')
    
    def __del__(self):
        if self.training_logfile is not None: self.training_logfile.close()
        if self.testing_logfile is not None: self.testing_logfile.close()
    
    def training_log(self, *strs):
        string = ' '.join(strs)
        self.training_logfile.write(string + '\n')
        tqdm.write(string)
    
    def testing_log(self, *strs):
        string = ' '.join(strs)
        self.testing_logfile.write(string + '\n')
        tqdm.write(string) 

class Picture_Drawer:
	def __init__(self):
		pass

	@staticmethod
	def draw_sopf_train_test(
		data_path: pathlib.Path, 
		n_processor: int, 
		total_step: int, 
		interval: int, 
		size=12,
		):
		font1 = {'size': size}
		mpl.rcParams['xtick.labelsize'] = size
		mpl.rcParams['ytick.labelsize'] = size
		
		training_data = np.load(data_path, allow_pickle=True)
		train = training_data['train'].reshape(-1,n_processor)
		eval = training_data['eval']
		x = np.arange(1, total_step+1)
		plt.xlim((0, total_step+1))
		plt.ylim((-1000, 1000))
		plt.xlabel('Training Step', fontdict=font1)
		plt.ylabel('Average Reward', fontdict=font1)
		plt.tick_params(labelsize=size)
		avg_train_reward = np.sum(train, axis=1) / n_processor
		plt.scatter(x, avg_train_reward,
		            s = 5,
		            label="training", 
		            c = 'b',
		            )
		# for i in range(n_processor):
		# 	plt.scatter(x, train[:, i],
		#             	s = 5,
		#             	label=f"training_processor{i}", 
		#             	c = 'b',
		#             	)
		x = np.arange(0, total_step+1, interval)
		avg_test_reward = np.average(eval[:x.shape[0]], axis=1)
		plt.plot(x, avg_test_reward,
		         label="evaluation", 
		         color = 'r',
		         )
		plt.legend(loc='best')
		plt.savefig(data_path.parent / 'training.jpg', dpi=300, bbox_inches='tight', format='jpg')
		plt.close()

	@staticmethod
	def contour_plot(xx, yy, zz):
		contour = plt.contour(xx, yy, zz, 100, cmap='rainbow')
		plt.clabel(contour, fontsize=6, colors='k')
		# 去掉坐标轴刻度
		# plt.xticks(())
		# plt.yticks(())
		# 填充颜色，f即filled,6表示将三色分成三层，cmap那儿是放置颜色格式，hot表示热温图（红黄渐变）
		# 更多颜色图参考：https://blog.csdn.net/mr_cat123/article/details/80709099
		# 颜色集，6层颜色，默认的情况不用写颜色层数,
		# c_set = plt.contourf(xx, yy, zz, cmap='rainbow')
		# or c_map='hot'
		# 设置颜色条，（显示在图片右边）
		# plt.colorbar(c_set)
		# 显示
		plt.show()
	
	@staticmethod
	def plot_3d(xx, yy, zz):
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		plt.xlim((xx.min(), xx.max()))
		plt.ylim((yy.min(), yy.max()))
		plt.clabel(ax, fontsize=6, colors='k')
		# ax.plot_surface(xx, yy, zz, cmap='rainbow')
		ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha=0.75, cmap='rainbow')
		ax.contour(xx, yy, zz, zdir='z', offset=zz.min(), cmap='rainbow')  # 等高线图，要设置offset，为Z的最小值
		ax.contour(xx, yy, zz, zdir='y', offset=yy.max(), cmap='rainbow')
		ax.contour(xx, yy, zz, zdir='x', offset=xx.min(), cmap='rainbow')
		# plt.show()
		# ax.scatter3D(xx, yy, zz, c='k', marker='o', s=50, linewidths=5, zorder=2)  # 绘制散点图
		plt.savefig('./test.jpg', dpi=300, bbox_inches='tight', format='jpg')
		plt.close()

	@staticmethod
	def draw_3d_pic(
		x: np.ndarray,
		y: np.ndarray,
		z: np.ndarray
	):
		mesh = int(x.shape[0] ** 0.5)
		x_mesh = x.reshape(mesh, mesh)
		y_mesh = y.reshape(mesh, mesh)
		z_mesh = z.reshape(mesh, mesh)
		Picture_Drawer.plot_3d(x_mesh, y_mesh, z_mesh)
