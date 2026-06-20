import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib import cm

# Parte 1 - Codigo de um neuronio ----------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

x = np.linspace(-5.0, 5.0)
y = sigmoid(x)

plt.plot(x,y)

class Neuron:
    def __init__(self, input_size = 5):
        self.w = np.random.random(input_size)
        self.b = np.random.random()
    def compute(self, inputs):
        s = np.dot(self.w, inputs) + self.b
        z = sigmoid(s)
        return z
    
n = Neuron()
n.compute([1.0, 2.0, 3.0, 4.0, 5.0])

# Aula 2 - Nêuronios das portas lógicas AND, NOT e OR ----------------------------------------

def linear(x):
    return np.clip(x, 0, 1)

f = np.linspace(-5.0, 5.0, 1000)  # mais pontos = mais suave
g = linear(f)

plt.plot(f, g)
plt.show()

# ------------------------------------------------------------------------

class Neuronio:
    def __init__(self, w0, w1, bias):
        self.w0 = w0
        self.w1 = w1
        self.bias = bias
    def compute(self, a, b):
        s = self.w0 * a + self.w1 * b + self.bias
        z = linear(s)
        return z
    
# ------------------------------------------------------------------------
 
neur_not = Neuronio(-1.0, 0.0, 1.0)
neur_and = Neuronio(1.0, 1.0, -1.0)
neur_or = Neuronio(1.0, 1.0, 0.0)

# ------------------------------------------------------------------------

entradas = [(0,0), (0,1), (1,0), (1,1)]
for a,b in entradas:
  print('a =',a,'b =',b,\
        '| and =',neur_and.compute(a,b),\
        'or =',neur_or.compute(a,b),\
        'not =',neur_not.compute(a,0))
  
# Aula 3 - Uso dataset Iris FLower ----------------------------------------

from google.colab import files
import io

uploaded = files.upload()
f = io.BytesIO(uploaded['Iris.csv'])

f.seek(0)
lines = f.readlines()

# ------------------------------------------------------------------------

x = np.zeros((len(lines)-1,4))
y = np.zeros((len(lines)-1,3))
cat = np.array(['Iris-setosa','Iris-versicolor','Iris-virginica'])

# ------------------------------------------------------------------------

for i, line in enumerate(lines[1:]):
  s = line.decode()[:-1]
  _,sl,sw,pl,pw,sp = s.split(',')
  sl = float(sl)
  sw = float(sw)
  pl = float(pl)
  pw = float(pw)
  x[i:] = np.array([sl,sw,pl,pw])
  y[i:] = (cat == sp).astype('float')

print(y)

# Aula 4 - Descida do gradiente (Aprendizado da máquina) ----------------------------------------

def func(x,y):
  return (x**2) * np.sin(y)

def grad(x,y):
  return np.array([2*x*np.sin(y), (x**2)*np.cos(y)])

# ------------------------------------------------------------------------

x = np.outer(np.linspace(-5.0, 5.0, 100), np.ones(100))
y = np.outer(np.ones(100), np.linspace(-5.0, 5.0, 100))
z = func(x,y)

# ------------------------------------------------------------------------

ax = plt.axes(projection='3d')
ax.view_init(50, 30)
ax.plot_surface(x, y, z, cmap=cm.coolwarm)
ax.set_xlabel('eixo x')
ax.set_ylabel('eixo y')
ax.set_zlabel('eixo z')

# ------------------------------------------------------------------------

ax = plt.axes()
ax.imshow(-z.T, cmap=cm.coolwarm, extent=(-5, 5, -5, 5), interpolation='bilinear')
ax.set_xlabel('eixo x')
ax.set_ylabel('eixo y')

# ------------------------------------------------------------------------

qx = []
qy = []
qu = []
qv = []

for xi in np.linspace(-1.5*np.pi, 1.5*np.pi, 15):
  for yi in np.linspace(-1.5*np.pi, 1.5*np.pi, 15):
    ui, vi = grad(xi, yi)
    qx.append(xi)
    qy.append(yi)
    qu.append(ui)
    qv.append(vi)

# ------------------------------------------------------------------------

ax = plt.axes()
ax.imshow(-z.T, cmap=cm.coolwarm, extent=(-5, 5, -5, 5), interpolation='bilinear')
Q = ax.quiver(qx, qy, qu, qv, scale=360.0, pivot='mid')
qk = ax.quiverkey(Q, 0.1, 0.1, 0.1, '', labelpos='E', coordinates='figure')

ax.set_xlabel('eixo x')
ax.set_ylabel('eixo y')
# plt.savefig('gradiente3.png', dpi=300)
# files.download('gradiente3.png')

print(func(2,-2))
print(grad(2, -2))

# ------------------------------------------------------------------------

def func3d(x):
  x1, x2 = x
  f = 0.8
  return np.cos(x1*f) * (25 - x1**2) - np.cos(x2*f) * (25 - x2**2)

# ------------------------------------------------------------------------

def grad3d(x):
  x1, x2 = x
  h = 0.01
  h1 = np.array([h, 0.0])
  h2 = np.array([0.0, h])
  z = func3d(x)
  df_dx = (func3d(x+h1) - z)/h
  df_dy = (func3d(x+h2) - z)/h
  return np.array([df_dx, df_dy])

# ------------------------------------------------------------------------

def grad_desc(x, step=0.005):
  x = x - step * grad3d(x)
  return x

# ------------------------------------------------------------------------

x = np.array([np.outer(np.linspace(-5.0, 5.0, 100), np.ones(100)), \
     np.outer(np.ones(100), np.linspace(-3.0, 7.0, 100))])
z = func3d(x)

# ------------------------------------------------------------------------

ax = plt.axes(projection='3d')
ax.view_init(50, -45)
ax.plot_surface(x[0,:], y[1,:], z, cmap=cm.coolwarm)
plt.axis('off')

# ------------------------------------------------------------------------

xi = np.array([0.5, 3.0])
xs = []
zs = []

for i in range(200):
  zi = func3d(xi)
  xs.append(xi)
  zs.append(zi)
  xi = xi - 0.001 * grad3d(xi)

xs = np.array(xs)
zs = np.array(zs)

ax = plt.axes(projection='3d')
ax.view_init(50, -45)
ax.plot_surface(x[0,:], x[1,:], z, cmap=cm.coolwarm)
ax.plot(xs[:,0], xs[:,1], zs, '-', c='b', zorder=100)
# plt.axis('off')

# Aula 4: Implementação do backpropagation ----------------------------------------

!rm -rf Iris.csv

uploaded = files.upload()
f = io.BytesIO(uploaded['Iris.csv'])

f.seek(0)

lines = f.readlines()

X = list()
Y = list()
cats = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

for line in lines[1:]:
  line = line.decode('utf-8')  # ← adicionada essa linha

  _,sl,sw,pl,pw,sp = line[:-1].split(',')
  sl = float(sl)
  sw = float(sw)
  pl = float(pl)
  pw = float(pw)
  sp = [1.0 if sp == cat else 0.0 for cat in cats]
  X.append([sl,sw,pl,pw])
  Y.append(sp)

# ------------------------------------------------------------------------

total = len(X)
indexes = list(range(total))
np.random.shuffle(indexes)
Xs = [X[i] for i in indexes]
Ys = [Y[i] for i in indexes]

Xs = np.array(Xs)
Ys = np.array(Ys)

sep = int(total * 0.1)
total_train = total-sep
total_test = sep
Xtrain = Xs[:total_train,:]
Ytrain = Ys[:total_train,:]
Xtest = Xs[total_train:,:]
Ytest = Ys[total_train:,:]

# Perceptron ----------------------------------------

class Perceptron:
  def __init__ (self):
    self.wh = np.random.random((8,4)) * 2.0 - 1.0
    self.bh = np.random.random((8,1)) * 2.0 - 1.0
    self.wo = np.random.random((3,8)) * 2.0 - 1.0
    self.bo = np.random.random((3,1)) * 2.0 - 1.0
    self.eta = 0.01
  def compute(self, x):
    x = np.reshape(x, (4,1))
    self.sh = np.dot(self.wh, x) + self.bh
    self.zh = sigmoid(self.sh)
    self.so = np.dot(self.wo, self.zh) + self.bo
    self.zo = sigmoid(self.so)
    return self.zo
  def backprop(self, X, Y):
    Err = 0.0
    total = X.shape[0]
    for i in range(total):
      x = X[i,:]
      x = np.reshape(x,(4,1))
      y_hat = Y[i,:]
      y_hat = np.reshape(y_hat,(3,1))
      y = self.compute(x)
      err = (np.sum((y-y_hat)**2))/2.0
      Err += err
      self.do = (y-y_hat) * y * (1.0 - y)
      self.dh = (np.dot(self.wo.T, self.do)) * self.zh * (1.0 - self.zh)
      self.wo -= self.eta * np.dot(self.do, self.zh.T)
      self.bo -= self.eta * self.do
      self.wh -= self.eta * np.dot(self.dh, x.T)
      self.bh -= self.eta * self.dh
    Err /= total
    return Err
  
# ------------------------------------------------------------------------

p = Perceptron()
Err = list()
for i in range(10000):
  Err.append(p.backprop(Xtrain,Ytrain))
  if not (i % 1000) or i == 0:
    print('Err=', Err[-1])

# ------------------------------------------------------------------------

Err = np.array(Err)
plt.plot(Err)

np.set_printoptions(formatter={'float': lambda x: '%+01.2f' % x})

for i in range(total_test):
  y = p.compute(Xtest[i,:])
  y_hat = Ytest[i,:]
  print(y_hat,y.T[0])
