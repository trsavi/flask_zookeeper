from kazoo.client import KazooClient,KazooState

import json
import array
import pickle
import random



def my_listener(state):
  if state == KazooState.LOST:
    # Register somewhere that the session was lost
    print('Connection lost !!')
  elif state == KazooState.SUSPENDED:
    # Handle being disconnected from Zookeeper
    print('Connection suspended !!')
  else:
    # Handle being connected/reconnected to Zookeeper
    print('Connected !!')


def init_zoo():
  zk = KazooClient(hosts='127.0.0.1:2181')
  zk.start()
  return zk


def start_kazoo():
   
  zk = KazooClient(hosts='127.0.0.1:2181')
  zk.stop()
  zk.start()
  zk.add_listener(my_listener)



  IRIS_MODEL_KNN = pickle.dumps(pickle.load( open( "iris_knn.pkl", "rb" ) ))
  IRIS_MODEL_TREE = pickle.dumps(pickle.load( open( "iris_tree.pkl", "rb" ) ))
  IRIS_MODEL_NAIVE = pickle.dumps(pickle.load( open( "iris_naive.pkl", "rb" ) ))

  models = pickle.dumps({1: IRIS_MODEL_KNN, 2:IRIS_MODEL_TREE})

  try:
    zk.ensure_path("/my/")

    # Create a node with data
    zk.create("/my/node1", models)

    # Create a node with data
    zk.create("/my/node2", models)

    # Create a node with data
    zk.create("/my/node3", models)
    zk.add_listener(my_listener)
    return zk


  except:
    zk.delete("/my/", recursive=True)
    zk.add_listener(my_listener)
    print('Error!')
    #zk.stop()




def predict(number, zk):

  nodes = ['node1', 'node2', 'node3']

  node = random.choice(nodes)

  data, stat = zk.get("/my/" + node)
  if (len(nodes)-number)<0:
    return "Node not found!", 0
  else:
    model = pickle.loads(pickle.loads(data)[number])
    return model, node

def put_model(number, model, zk):

  data, stat = zk.get("/my/node1")
  models = pickle.loads(data)
  models[number] = model
  models_zoo = pickle.dumps(models)
  zk.set("/my/node1", models_zoo)
  zk.set("/my/node2", models_zoo)
  zk.set("/my/node3", models_zoo)

  return models_zoo



# List the children

#children = zk.get_children("/my/")
#print("There are %s children with names %s" % (len(children), children))


def delete_nodes(zk):
  zk.delete("/my/", recursive=True)


  






