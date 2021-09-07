import tf_encrypted as tfe


@tfe.local_computation('server1')
def provide_input():
    # normal TensorFlow operations can be run locally
    # as part of defining a private input, in this
    # case on the machine of the input provider
    return tf.ones(shape=(5, 10))

def start_slave(cluster_config_file):
  print("Starting bob...")
  remote_config = tfe.RemoteConfig.load(cluster_config_file)
  tfe.set_config(remote_config)
  tfe.set_protocol(tfe.protocol.ABY3())
  players = remote_config.players
  bob = remote_config.server(players[1].name)
  print("server_name = ", players[1].name)
  bob.join()

if __name__ == "__main__":
  start_slave("config.json")