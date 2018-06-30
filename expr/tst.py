import os
import socket


'''func
'''


'''expr
'''
def client():
  tst_str = "[" \
        + "{\"hyp\":\"a white bird some sand water rocks and grass\", \"ref\": [\"a white bird sitting on top of a lake\",\"a bird cleans itself next to a body of water\",\"a large good standing in the dirt near the ocean\",\"a swan floating along the water nibbling at itself\"], \"id\": \"1\"}, " \
        + "{\"hyp\": \"a white bird some sand water rocks and grass\",\"ref\":[\"a white bird sitting on top of a lake\",\"a bird cleans itself next to a body of water\",\"a large good standing in the dirt near the ocean\",\"a swan floating along the water nibbling at itself\"], \"id\": \"2\"}]";

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.connect(('127.0.0.1', 9090))

  sock.sendall(tst_str + '\n') 
  f = sock.makefile()
  for line in f:
    line = line.strip()
    print line


if __name__ == '__main__':
  client()
