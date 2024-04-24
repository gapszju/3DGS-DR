
import socket

host = "127.0.0.1"
port = 6009


conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    global host, port, conn
    host = wish_host
    port = wish_port
    #conn.settimeout(1)

def connect():
    global host, port
    ret = conn.connect_ex((host, port))
    return ret == 0

def read():
    global conn
    len_buf = conn.recv(32)
    if len_buf == None:
      return None
    length = int.from_bytes(len_buf, 'little')
    buf = b''
    while length:
      newbuf = conn.recv(length)
      if newbuf == None:
        print('Error: incomplete msg')
        break
      buf += newbuf
      length -= len(newbuf)
    return buf

def send(byte_data):
    global conn
    data_len = len(byte_data).to_bytes(32, 'little')
    conn.sendall(data_len)
    conn.sendall(byte_data)