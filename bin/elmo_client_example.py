'''
ELMo client example that makes code embedding queries.

'''

import json
import pickle
import socket
import sys
from elmo_server import *


def connect(server, port):
    """Returns a socket connected to the specified server and port number.
    
    Arguments:
        server {[type]} -- [description]
        port {[type]} -- [description]
    """
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = (server, port)
    eprint('Connecting to %s port %s' % server_address)
    sock.connect(server_address)
    return sock


def query(code, socket, options={'top_layer_only' : False, 'token_embeddings_only' : False}):
    """Performs...
    
    Arguments:
        code {list/string} -- a list of strings (mulitple queries) or a string (single query) of code tokens
        socket {a TCP/IP socket} -- a TCP/IP socket connected to the ELMo server
    """
    try:
        basestring
    except NameError:
        basestring = str
    if isinstance(code, list):
        for code_sequence in code:
            if not isinstance(code_sequence, basestring): raise ValueError
        code_sequences = [code_sequence.split() for code_sequence in code]
    elif isinstance(code, basestring):
        code_sequences = [code.split()]
    else: raise ValueError
    
    eprint('Sending code sequence "%s"' % code_sequences)
    query = json.dumps({'sequences': code_sequences, 'options': options})
    data = pickle.dumps(query)
    # eprint("Pickled code sequences: ", data)
    socket.sendall(data)
    socket.sendall(END)
    
    received_data = ''.encode()
    received = socket.recv(MAX_PACKET_SIZE)
    while not received[-len(END): ] == END:
        received_data += received
        received = socket.recv(MAX_PACKET_SIZE)
    received_data += received[: -len(END)]
    # eprint('received pickled ELMo embeddings: "%s"' % received_data)
    elmo_representations = pickle.loads(received_data)
    eprint('Received ELMo embeddings:', elmo_representations)
    

if __name__ == '__main__':
    SERVER = 'localhost'
    top_layer_only = True
    token_embeddings_only = False
    options = {'top_layer_only' : top_layer_only, 'token_embeddings_only' : token_embeddings_only}

    try:
        socket = connect('localhost', PORT)

        try:
            code = 'STD:var ID:gfm STD:= ID:require STD:( LIT:github-flavored-markdown STD:) STD:;' 
            query(code, socket, options)
            
            code = ['STD:var ID:gfm STD:= ID:require STD:( LIT:github-flavored-markdown STD:) STD:;',
                'STD:var ID:gfm STD:= ID:require STD:( LIT:github-flavored-markdown STD:) STD:;'
            ]
            query(code, socket)
            
            options[token_embeddings_only] = True
            print(options)
            query(code, socket, options)
            query(code, socket, options)
            
            # No more data, ask to close the connection
            socket.sendall(CONN_END)
        finally:
            # Either done or an error occured. Make sure to always close the connection
            eprint('Closing socket!')
            socket.close()
    except ConnectionRefusedError:
        eprint('Could not connect to server %s port %d' % (SERVER, PORT))

