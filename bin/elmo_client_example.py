'''
ELMo client example that makes code embedding queries.

'''

import pickle
import socket
import sys
from elmo_server import *


def query(code):
    """Performs...
    
    Arguments:
        code {list/string} -- [a list of strings (mulitple queries) or a string (single query) of code tokens]
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
    
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('localhost', PORT)
    eprint('connecting to %s port %s' % server_address)
    sock.connect(server_address)

    try:
        eprint('Sending code sequence "%s"' % code_sequences)
        data = pickle.dumps(code_sequences)
        eprint("Pickled code sequences: ", data)
        sock.sendall(data)
        sock.sendall(END)

        # Look for the response
        amount_received = 0
        amount_expected = len(data)
        
        received_data = ''.encode()
        received = sock.recv(MAX_PACKET_SIZE)
        while not received.decode().endswith(END.decode()):
            received_data += received
            received = sock.recv(MAX_PACKET_SIZE)
        received_data += received[:-len(END)]
        eprint('received pickled ELMo embeddings: "%s"' % received_data)
        elmo_representations = pickle.loads(received_data)
        eprint('Received ELMo embeddings:', elmo_representations)
    finally:
        eprint('Closing socket!')
        sock.close()

if __name__ == '__main__':
    query('STD:var ID:gfm STD:= ID:require STD:( LIT:github-flavored-markdown STD:) STD:;')
    code = ['STD:var ID:gfm STD:= ID:require STD:( LIT:github-flavored-markdown STD:) STD:;',
        'STD:var ID:gfm STD:= ID:require STD:( LIT:github-flavored-markdown STD:) STD:;'
    ]
    query(code)
