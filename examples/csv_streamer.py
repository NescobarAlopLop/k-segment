import socket
from time import sleep

host = 'localhost'
port = 9990

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(1)
while True:
    print('\nListening for a client at', host, port)
    conn, addr = s.accept()
    print('\nConnected by', addr)
    try:
        print('\nReading file...\n')
        with open('/home/ge/k-segment/datasets/KO.csv') as f:
            for line in f:
                out = line.encode('utf-8')
                print('Sending line',line)
                conn.send(out)
                sleep(0.02)
            print('End Of Stream.')
    except socket.error:
        print ('Error Occured.\n\nClient disconnected.\n')
    except OSError:
        conn.close()
        conn.shutdown(socket.SHUT_RDWR)
        s.shutdown(socket.SHUT_RDWR)
        s.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)

    except KeyboardInterrupt:
        conn.close()
        conn.shutdown(socket.SHUT_RDWR)
        s.shutdown(socket.SHUT_RDWR)
        s.close()
