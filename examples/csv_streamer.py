import socket
from time import sleep
import sys
from typing import Union


PATH = '/home/ge/k-segment/datasets/KO.csv'
HOST = 'localhost'
PORT = 9990


def setup_socket(host_address: str, host_port: Union[str, int]):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.bind((host_address, host_port))
    soc.listen(1)
    return soc


def destroy_socket(conn, soc) -> None:
    conn.close()
    conn.shutdown(socket.SHUT_RDWR)
    soc.shutdown(socket.SHUT_RDWR)
    soc.close()


def send_line_via_connection(line, conn) -> None:
    out = line.encode('utf-8')
    conn.send(out)


def infinite_stream_txt_file_over_socket(file_path: str, soc) -> None:
    while True:
        print('\nListening for a client at', host, port)
        conn, addr = soc.accept()
        print('\nConnected by', addr)
        try:
            print('\nReading file...\n')
            with open(file_path) as f:
                for line in f:
                    send_line_via_connection(line, conn)
                    print('Sending line', line)
                    sleep(0.02)
                print('End Of Stream.')
        except socket.error:
            print('Error Occurred.\n\nClient disconnected.\n')
        except OSError:
            destroy_socket(conn, soc)
            soc = setup_socket(host, port)
        except KeyboardInterrupt:
            destroy_socket(conn, soc)


if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) == 4:
        path = sys.argv[1]
        host = sys.argv[2]
        port = sys.argv[3]
    else:
        print("How to use?")
        print("\tif no parameters provided default valeus will be used.")
        print("\tdefault:\n\tpath {}\n\thost {}\n\tport {}".format(PATH, HOST, PORT))
        print("Example call:\n\tpython3 csv_streamer.py path/to/file.txt 127.0.0.1 9999")
        path = PATH
        host = HOST
        port = PORT
    s = setup_socket(host, port)
    infinite_stream_txt_file_over_socket(path, s)
