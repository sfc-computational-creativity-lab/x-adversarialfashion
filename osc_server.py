import threading
from pythonosc import dispatcher
from pythonosc import osc_server


class OscServer(object):
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.is_done = False
        self.img_path = ''

    def activate(self, address='/done'):
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map(address, self._done)

        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
        print('Serving on {}'.format(self.server.server_address))

        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.setDaemon(True)
        self.server_thread.start()

    def shutdown(self):
        self.server.shutdown()

    def _done(self, unused_addr, img_path):
        self.is_done = True
        self.img_path = img_path
