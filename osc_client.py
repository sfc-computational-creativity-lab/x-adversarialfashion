from pythonosc import osc_message_builder
from pythonosc import udp_client


class OscClient(object):
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.client = udp_client.UDPClient(self.ip, self.port)
        print('ip: {}, port: {}'.format(self.ip, self.port))

    def send_string(self, address, messages):
        msg = osc_message_builder.OscMessageBuilder(address=address)
        for m in messages:
            msg.add_arg(m)
        msg = msg.build()
        self.client.send(msg)
