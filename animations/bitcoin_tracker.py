from datetime import datetime
import numpy as np
from websocket import create_connection
import json

from tree_animator import TreeAnimator

WS_OPTIONS = {'origin': 'https://exchange.blockchain.com'}
WS_URL = 'wss://ws.prod.blockchain.info/mercury-gateway/v1/ws'
API_SECRET = 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkFQSSJ9.eyJhdWQiOiJtZXJjdXJ5IiwidWlkIjoiNzlkNTg0ZjMtODNhYS00ZjQ0LWE3ZTEtMjQ5YTFhNTQxMGQ1IiwiaXNzIjoiYmxvY2tjaGFpbiIsInJkbyI6dHJ1ZSwiaWF0IjoxNjM4NzM1MzEwLCJqdGkiOiJiYzMzNjQ3MC00ODgzLTQ1NzgtYTQwYy1jYzRlNmE3NWQ1MTgiLCJzZXEiOjQ5NDgxMDIsIndkbCI6ZmFsc2V9.H9rewOLUNhKwvfFJ9iCIkBQXhDlSNivkxOw8Ep7DWyBMIQX9wdTA11lUvJD9JHm2pRrjH5iaVB6/eOtsTTP3zLQ='
# ^^ never do this in real life lol

COLORS = {
    'GREEN': [112, 130, 56],
    'VERY_GREEN': [0, 255, 0],
    'RED': [255, 87, 51],
    'VERY_RED': [255, 0, 0]
}


class BTCAnimator(TreeAnimator):
    def _get_btc_price(self):
        result = json.loads(self.connection.recv())

        try:
            price = round(result['mark_price'])

        except KeyError:
            # mark_price isn't always there for whatever reason
            price = self.last_sig_price

        return price

    def initialize_animation(self):
        # create connection to web socket and authentication connection
        self.last_sig_price = 0  # last significant price
        self.sig_change = 5  # difference in price +/- necessary to change lights
        self.current_light_color = 'GREEN'  # always start green cause BTC is going to the moon! Plus it's easier
        self.same_color_iterations = 0  # do animations depending how long we're the same color
        self.num_light_rows = 9  # iterations start at 0
        self.connection = create_connection(WS_URL, **WS_OPTIONS)

        data = {'token': API_SECRET, 'action': 'subscribe', 'channel': 'auth'}
        self.connection.send(json.dumps(data))
        result = self.connection.recv()
        print(f'Auth result: {result}')

        # subscribe to BTC ticker
        data = {'action': 'subscribe', 'channel': 'ticker', 'symbol': 'BTC-USD'}
        self.connection.send(json.dumps(data))

        self.last_sig_price = self._get_btc_price()


    def calculate_colors(self, xyz_coords, start_time):
        # this function gets called every few milliseconds, and it's purpose is to return the colors we want each light to be.
        current_price = self._get_btc_price()
        difference_in_price = abs(self.last_sig_price - current_price)

        print('\n')
        print(f'current_price: {current_price}')
        print(f'self.last_sig_price: {self.last_sig_price}')
        print(f'difference_in_price: {difference_in_price}')

        new_color = self.current_light_color  # hacky default to current colour, no change is kind of a change right?

        if difference_in_price >= self.sig_change:
            if current_price < self.last_sig_price:
                print('Price is down! Turn the tree red and cry :\'(')
                new_color = 'RED'
            else:
                print('Price is up! To the moon baby!!!!')
                new_color = 'GREEN'

            self.last_sig_price = current_price

        if new_color == self.current_light_color and self.same_color_iterations < self.num_light_rows:
            self.same_color_iterations += 1
        else:
            self.same_color_iterations = 0

        self.current_light_color = new_color

        print(self.current_light_color)
        print(self.same_color_iterations)

        chunk_size = int(self.NUM_LIGHTS / (self.num_light_rows + 1))
        lower_boundary = chunk_size * self.same_color_iterations
        upper_boundary = (chunk_size * self.same_color_iterations) + chunk_size

        print(chunk_size)
        print(lower_boundary)
        print(upper_boundary)

        # This function is expected to return colors. colors are in RGB and in the range 0-255, for example black is [0,0,0], white is [255,255,255], red is [255,0,0]
        colors = np.full((self.NUM_LIGHTS, 3), COLORS[self.current_light_color], dtype=np.uint8)

        i = 1
        for color in colors:
            if i >= lower_boundary and i < upper_boundary:
                colors[i] = COLORS[f'VERY_{self.current_light_color}']

            i += 1

        return colors


if __name__ == '__main__':
    anim = BTCAnimator()

    anim.animation_loop()
