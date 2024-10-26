import subprocess
import select
import numpy as np
import json
import re

start_battle_path = "."

class Pokemon:
    def __init__(self, json):
        match = re.match(r"(\d+)/(\d+)", json["condition"])
        self.stats = [int(match.group(2)), json["stats"]["atk"], json["stats"]["def"], json["stats"]["spa"], json["stats"]["spd"], json["stats"]["spe"]]
        self.current_hp = match.group(1)
        self.ability = json["ability"]
        self.item = json["item"]
        self.moves = json["moves"]

class Team:
    def __init__(self, json_val):
        "Initialize team from json provided by Showdown"
        self.actor = json_val["side"]["id"]
        self.name = json_val["side"]["name"]
        self.pokemon = [Pokemon(json_poke) for json_poke in json_val["side"]["pokemon"]]

class Battler:
    #DFA which handles text output
    states = {'start', 'sideupdate', 'p1', 'p2', 'update', 'end'}

    transitions = {
        'start': {'sideupdate': 'sideupdate', 'await': 'await'},
        'sideupdate': {'update': 'update', 'sideupdate': 'sideupdate', 'p1': 'p1', 'p2': 'p2', 'await': 'await'},
        'p1': {'update': 'update', 'sideupdate': 'sideupdate', '': 'p1', 'await': 'await'},
        'p2': {'update': 'update', 'sideupdate': 'sideupdate', '': 'p2', 'await': 'await'},
        'update': {'update': 'update', 'sideupdate': 'sideupdate', '': 'update', 'await': 'await'},
    }

    def __init__(self):
        self.current_state = 'start'
        self.commands = {
            'request': self.request,
            'turn': self.turn
        }

        self.team1 = None
        self.team2 = None

        self.p1move = False
        self.p2move = False

        self.process = subprocess.Popen(
            ['node', f'{start_battle_path}/start_battle.js'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        self.send_command('>start {"formatid":"gen9randombattle", "p1": {"name":"BOT_1"}, "p2": {"name":"BOT_2"}}')

        # self.receive_output()
        # print("received output.")

        # print(self.team1)

        # self.send_command('>p1 move 1')
        # self.send_command('>p2 switch 3')

        # self.receive_output()
        # print("received output.")

        # self.send_command('>p1 move 3')
        # self.send_command('>p2 move 2')

        # lines = self.receive_output()
        # for line in lines:
            # print(line)

    def send_command(self, command):
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

    def receive_output(self):
        lines = []
        while self.current_state != 'await':
            output = self.process.stdout.readline().strip().split('|')
            self.current_state = Battler.transitions[self.current_state][output[0]]
            if output[0] == '' and output[1] in self.commands:
                self.commands[output[1]](output)

        self.current_state = 'start'

    #functions for commands
    def request(self, output):
        if self.current_state == 'p1':
            self.team1 = Team(json.loads(output[2]))
            self.p1move = True
        else:
            self.team2 = Team(json.loads(output[2]))
            self.p2move = True

    def turn(self, output):
        self.turn = output[2]




battler = Battler()
