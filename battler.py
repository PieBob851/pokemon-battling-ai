import subprocess
import select
import numpy as np
import json
import re

start_battle_path = "."

class Pokemon:
    def __init__(self, json):
        self.name = json['details']
        match = re.match(r"(\d+)/(\d+)", json["condition"])
        self.stats = [int(match.group(2)), json["stats"]["atk"], json["stats"]["def"], json["stats"]["spa"], json["stats"]["spd"], json["stats"]["spe"]]
        self.current_hp = match.group(1)
        self.ability = json["ability"]
        self.item = json["item"]
        self.moves = json["moves"]
        
    def print_short_info(self):
        print(self.name)
        stat_names = ["hp", "atk", "def", "spa", "spd", "spe"]
        print(f"  {stat_names[0]}: {self.current_hp}/{self.stats[0]}")
        for i in range(5):
            print(f"  {stat_names[i + 1]}: {self.stats[i + 1]}")
        print(f"  {self.ability}")
        print(f"  {self.item}")
        print("  moves:")
        for move in self.moves:
            print(f"    {move}")

class Team:
    def __init__(self, json_val=None):
        "Initialize team from json provided by Showdown"
        self.actor = json_val["side"]["id"]
        self.name = json_val["side"]["name"]
        self.pokemon = [Pokemon(json_poke) for json_poke in json_val["side"]["pokemon"]]
        
    def print_short_info(self):
        self.pokemon[0].print_short_info()
        print(f"{self.pokemon[1].name}, {self.pokemon[2].name}, {self.pokemon[3].name}, {self.pokemon[4].name}, {self.pokemon[5].name}")
        
class Actor:
    def __init__(self, team: Team):
        self.team = team
    
    def pick_move(self, knowledge) -> str:
        raise NotImplementedError("Subclasses should implement this method.")

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

    def __init__(self, actor1: Actor, actor2: Actor):
        self.current_state = 'start'
        self.commands = {
            'request': self.request,
            'turn': self.turn
        }

        self.actor1 = actor1
        self.actor2 = actor2

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
        
    def make_moves(self):
        if self.p1move:
            knowledge = {"field": None, "opponent":self.actor2.team}
            move = f">p1 {self.actor1.pick_move(knowledge)}"
            
        if self.p2move:
            knowledge = {"field": None, "opponent":self.actor1.team}
            move = f">p1 {self.actor2.pick_move(knowledge)}"
        
        self.receive_output()
            

    #functions for commands
    def request(self, output):
        if self.current_state == 'p1':
            self.actor1.team = Team(json.loads(output[2]))
            self.p1move = True
        else:
            self.actor2.team = Team(json.loads(output[2]))
            self.p2move = True

    def turn(self, output):
        self.turn = output[2]