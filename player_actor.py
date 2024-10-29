from battler import Pokemon, Team, Actor, Battler

class PlayerActor(Actor):
    def __init__(self, team: Team):
        super().__init__(team)
    
    def pick_move(self, knowledge) -> str:
        self.team.print_short_info()
        knowledge['opponent'].print_short_info()
        move_choice = input()
        return move_choice
        
actor1 = PlayerActor(None)
actor2 = PlayerActor(None)

battler = Battler(actor1, actor2)

iteration = 0
while battler.current_state != 'end':
    battler.make_moves()
    print("iteration:", iteration)
    iteration += 1