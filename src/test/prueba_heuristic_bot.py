import numpy as np
from poke_env.data import GenData
from poke_env import Player

class HeuristicPlayer(Player):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.current_turn : int = 0


    def choose_move(self, battle):
        debug : bool = False
        # If the player can attack, it will
        if battle.available_moves:
            # Finding out which move tend to be the best
            # 1. Find out the move that does the most damage according to the battle state and a heuristic
            # 2. Otherwise, change the active pokemon to one that is not weak to the opponent's active pokemon
            my_mon = battle.active_pokemon
            opp_mon = battle.opponent_active_pokemon
            contextual_score : int = 0

            if self.is_faster(battle,my_mon, opp_mon):
                    contextual_score += 2

            contextual_score += self.stats_balance(my_mon, opp_mon, debug)
            contextual_score += self.typing_advantage(my_mon, opp_mon, debug)
            if debug:
                print(f'Turn {self.current_turn}\nContextual score: {contextual_score}\n')
           
            if contextual_score <= -1.5 and battle.available_switches != []: # If the score is negative, we change the active pokemon
                return self.create_order(order=self.best_switch_action(battle.available_switches, opp_mon))
    
            # Checking the best move to use
            best_move = max(battle.available_moves, key=lambda move: move.base_power/100 * move.type.damage_multiplier(opp_mon.type_1,opp_mon.type_2,type_chart=GenData.from_gen(8).type_chart))
            return self.create_order(best_move)
                
        else:
            return self.choose_random_move(battle)
        
    def is_faster(self,battle,my_mon, opp_mon) -> bool:
        field_state = battle.fields
        trick_room : bool = False

        # Checking whether trick room is active
        for field in field_state:
            if int(field.value) == 11 and int(field_state[field])+3 > self.current_turn:
                trick_room = True
                break

        self.increment_turn()
        if trick_room:
            return my_mon.base_stats['spe'] <= opp_mon.base_stats['spe']
        
        return my_mon.base_stats['spe'] >= opp_mon.base_stats['spe']
    
    def stats_balance(self, my_mon, opp_mon, debug):
        '''
            Checks the balance between the buffs and deffus between the two pokemons
        '''
        my_mon_boosts = my_mon.boosts
        opp_mon_boosts = opp_mon.boosts
        balance : int = 0

        for stat in my_mon_boosts:
            balance += my_mon_boosts[stat] - opp_mon_boosts[stat]

        if debug:
            print(f"\nbalance of stats: {balance}")

        return balance
    
    def typing_advantage(self, my_mon, opp_mon, debug):
        # We evaluate the performance on mon_a against mon_b as its type advantage
        a_on_b = b_on_a = -np.inf
        for type_ in my_mon.types:
            if type_:
                a_on_b = max(
                    a_on_b,
                    type_.damage_multiplier(
                        *opp_mon.types, type_chart=GenData.from_gen(8).type_chart
                    ),
                )
        # We do the same for mon_b over mon_a
        for type_ in opp_mon.types:
            if type_:
                b_on_a = max(
                    b_on_a,
                    type_.damage_multiplier(
                        *my_mon.types, type_chart=GenData.from_gen(8).type_chart
                    ),
                )
        # Our performance metric is the different between the two
        if debug:
            print(f'performance type advantage: {a_on_b - b_on_a}')
        return a_on_b - b_on_a
    
    def best_switch_action(self, available_switches : list, opp_mon):
        '''
            Returns the best switch action to take with the aviable mons
        '''
        best_score = -np.inf
        best_mon = None

        for mons in available_switches:
            actual_score = self.typing_advantage(mons, opp_mon,False)
            if actual_score > best_score:
                best_score = actual_score
                best_mon = mons

        return best_mon
    
    def increment_turn(self):
        self.current_turn += 1
                