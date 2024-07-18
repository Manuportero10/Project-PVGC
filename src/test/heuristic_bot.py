import numpy as np
import random
from poke_env.data import GenData
from poke_env import Player
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.pokemon import Pokemon


class HeuristicPlayer(Player):

    def __init__(self, training_mode:bool=None,par_stats=None, par_typing=None, par_hp=None, par_status=None, par_weather=None,**kwargs):
        super().__init__(**kwargs)
        self.current_turn : int = 0
        self.training_mode = training_mode
        self.par_stats = 1.113727387000199
        self.par_typing = 0.9582001361390107
        self.par_hp = 1.6876925933704248
        self.par_status = 1.2996384594734227
        self.par_weather = 1.8776382680824122
        self.last_move = None

        if self.training_mode:
            self.par_stats = random.uniform(1,2)
            self.par_typing = random.uniform(1,2)
            self.par_hp = random.uniform(1,2)
            self.par_status = random.uniform(1,2)
            self.par_weather = random.uniform(1,2)
 

    def modify_parameter(self,parameter):
        return parameter + random.uniform(-0.05,0.05)
    
    def adjust_parameters(self, best:bool,dicc_par : dict = None):
        if best:
            self.par_stats = self.modify_parameter(self.par_stats)
            self.par_typing = self.modify_parameter(self.par_typing)
            self.par_hp = self.modify_parameter(self.par_hp)
            self.par_status = self.modify_parameter(self.par_status)
            self.par_weather = self.modify_parameter(self.par_weather)
        else:
            self.par_stats = self.modify_parameter(dicc_par['par_stats'])
            self.par_typing = self.modify_parameter(dicc_par['par_typing'])
            self.par_hp = self.modify_parameter(dicc_par['par_hp'])
            self.par_status = self.modify_parameter(dicc_par['par_status'])
            self.par_weather = self.modify_parameter(dicc_par['par_weather'])

    def choose_move(self, battle):
        debug : bool = False
        # If the player can attack, it will
        if battle.available_moves:
            # Finding out which move tend to be the best
            # 1. Find out the move that does the most damage according to the battle state and a heuristic
            # 2. Otherwise, change the active pokemon to one that is not weak to the opponent's active pokemon
            my_mon = battle.active_pokemon
            opp_mon = battle.opponent_active_pokemon
            contextual_score : float = 0.0

            if self.is_faster(battle,my_mon, opp_mon):
                    contextual_score += 2

            contextual_score += self.par_stats * self.stats_balance(my_mon, opp_mon, debug)
            contextual_score += self.par_typing * self.typing_advantage(my_mon, opp_mon, debug)
            contextual_score += self.par_hp * (my_mon.current_hp_fraction - opp_mon.current_hp_fraction)
            contextual_score += self.par_status * self.status_condition(my_mon, opp_mon, debug)
            contextual_score += self.par_weather * self.weather_condition(my_mon, opp_mon, battle, debug)

            if debug:
                print(f'Turn {self.current_turn}\nContextual score: {contextual_score}\n')
           
            if contextual_score <= -1 and battle.available_switches != [] and not isinstance(self.last_move,Pokemon): # If the score is negative, we change the active pokemon
                best_move = self.best_switch_action(battle.available_switches, opp_mon, debug)
                self.last_move = best_move

            if opp_mon.terastallized:
                # Checking the best move to use - Also we need to check the ability of the pokemon
                best_move = self.choose_best_move(battle, opp_mon, True)
                self.last_move = best_move
            else:
                best_move = self.choose_best_move(battle, opp_mon, False)
                self.last_move = best_move
                
            
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
        my_mon_stats = my_mon.stats
        opp_mon_boosts = opp_mon.boosts
        balance : float = 0.0

        for stat in my_mon_boosts:
            balance += my_mon_boosts[stat] - opp_mon_boosts[stat]

        # If my mon has more atk than spa (physical attacker)
        if my_mon_stats['atk'] >= my_mon_stats['spa']:
            balance += opp_mon_boosts['spd']

        # If my mon has more spa than atk (special attacker)
        if my_mon_stats['spa'] >= my_mon_stats['atk']:
            balance += opp_mon_boosts['def']
            

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
    
    def moves_advantage(self, my_mon, opp_mon, debug):
        '''
            Checks the advantage of the moves of the pokemon
        '''
        my_moves = my_mon.moves
        advantage : float = 0.0

        for move in my_moves.values():
            if move:
                advantage += move.base_power/100 * move.type.damage_multiplier(*opp_mon.types,type_chart=GenData.from_gen(8).type_chart)
        if debug:
            print(f'Moves advantage: {advantage}')

        return advantage
    
    def best_switch_action(self, available_switches : list, opp_mon, debug : bool):
        '''
            Returns the best switch action to take with the aviable mons
            according to the type advantage and the moves of the pokemon waiting for the switch
        '''
        best_score = -np.inf
        best_mon = None

        for mons in available_switches:
            actual_score = self.typing_advantage(mons, opp_mon,False) + self.moves_advantage(mons, opp_mon, debug)
            if actual_score > best_score:
                best_score = actual_score
                best_mon = mons

        return best_mon
    
    def increment_turn(self):
        self.current_turn += 1

    def status_condition(self, my_mon, opp_mon, debug):
        '''
            Checks the status condition of the pokemon
        '''
        my_status = my_mon.status
        my_stats = my_mon.stats
        opp_status = opp_mon.status
        status_condition : float = 0.0

        if my_status:
            if my_status == 1 or my_stats == 5: # If the pokemon is burned or poisoned
                status_condition -= 1
                if my_stats['atk'] > my_stats['spa']:
                    status_condition -= 1
            elif my_status == 7: # If the pokemon has toxic
                status_condition -= 1*my_mon.status_counter
            elif my_status == 4: # if the pokemon is paralyzed
                status_condition -= 1.5
            elif my_status == 6: # if the pokemon is asleep
                status_condition -= 1.5*1/my_mon.status_counter+1
                
        if opp_status:
            if opp_status == 1 or opp_status == 5: # If the pokemon is burned or poisoned
                status_condition += 1
                if opp_mon.base_stats['atk'] > opp_mon.base_stats['spa']:
                    status_condition += 1
            elif opp_status == 7: # If the pokemon has toxic
                opp_status += 1*opp_mon.status_counter
            elif opp_status == 4: # if the pokemon is paralyzed
                status_condition += 1.5
            elif opp_status == 6: # if the pokemon is asleep
                status_condition += 1.5*1/opp_mon.status_counter+1

        if debug:
            print(f"Status condition: {status_condition}")

        return status_condition
    
    def weather_condition(self, my_mon, opp_mon, battle, debug):
        weather_state = battle.weather
        wheather_condition : float = 0.0
        actual_weather : str = 'None'

        for weather in weather_state:
            # Checking whether the weather is sunny
            if int(weather.value) == 9:
                actual_weather = 'Sunny'
                if self.check_desire_type(my_mon,PokemonType.FIRE): # Checking if the pokemon is a fire type
                    wheather_condition += 1
                    if PokemonType.FIRE.damage_multiplier(*opp_mon.types, type_chart=GenData.from_gen(8).type_chart) > 1:
                        wheather_condition += 0.5

                elif self.check_desire_type(opp_mon,PokemonType.FIRE):
                    wheather_condition -= 1
                    if PokemonType.FIRE.damage_multiplier(*my_mon.types, type_chart=GenData.from_gen(8).type_chart) > 1:
                        wheather_condition -= 0.5

                if self.check_desire_type(my_mon,PokemonType.WATER): # Checking if the pokemon is a water type
                    wheather_condition -= 0.5
                elif self.check_desire_type(opp_mon,PokemonType.WATER):
                    wheather_condition += 0.5
            # Checking whether the weather is rainy
            elif int(weather.value) == 6:
                actual_weather = 'Rainy'
                if self.check_desire_type(my_mon,PokemonType.WATER): # Checking if the pokemon is a water type
                    wheather_condition += 1
                    if PokemonType.WATER.damage_multiplier(*opp_mon.types, type_chart=GenData.from_gen(8).type_chart) > 1:
                        wheather_condition += 0.5

                elif self.check_desire_type(opp_mon,PokemonType.WATER):
                    wheather_condition -= 1
                    if PokemonType.WATER.damage_multiplier(*my_mon.types, type_chart=GenData.from_gen(8).type_chart) > 1:
                        wheather_condition -= 0.5

                if self.check_desire_type(my_mon,PokemonType.FIRE): # Checking if the pokemon is a fire type
                    wheather_condition -= 0.5
                elif self.check_desire_type(opp_mon,PokemonType.FIRE):
                    wheather_condition += 0.5
            # Checking whether the weather is sandstorm
            elif int(weather.value) == 7:
                actual_weather = 'Sandstorm'
                if self.check_desire_type(my_mon,PokemonType.ROCK): # Checking if the pokemon is a rock type
                    wheather_condition += 1
                    if PokemonType.ROCK.damage_multiplier(*opp_mon.types, type_chart=GenData.from_gen(8).type_chart) > 1:
                        wheather_condition += 0.5

                elif self.check_desire_type(opp_mon,PokemonType.ROCK):
                    wheather_condition -= 1
                    if PokemonType.ROCK.damage_multiplier(*my_mon.types, type_chart=GenData.from_gen(8).type_chart) > 1:
                        wheather_condition -= 0.5
            # Checking wheter the weather is hail or snowy
            elif int(weather.value) == 4 or int(weather.value) == 8:
                actual_weather = 'Hail/Snowy'
                if self.check_desire_type(my_mon,PokemonType.ICE): # Checking if the pokemon is an ice type
                    wheather_condition += 1
                    if PokemonType.ICE.damage_multiplier(*opp_mon.types, type_chart=GenData.from_gen(8).type_chart) > 1:
                        wheather_condition += 0.5
                elif self.check_desire_type(opp_mon,PokemonType.ICE):
                    wheather_condition -= 1
                    if PokemonType.ICE.damage_multiplier(*my_mon.types, type_chart=GenData.from_gen(8).type_chart) > 1:
                        wheather_condition -= 0.5

        if debug:
            print(f'Weather condition score: {wheather_condition} in {actual_weather}')    

        return wheather_condition
    
    def check_desire_type(self, mon, type_id):

        if mon.terastallized:
            if mon.tera_type == type_id:
                return True
        elif mon.types[0] == type_id or mon.types[1] == type_id:
            return True
        
        return False
    
    def choose_best_move(self, battle, opp_mon, terastallized : bool):
        # Checking if the mon has al least one move which affects neutral against the opponent.
        # Otherwise, we will change the active pokemon
        best_move = None
        best_score = -np.inf

        if terastallized:
            if all(move.type.damage_multiplier(opp_mon.tera_type,type_chart=GenData.from_gen(8).type_chart) < 1 for move in battle.available_moves) and battle.available_switches != [] and not isinstance(self.last_move,Pokemon):
                return self.best_switch_action(battle.available_switches, opp_mon, False)
            
            for move in battle.available_moves:
                if move.base_power/100 * move.type.damage_multiplier(opp_mon.tera_type,type_chart=GenData.from_gen(8).type_chart) >= best_score:
                    best_score = move.base_power/100 * move.type.damage_multiplier(*opp_mon.types,type_chart=GenData.from_gen(8).type_chart)
                    best_move = move
        else:
            if all(move.type.damage_multiplier(*opp_mon.types,type_chart=GenData.from_gen(8).type_chart) < 1 for move in battle.available_moves) and battle.available_switches != [] and not isinstance(self.last_move,Pokemon):
                return self.best_switch_action(battle.available_switches, opp_mon, False)
            
            for move in battle.available_moves:
                if move.base_power/100 * move.type.damage_multiplier(*opp_mon.types,type_chart=GenData.from_gen(8).type_chart) >= best_score:
                    best_score = move.base_power/100 * move.type.damage_multiplier(*opp_mon.types,type_chart=GenData.from_gen(8).type_chart)
                    best_move = move

        return best_move
