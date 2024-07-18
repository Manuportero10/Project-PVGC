import sys
import asyncio
import textwrap
import time
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from poke_env import AccountConfiguration,Player,RandomPlayer
from max_bot import MaxDamagePlayer
from heuristic_bot import HeuristicPlayer
from simpleRL_bot import SimpleRLPlayer
from gymnasium.utils.env_checker import check_env


#battle_format="gen9vgc2024regg"

first_player = RandomPlayer(start_timer_on_battle_start=True)
second_player = RandomPlayer(start_timer_on_battle_start=True)
max_bot = MaxDamagePlayer(start_timer_on_battle_start=True)
_heuristic_bot = HeuristicPlayer(battle_format="gen9randombattle",
                                 start_timer_on_battle_start=True,
                                 training_mode=False
                                 )


def main():
    option : int = 0
    try:
        while option != range(1,6):
            print(
                textwrap.dedent(
                    """
                    Select the option you want to test:
                    1. Random player vs Random player
                    2. Max damage player vs Random player
                    3. Heuristic player vs Random player
                    4. Heuristic player vs Max damage player
                    5. Checking the enviroment of the SimpleRLPlayer
                    6. Heuristic player vs Max damage player finding the best configuration
                    0. Exit
                    """
                )
            )            
            
            option = int(input("Enter the option you want to test: "))

            if option == 0:
                break

            n_battles = int(input('Enter the number of battles you want to play: '))
            
            if option == 1:
                # Game between two players
                asyncio.run(create_battle(first_player, second_player,n_battles))
            elif option == 2:
                asyncio.run(create_battle(max_bot, second_player,n_battles))
            elif option == 3:
                asyncio.run(create_battle(heuristic_bot, second_player,n_battles))
            elif option == 4:
                asyncio.run(create_battle(_heuristic_bot, max_bot,n_battles))
            elif option == 5:
                try:
                    rl_bot = SimpleRLPlayer(opponent=_heuristic_bot, start_challenging=True)
                    test_env = rl_bot
                    check_env(test_env,skip_render_check=True)
                    test_env.close()
                    print('Enviroment is correct')
                except Exception as e:
                    print(f"Error: {e.__str__()}")
                    sys.exit(1)
            elif option == 6:
                epochs = int(input('Enter the number of iterations you want to play: '))
                best_win_rate : int = -1
                last_win_rate : int = 0
                acumulative_wins : int = 0
                heuristic_bot = HeuristicPlayer(start_timer_on_battle_start=True,training_mode=True)
                list_wins : list = []
                
                # Game between two players
                for _ in range(epochs):
                    asyncio.run(create_battle(heuristic_bot, max_bot,n_battles))
                    last_win_rate = heuristic_bot.n_won_battles - acumulative_wins
                    acumulative_wins = heuristic_bot.n_won_battles
                    list_wins.append(last_win_rate)
                    print(f'Epoch {_+1} completed\nWin rate: {last_win_rate}')

                    if last_win_rate > best_win_rate: #Best performance
                        best_win_rate = last_win_rate
                        heuristic_bot.adjust_parameters(True) #Trying to improve the performance
                        dicc_par = save_parameters(heuristic_bot)
                    else:
                        heuristic_bot.adjust_parameters(False,dicc_par= dicc_par) #Reverting the changes

                print(f'Training completed:\nBest win rate: {best_win_rate}\nBest parameters: {dicc_par}')
                print(f'Mean of all win rates: {np.mean(list_wins)}')
                plt.plot(list_wins)
                plt.show()
            else:
                print('Error: Invalid option')
    except Exception as e:
        print(f'Error: {e}\nMONDONGO\n')
        sys.exit(1)


async def create_battle(player1, player2, n_battles):
    start_time = time.time()
    await player1.battle_against(player2, n_battles=n_battles)
    end_time = time.time()
    print(
        f"Player {player1.username} won {player1.n_won_battles} out of {player1.n_finished_battles} played\nTime elapsed with {n_battles} battles: {end_time - start_time}"
    )
    print(
        f"Player {player2.username} won {player2.n_won_battles} out of {player2.n_finished_battles} played\nTime elapsed with {n_battles} battles: {end_time - start_time}"
    )

def save_parameters(heuristic_bot):
    dicc_par = {}
    dicc_par['par_stats'] = heuristic_bot.par_stats
    dicc_par['par_typing'] = heuristic_bot.par_typing
    dicc_par['par_hp'] = heuristic_bot.par_hp
    dicc_par['par_status'] = heuristic_bot.par_status
    dicc_par['par_weather'] = heuristic_bot.par_weather

    return dicc_par

if __name__ == "__main__":
    main()

