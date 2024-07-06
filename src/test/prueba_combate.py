import sys
import asyncio
import textwrap
from poke_env import AccountConfiguration,Player,RandomPlayer
from prueba_max_bot import MaxDamagePlayer
from prueba_heuristic_bot import HeuristicPlayer


#battle_format="gen9vgc2024regg"

first_player = RandomPlayer()
second_player = RandomPlayer()
max_bot = MaxDamagePlayer()
heuristic_bot = HeuristicPlayer()

def main():
    option : int = 0
    try:
        while option != range(1,4):
            print(
                textwrap.dedent(
                    """
                    Select the option you want to test:
                    1. Random player vs Random player
                    2. Max damage player vs Random player
                    3. Heuristic player vs Random player
                    4. Heuristic player vs Max damage player
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
                asyncio.run(create_battle(heuristic_bot, max_bot,n_battles))
            else:
                print('Error: Invalid option')
    except Exception as e:
        print(f'Error: {e}\nMONDONGO\n')
        sys.exit(1)


async def create_battle(player1, player2, n_battles):
    await player1.battle_against(player2, n_battles=n_battles)
    print(
        f"Player {player1.username} won {player1.n_won_battles} out of {player1.n_finished_battles} played"
    )
    print(
        f"Player {player2.username} won {player2.n_won_battles} out of {player2.n_finished_battles} played"
    )
    
if __name__ == "__main__":
    main()