import numpy as np
from gymnasium.spaces import Space, Box
from poke_env.data import GenData
from poke_env.player import Gen9EnvSinglePlayer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

class SimpleRLPlayer(Gen9EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        if (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.agent.create_order(battle.available_moves[action])
        elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)
        ):
            return self.agent.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.agent.create_order(
                battle.available_moves[action - 8], mega=True
            )
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.agent.create_order(
                battle.available_moves[action - 12], dynamax=True
            )
        elif (
            battle.can_tera
            and 0 <= action - 16 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.agent.create_order(
                battle.available_moves[action - 16], terastallize=True
            )
        elif 0 <= action - 20 < len(battle.available_switches):
            return self.agent.create_order(battle.available_switches[action - 20])
        else:
            return self.agent.choose_random_move(battle)

    def embed_battle(self, battle : AbstractBattle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(8).type_chart
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )