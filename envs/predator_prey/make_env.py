from .environment import MultiAgentEnv
from .scenarios import load


def pp_make_env(args, benchmark=False):

    # load scenario from script
    scenario = load(args.scenario + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
        )
    else:
        env = MultiAgentEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation
        )
    return env
