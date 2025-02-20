import simpy
from MTGP_KNN.GPFC import shopfloor


# if __name__ == "__main__":
def test_run():
    env = simpy.Environment()
    spf = shopfloor(
        env,
        1000,
        6,
        3,
        None,
        None,
        routing_rule="GP_evolve_R",
        sequencing_rule="GP_evolve_S",
        seed=0,
        ifPrint=False,
        dataset_name="HH",
    )
    spf.simulation(until=3)
    print(f"After run, time: {spf.get_env.now}, queue: {spf.get_env._queue}")
