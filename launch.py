from environment import Environment
from parameter import Parameter
from mac_generator import MacGenerator
from job_generator import JobGenerator
from element import Machine, Job
from agent import ecs_agent, ecs_dp_agent, ecs_ml_agent, swarm_agent, pack_agent, k8s_agent
import plot

def run(agent):

    Machine.reset()
    Job.reset()

    pa = Parameter()
    pa.agent = agent
    env = Environment(pa)

    mac_gen = MacGenerator(pa)
    job_gen = JobGenerator(pa)


    for i in mac_gen.mac_sequence:
        env.add_machine(i)

    job_idx = 0
    current_time = 0




    while True:


        env.step()
        env.time_log()

        while ( env.job_count < pa.job_queue_num and
                job_gen.job_sequence[job_idx] is not None and
                job_gen.job_sequence[job_idx].submission_time <= current_time
        ):
            if (job_gen.job_sequence[job_idx].submission_time == current_time):   #  add job to environment
                env.add_job(job_gen.job_sequence[job_idx])
            job_idx += 1

        if agent == "ecs":
            ecs_agent.schedule(env)
        elif agent =="k8s":
            k8s_agent.schedule(env)
        elif agent == "pack":
            pack_agent.schedule(env)
        elif agent == "ecs_dp":
            ecs_dp_agent.schedule(env)
        elif agent == "ecs_ml":
            ecs_ml_agent.schedule(env)
        elif agent == "swarm":
            swarm_agent.schedule(env)

        if job_gen.job_sequence[job_idx] is None:
            if env.status() == "Idle": # finish all jobs
                break
        current_time += 1

    env.job_log()
    env.finish()

def main():
    run("ecs_dp")
    run("ecs_ml")
    run("k8s")
    run("swarm")
    plot.run()
    # run("pack")

if __name__ == "__main__":
    main()