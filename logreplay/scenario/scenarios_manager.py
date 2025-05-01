import os
import sys
import time
import gc
import psutil
import subprocess
from collections import OrderedDict
from opencood.hypes_yaml.yaml_utils import load_yaml
from logreplay.scenario.scene_manager import SceneManager


def init_carla():
    """
    Initializes Carla subprocess
    """
    try:
        carla_subprocess = subprocess.Popen(
            ["/home/ws-ids-es3-01/PycharmProjects/advercitydataset/CARLA_0.9.12/CarlaUE4.sh"]
        )
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("-" * 40)
        print(f"CarlaUE4.sh NOT FOUND!\nChange the path to CarlaUE4.sh on:\n{fname}, line {exc_tb.tb_lineno}")
        print("-" * 40)
        exit(1)

    # Sleep for Carla to initialize properly
    time.sleep(10)

    carla_pid = carla_subprocess.pid
    for proc in psutil.process_iter(["pid", "name", "username"]):
        if proc.info["pid"] == carla_pid:
            return proc.info["username"]

    return None


def kill_carla(user):
    """
    Kills Carla process
    """
    for proc in psutil.process_iter(["pid", "name", "username"]):
        try:
            if user and proc.info["username"] != user:
                continue

            if "CarlaUE4-Linux-Shipping" in proc.info["name"]:
                proc.kill()
                proc.wait()
                print(f"Process with PID {proc.info['pid']} ({proc.info['name']}) terminated.")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    time.sleep(10)


def restart_carla(user):
    """
    Kills Carla and restarts it
    """
    kill_carla(user)
    gc.collect()
    return init_carla()

class ScenariosManager:
    """
    Format all scenes in a structured way.

    Parameters
    ----------
    scenario_params: dict
        Overall parameters for the replayed scenes.

    Attributes
    ----------

    """

    def __init__(self, scenario_params):
        # this defines carla world sync mode, weather, town name, and seed.
        self.scene_params = scenario_params

        # e.g. /opv2v/data/train
        root_dir = self.scene_params['root_dir']

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        self.scenario_database = OrderedDict()

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            scene_name = os.path.split(scenario_folder)[-1]
            self.scenario_database.update({scene_name: OrderedDict()})

            # load the collection yaml file
            protocol_yml = [x for x in os.listdir(scenario_folder)
                            if x.endswith('.yaml')]
            collection_params = load_yaml(os.path.join(scenario_folder,
                                                       protocol_yml[0]))

            # create the corresponding scene manager
            cur_sg = SceneManager(scenario_folder,
                                  scene_name,
                                  collection_params,
                                  scenario_params)
            self.scenario_database[scene_name].update({'scene_manager':
                                                       cur_sg})

    def tick(self):
        """
        Tick for every scene manager to do the log replay.
        """
        # Initialize Carla at the start
        carla_user = init_carla()
        print("Carla started. Waiting additional 10 seconds...")
        time.sleep(10)


        for scene_name, scene_content in self.scenario_database.items():
            print('log replay %s' % scene_name)
            scene_manager = scene_content['scene_manager']
            run_flag = True

            scene_manager.start_simulator()

            while run_flag:
                run_flag = scene_manager.tick()

            scene_manager.close()
            time.sleep(3)

            print(f"Replay for {scene_name} completed. Restarting Carla...")
            kill_carla(carla_user)
            print("Waiting 60 seconds before restarting Carla...")
            time.sleep(30)
            carla_user = init_carla()

            # Kill Carla after all scenarios are done
        print("All scenarios completed. Killing Carla...")
        kill_carla(carla_user)


if __name__ == '__main__':
    from opencood.hypes_yaml.yaml_utils import load_yaml
    scene_params = load_yaml('/home/ws-ids-es3-01/PycharmProjects/hamdard_bm2cp/logreplay/hypes_yaml/replay.yaml')
    scenarion_manager = ScenariosManager(scenario_params=scene_params)
    scenarion_manager.tick()
    print('test passed')



