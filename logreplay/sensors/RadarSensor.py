import weakref
import carla
import numpy as np
import os
from logreplay.sensors.base_sensor import BaseSensor
from opencood.hypes_yaml.yaml_utils import save_yaml_wo_overwriting


class RadarSensor(BaseSensor):
    def __init__(self, agent_id, vehicle, world, config, global_position):
        super().__init__(agent_id, vehicle, world, config, global_position)

        if vehicle is not None:
            world = vehicle.get_world()

        self.agent_id = agent_id

        blueprint = world.get_blueprint_library().find('sensor.other.radar')
        # Configure radar attributes based on configuration
        blueprint.set_attribute('horizontal_fov', str(config['horizontal_fov']))
        blueprint.set_attribute('vertical_fov', str(config['vertical_fov']))
        blueprint.set_attribute('range', str(config['range']))
        blueprint.set_attribute('points_per_second', str(config['points_per_second']))
        blueprint.set_attribute('sensor_tick', str(config['sensor_tick']))

        relative_position = config['relative_pose']
        self.name = 'radar' + str(relative_position)

        relative_location = carla.Location(x=config['position'][0],
                                           y=config['position'][1],
                                           z=config['position'][2])
        relative_rotation = carla.Rotation(yaw=config['rotation'][0],
                                           pitch=config['rotation'][1],
                                           roll=config['rotation'][2])
        relative_position = carla.Transform(relative_location, relative_rotation)

        spawn_point = self.spawn_point_estimation(vehicle, relative_position)

        if vehicle is not None:
            self.sensor = world.spawn_actor(blueprint, relative_position, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # Initialize radar data attributes
        self.detections = None
        self.timestamp = None
        self.frame = 0

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: RadarSensor._on_data_event(weak_self, event))

    @staticmethod
    def _on_data_event(weak_self, event):
        """Radar data event handler"""
        self = weak_self()
        if not self:
            return

        # Extract radar detection data
        data = np.frombuffer(event.raw_data, dtype=np.float32).reshape(-1, 4)  # (velocity, azimuth, altitude, depth)

        # Convert detections to XYZ points
        x = data[:, 3] * np.cos(data[:, 2]) * np.cos(data[:, 1])
        y = data[:, 3] * np.cos(data[:, 2]) * np.sin(data[:, 1])
        z = data[:, 3] * np.sin(data[:, 2])
        velocity = data[:, 0]

        # Store points in (x, y, z, velocity) format for PCD
        self.detections = np.column_stack((x, y, z, velocity))
        self.frame = event.frame
        self.timestamp = event.timestamp


    def on_tick(self):
        """Process radar data at each simulation tick"""
        if self.detections is None:
            return
        # Process the data as needed for real-time applications or visualization
        #print(f"Processing radar data at frame {self.frame} with timestamp {self.timestamp}")

    @staticmethod
    def spawn_point_estimation(vehicle, relative_position):
        # Get the current transform of the ego vehicle
        ego_transform = vehicle.get_transform()
        location = ego_transform.location
        rotation = ego_transform.rotation

        # Set relative positioning for radar sensors
        spawn_point_location = ego_transform.transform(relative_position.location)
        spawn_point_rotation = carla.Rotation(
            pitch=rotation.pitch + relative_position.rotation.pitch,
            yaw=rotation.yaw + relative_position.rotation.yaw,
            roll=rotation.roll + relative_position.rotation.roll
        )
        spawn_point = carla.Transform(spawn_point_location, spawn_point_rotation)
        return spawn_point

    def data_dump(self, output_folder, timestamp):
        """Dump radar data to a PCD file."""
        if self.detections is None:
            #print("No radar data to dump")
            return

        # Create the file path with the .npy extension
        file_path = os.path.join(output_folder, f"{timestamp}_{self.name}.npy")

        try:
            # Save detections as a Numpy array
            np.save(file_path, self.detections)
            #print(f"Radar data dumped to {file_path}")
        except Exception as e:
            print(f"Failed to dump radar data to {file_path}: {e}")

        # save pose yaml
        file_path_pose_yaml = os.path.join(output_folder, timestamp + '.yaml')
        yaml_data = {
            "sensors": {}
        }
        #print("Self Sensors:", self.sensor)
        #print("Self Name:", self.name)
        radar_transform = self.sensor.get_transform()
        # Note: The original yaml files without x,y,z,pitch,yaw,roll as key elements in dictionary
        # Compare thew original yaml file with the radar yaml files. Then you will understand the difference
        yaml_data["sensors"][f"{self.name}"] = [
            radar_transform.location.x,
            radar_transform.location.y,
            radar_transform.location.z,
            radar_transform.rotation.roll,
            radar_transform.rotation.yaw,
            radar_transform.rotation.pitch
        ]
        #print("Yaml Data:", yaml_data)
        try:
            save_yaml_wo_overwriting(yaml_data["sensors"], file_path_pose_yaml)
            #print(f"Pose and transform data saved to {file_path_pose_yaml}")

        except Exception as e:
            print(f"Failed to dump radar data to {file_path_pose_yaml}: {e}")

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()