from pypcd import pypcd


class PointCloud:
    def __init__(self, pc):
        self.pc_data = pc.pc_data

    @classmethod
    def from_path(cls, path):
        return cls(pypcd.PointCloud.from_path(path))
