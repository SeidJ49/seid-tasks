from opencood.models.pillarnet_radar_student_kd import PillarnetRadarStudentKd


class PillarnetSingleRadarBaseline(PillarnetRadarStudentKd):
    """WP1 PillarNet radar-only baseline.

    Reuses the WP3 radar PillarNet encoder/head without requiring a teacher or
    KD loss. The WP1 YAML pairs this model with ``radardistill_loss`` so only
    the detector loss returned by the model is optimized.
    """

    pass
