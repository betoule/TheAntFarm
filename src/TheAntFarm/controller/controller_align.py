from PySide2.QtCore import QObject
import qimage2ndarray
from double_side_manager import DoubleSideManager
from shape_core import alignment_solver
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AlignController(QObject):
    def __init__(self, settings):
        super(AlignController, self).__init__()
        self.settings = settings

        self.double_side_manager = DoubleSideManager()
        self.threshold_value = 0
        self.alignment_finder = None

        self.current_transform = None
        
    def update_threshold_value(self, new_threshold):
        self.threshold_value = new_threshold

    def update_drills(self, excellon, front_side):
        drills = np.rec.fromrecords([p.position for p in excellon.primitives], names=['x', 'y'])
        self.alignment_finder = alignment_solver.AlignmentFinder(drills, front=front_side)
        logger.info(f'Alignment finder loaded with {len(self.alignment_finder.quadrangle_map)} hash candidates.')
        
    def camera_new_frame(self):
        frame = self.double_side_manager.get_webcam_frame()
        logger.debug(str(self.threshold_value))
        frame, results = self.double_side_manager.detect_holes(frame, self.threshold_value)
        if self.alignment_finder is not None and len(results) > 4:
            holes = np.rec.fromrecords(results, names=['x', 'y', 'radius'])
            try:
                transform, index = self.alignment_finder.find_transform(holes)
                self.current_transform = transform
                if transform is not None:
                    self.alignment_finder.show_transform(holes, transform, index, frame)

            except Exception as e:
                import pdb
                pdb.set_trace()
        image = qimage2ndarray.array2qimage(frame)
        return image

    def flip_side(self, front_side):
        if self.alignment_finder is not None:
            self.alignment_finder.set_side(front_side)
        
